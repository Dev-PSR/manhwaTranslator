#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from deep_translator import GoogleTranslator
import math
import argparse

# ──────────── 1. EAST-BASED TEXT DETECTION ────────────
def decode_east(scores, geometry, score_thresh=0.5, nms_thresh=0.4):
    """
    Decode the output of the EAST detector.
    Returns a list of (x, y, w, h) boxes.
    """
    h, w = scores.shape[2:4]
    boxes = []
    confidences = []
    for y in range(h):
        for x in range(w):
            score = float(scores[0, 0, y, x])
            if score < score_thresh:
                continue
            # geometry: distances to box sides + angle
            d = geometry[0, :4, y, x]
            angle = float(geometry[0, 4, y, x])
            cos = math.cos(angle)
            sin = math.sin(angle)

            offset_x = x * 4.0
            offset_y = y * 4.0
            x0 = offset_x + cos * d[1] + sin * d[2]
            y0 = offset_y - sin * d[1] + cos * d[2]

            w_box = d[1] + d[3]
            h_box = d[0] + d[2]
            x_tl = int(x0 - w_box / 2)
            y_tl = int(y0 - h_box / 2)

            boxes.append((x_tl, y_tl, int(w_box), int(h_box)))
            confidences.append(score)

    # run NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, nms_thresh)
    filtered = []
    if len(indices) > 0:
        for i in indices:
            idx = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
            filtered.append(boxes[idx])
    return filtered


# ──────────── 2. CROP, OCR & TRANSLATE ────────────
def crop_region(img, box):
    x,y,w,h = box
    return img[y:y+h, x:x+w]

def ocr_and_translate(regions, src_img, dest_lang='en'):
    texts = []
    for x, y, w, h in regions:
        crop = src_img[y:y+h, x:x+w]
        txt = pytesseract.image_to_string(crop, lang='kor+kor_vert').strip()
        texts.append(txt)

    translator = GoogleTranslator(source='auto', target=dest_lang)
    # translate one by one (or batch in a comprehension)
    return [translator.translate(t) if t else "" for t in texts]

# ──────────── 3. INPAINT ORIGINAL TEXT ────────────
def make_mask(shape, boxes, pad = 5):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, bw, bh in boxes:
        # expand the box by pad, clamped to image bounds
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask

# ──────────── 4. RENDER TRANSLATED TEXT ────────────
def render_translations(img, boxes, translations, font_path=None):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    for (box, txt) in zip(boxes, translations):
        x, y, w, h = box

        # Auto-fit font size
        font_size = max(12, int(h * 0.6))
        try:
            font = ImageFont.truetype(font_path or "arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        # Wrap lines to fit width
        words = txt.split()
        lines, line = [], ""
        for word in words:
            test = (line + " " + word).strip()
            # measure test string
            bbox = draw.textbbox((0, 0), test, font=font)
            w_test = bbox[2] - bbox[0]
            if w_test > w and line:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)

        # Compute total height and starting y for vertical centering
        total_h = 0
        line_heights = []
        for l in lines:
            bbox = draw.textbbox((0, 0), l, font=font)
            lh = bbox[3] - bbox[1]
            line_heights.append(lh)
            total_h += lh

        y0 = y + (h - total_h) // 2

        # Draw each line centered in the box
        for l, lh in zip(lines, line_heights):
            bbox = draw.textbbox((0, 0), l, font=font)
            wl = bbox[2] - bbox[0]
            x0 = x + (w - wl) // 2
            draw.text((x0, y0), l, fill="black", font=font)
            y0 += lh

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ──────────── MAIN PIPELINE ────────────
def translate_panel(input_path, output_path, east_model, font_path=None):
    # 1. Load original
    img_orig = cv2.imread(input_path)
    H, W = img_orig.shape[:2]

    # 2. Compute a resize that’s divisible by 32
    newW = (W // 32) * 32
    newH = (H // 32) * 32
    # (If you’d rather round up instead of down, use:
    #   newW = ((W + 31) // 32) * 32
    #   newH = ((H + 31) // 32) * 32
    #  but you may need to pad the image then crop detection boxes.)

    # 3. Resize for EAST
    img = cv2.resize(img_orig, (newW, newH))

    # 4. Load EAST and run forward
    net = cv2.dnn.readNet(east_model)
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (newW, newH),
        (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)
    scores, geometry = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    # 5. Decode boxes in the resized coord space
    raw_boxes = decode_east(scores, geometry, score_thresh=0.5)

    # 6. Scale boxes back to original image coords
    ratioW = W / float(newW)
    ratioH = H / float(newH)
    boxes = [
        (
            int(x * ratioW),
            int(y * ratioH),
            int(w * ratioW),
            int(h * ratioH)
        )
        for (x, y, w, h) in raw_boxes
    ]

    # 7. If nothing found, just copy the original
    if not boxes:
        print("✅ No text detected.")
        cv2.imwrite(output_path, img_orig)
        return

    # 8. OCR + Translate on original-resolution crop
    translations = ocr_and_translate(boxes, img_orig, dest_lang='en')

    # 9. Inpaint original text on original-resolution image
    mask = make_mask(img_orig.shape, boxes)
    clean = cv2.inpaint(img_orig, mask, 3, cv2.INPAINT_TELEA)

    # 10. Render translated text back onto the clean original-resolution image
    final = render_translations(clean, boxes, translations, font_path)

    # 11. Save at the exact same resolution as input
    cv2.imwrite(output_path, final)
    print(f"Translated panel saved to {output_path} ({W}×{H})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Translate a manhwa panel in-place via EAST + Tesseract + Google Translate"
    )
    ap.add_argument("input", help="Path to your input panel image")
    ap.add_argument("output", help="Where to write the translated PNG")
    ap.add_argument("--east", required=True,
                    help="Path to frozen_east_text_detection.pb")
    ap.add_argument("--font", default=None,
                    help="(Optional) path to .ttf font for rendering")
    args = ap.parse_args()

    translate_panel(args.input, args.output, args.east, args.font)
