"""
new_draw_bb.py  -  draw coordinates.csv boxes onto the test image
=================================================================
coordinates.csv (written by new_infer.c on the board) is in CANVAS-pixel
coordinates, so we resize the original image to CANVAS x CANVAS (the same
squash new_img2bin.py / new_train.py use) and draw there. Output is a JPG.

CSV columns:  x_min,y_min,x_max,y_max,score,class_id,class_name

Usage:
    python3 new_draw_bb.py test.jpg                          # reads coordinates.csv
    python3 new_draw_bb.py test.jpg coordinates.csv out.jpg 384
"""
import sys, os, csv
from PIL import Image, ImageDraw, ImageFont

CANVAS_DEFAULT = 384   # must match model_arch.h / new_img2bin.py

# per-class colors (small-vehicle, large-vehicle, human)
COLORS = [(0, 220, 0), (0, 128, 255), (255, 64, 64),
          (255, 200, 0), (200, 0, 255), (0, 255, 220)]


def load_dets(csv_path):
    dets = []
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found"); sys.exit(1)
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            row = [c.strip() for c in row if c.strip() != ""]
            if len(row) < 5:
                continue
            try:
                x1, y1, x2, y2 = (int(float(row[0])), int(float(row[1])),
                                  int(float(row[2])), int(float(row[3])))
                score = float(row[4])
                cls_id = int(float(row[5])) if len(row) > 5 else 0
                name = row[6] if len(row) > 6 else str(cls_id)
            except ValueError:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append((x1, y1, x2, y2, score, cls_id, name))
    return dets


def main():
    if len(sys.argv) < 2:
        print("[ERROR] Usage: python3 new_draw_bb.py <image> [coordinates.csv] [out.jpg] [canvas]")
        sys.exit(1)
    img_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else "coordinates.csv"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "detections.jpg"
    canvas = int(sys.argv[4]) if len(sys.argv) > 4 else CANVAS_DEFAULT

    if not os.path.exists(img_path):
        print(f"[ERROR] image not found: {img_path}"); sys.exit(1)

    img = Image.open(img_path).convert("RGB").resize((canvas, canvas))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    dets = load_dets(csv_path)
    print(f"[*] {len(dets)} boxes from {csv_path}")
    for (x1, y1, x2, y2, score, cls_id, name) in dets:
        color = COLORS[cls_id % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{name} {score:.2f}"
        ty = max(0, y1 - 11)
        try:
            tw = draw.textlength(label, font=font)
        except Exception:
            tw = 8 * len(label)
        draw.rectangle([x1, ty, x1 + tw + 2, ty + 11], fill=color)
        draw.text((x1 + 1, ty), label, fill=(0, 0, 0), font=font)

    img.save(out_path, quality=92)
    print(f"[SUCCESS] wrote {out_path}  ({canvas}x{canvas}, {len(dets)} boxes)")


if __name__ == "__main__":
    main()
