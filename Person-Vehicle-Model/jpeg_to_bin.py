"""
new_img2bin.py  -  image  ->  raw uint8 .bin for the aerial detector FPGA flow
==============================================================================
Resizes any image to CANVAS x CANVAS (squash, no letterbox -- this MUST match
how new_train.py resizes, so the detector sees the same geometry) and writes
raw uint8 HWC bytes (R,G,B interleaved, row-major).

The byte layout is exactly what new_infer.c expects:
    CANVAS * CANVAS * 3 bytes,  value = pixel/255*... handled in C.

IMPORTANT: CANVAS here MUST equal #define CANVAS in model_arch.h (default 384).

Usage:
    python3 new_img2bin.py /path/to/test.jpg                 # -> test.bin (384x384)
    python3 new_img2bin.py /path/to/test.jpg out.bin 384
"""
import sys, os
import numpy as np
from PIL import Image

CANVAS_DEFAULT = 384   # keep in sync with model_arch.h / new_train.py --canvas


def main():
    print("=================================================")
    print("  AERIAL DETECTOR: IMAGE -> RAW BINARY CONVERTER ")
    print("=================================================")
    if len(sys.argv) < 2:
        print("[ERROR] Usage: python3 new_img2bin.py <image> [out.bin] [canvas]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "test.bin"
    canvas = int(sys.argv[3]) if len(sys.argv) > 3 else CANVAS_DEFAULT

    if not os.path.exists(img_path):
        print(f"[ERROR] File not found: {img_path}")
        sys.exit(1)

    try:
        print(f"[*] Loading '{os.path.basename(img_path)}' and resizing to {canvas}x{canvas} (squash)...")
        img = Image.open(img_path).convert("RGB").resize((canvas, canvas))
        arr = np.asarray(img, dtype=np.uint8)              # HWC, channel-last R,G,B

        expected = canvas * canvas * 3
        if arr.nbytes != expected:
            print(f"[ERROR] Size mismatch. Expected {expected}, got {arr.nbytes}.")
            sys.exit(1)

        arr.tofile(out_path)
        print(f"[SUCCESS] Wrote {arr.nbytes} bytes to '{out_path}'  ({canvas}x{canvas}x3 uint8 HWC).")
        print(f"[*] scp this to the board and run:  ./new_infer {os.path.basename(out_path)} weights.bin biases.bin")
        print(f"[*] Then scp back coordinates.csv and draw with new_draw_bb.py "
              f"(use the SAME --canvas {canvas}).")
    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
