import sys
import os
import numpy as np
from PIL import Image

def main():
    print("=================================================")
    print("  RAILWAY DEFECT: IMAGE TO RAW BINARY CONVERTER  ")
    print("=================================================")

    if len(sys.argv) < 2:
        print("[ERROR] Usage: python3 img2bin.py /path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = "defective.bin"
    target_sz = 64

    if not os.path.exists(img_path):
        print(f"[ERROR] File not found: {img_path}")
        sys.exit(1)

    try:
        # 1. Load image, force RGB (drops alpha channel if PNG), and resize to 64x64
        print(f"[*] Loading and resizing '{os.path.basename(img_path)}' to {target_sz}x{target_sz}...")
        img = Image.open(img_path).convert("RGB").resize((target_sz, target_sz))

        # 2. Convert to NumPy array (PIL naturally creates HWC layout)
        arr = np.asarray(img, dtype=np.uint8)

        # 3. Verify dimensions and size
        expected_size = target_sz * target_sz * 3
        actual_size = arr.nbytes
        
        if actual_size != expected_size:
            print(f"[ERROR] Size mismatch. Expected {expected_size} bytes, got {actual_size}.")
            sys.exit(1)

        # 4. Dump raw bytes to file
        arr.tofile(out_path)
        print(f"[SUCCESS] Wrote exactly {actual_size} bytes to '{out_path}'.")
        print(f"[*] Ready to be sent to the FPGA!")

    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}")

if __name__ == "__main__":
    main()
