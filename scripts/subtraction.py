import cv2
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE_DIR = ROOT / "dataset" / "PCB_DATASET"
TEMPLATE_DIR = BASE_DIR / "PCB_USED"
DEFECT_DIR = BASE_DIR / "images"

OUTPUT_SUB = ROOT / "output/subtraction"
OUTPUT_THR = ROOT / "output/threshold"

def load_pairs():
    template_files = list(TEMPLATE_DIR.glob("*.jpg"))
    defect_files = list(DEFECT_DIR.rglob("*.jpg"))

    template_map = {t.stem: t for t in template_files}
    pairs = []

    for defect_path in defect_files:
        board_id = defect_path.stem.split("_")[0]
        if board_id in template_map:
            pairs.append((template_map[board_id], defect_path))

    print(f"Total image pairs found: {len(pairs)}")
    return pairs

def process_pair(template_path, defect_path):
    name = defect_path.stem
    parts = name.split("_")
    defect_type = "_".join(parts[1:-1])

    sub_dir = OUTPUT_SUB / defect_type
    thr_dir = OUTPUT_THR / defect_type
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(thr_dir, exist_ok=True)

    template = cv2.imread(str(template_path))
    defect   = cv2.imread(str(defect_path))

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    defect_gray   = cv2.cvtColor(defect, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(template_gray, defect_gray)


    blur = cv2.GaussianBlur(diff, (5,5), 0)
    blur = cv2.medianBlur(blur, 5)
    _, blur = cv2.threshold(blur, 25, 255, cv2.THRESH_TOZERO)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite(str(sub_dir / f"{name}_diff.jpg"), diff)
    cv2.imwrite(str(thr_dir / f"{name}_mask.jpg"), thresh)

def main():
    pairs = load_pairs()
    for template_path, defect_path in pairs:
        process_pair(template_path, defect_path)

    print("Subtraction & Thresholding completed")

if __name__ == "__main__":
    main()