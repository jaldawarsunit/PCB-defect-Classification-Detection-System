import cv2
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MASK_BASE = ROOT / "output/threshold"
ORIG_BASE = ROOT / "dataset/PCB_DATASET/images"
ROI_BASE  = ROOT / "output/roi_dataset"

AREA_THRESHOLD = 120     
PADDING = 25             

FOLDER_MAP = {
    "missing_hole": "Missing_hole",
    "mouse_bite": "Mouse_bite",
    "open_circuit": "Open_circuit",
    "short": "Short",
    "spur": "Spur",
    "spurious_copper": "Spurious_copper"
}

def extract_rois():
    roi_index = 0

    for defect_folder in MASK_BASE.iterdir():

        defect_lower = defect_folder.name
        defect_upper = FOLDER_MAP[defect_lower]

        save_dir = ROI_BASE / defect_upper
        os.makedirs(save_dir, exist_ok=True)

        for mask_path in defect_folder.glob("*.jpg"):

            mask = cv2.imread(str(mask_path), 0)
            if mask is None:
                continue

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            parts = mask_path.stem.split("_")[:-1]
            orig_name = "_".join(parts) + ".jpg"

            orig_path = ORIG_BASE / defect_upper / orig_name
            orig_img = cv2.imread(str(orig_path))
            if orig_img is None:
                continue

            img_h, img_w = orig_img.shape[:2]

            for cnt in contours:

                if cv2.contourArea(cnt) < AREA_THRESHOLD:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                x1 = max(x - PADDING, 0)
                y1 = max(y - PADDING, 0)
                x2 = min(x + w + PADDING, img_w)
                y2 = min(y + h + PADDING, img_h)

                roi = orig_img[y1:y2, x1:x2]

                save_path = save_dir / f"{orig_name[:-4]}_roi_{roi_index}.jpg"
                cv2.imwrite(str(save_path), roi)

                roi_index += 1

    print("ROI extraction completed successfully")

if __name__ == "__main__":
    extract_rois()