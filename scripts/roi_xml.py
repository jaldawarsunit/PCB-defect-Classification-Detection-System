import cv2
import os
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IMAGE_BASE = ROOT / "dataset/PCB_DATASET/images"
ANNOT_BASE = ROOT / "dataset/PCB_DATASET/Annotations"
ROI_BASE   = ROOT / "output/roi_from_xml"

PADDING = 5

def extract_rois_from_xml():

    total_rois = 0

    for class_folder in ANNOT_BASE.iterdir():

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        image_folder = IMAGE_BASE / class_name
        save_folder  = ROI_BASE / class_name
        os.makedirs(save_folder, exist_ok=True)

        for xml_file in class_folder.glob("*.xml"):

            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.find("filename").text
            image_path = image_folder / filename

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            img_h, img_w = image.shape[:2]

            for i, obj in enumerate(root.findall("object")):

                label = obj.find("name").text
                bbox  = obj.find("bndbox")

                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                xmin = max(0, xmin - PADDING)
                ymin = max(0, ymin - PADDING)
                xmax = min(img_w, xmax + PADDING)
                ymax = min(img_h, ymax + PADDING)

                roi = image[ymin:ymax, xmin:xmax]

                roi_name = f"{filename[:-4]}_roi_{i}.jpg"
                cv2.imwrite(str(save_folder / roi_name), roi)

                total_rois += 1

    print(f"ROI extraction completed. Total ROIs saved: {total_rois}")


if __name__ == "__main__":
    extract_rois_from_xml()