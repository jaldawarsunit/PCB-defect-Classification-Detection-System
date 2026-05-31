import cv2
import os
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IMAGE_BASE = ROOT / "dataset/PCB_DATASET/images"
ANNOT_BASE = ROOT / "dataset/PCB_DATASET/Annotations"
SAVE_BASE  = ROOT / "output/xml_visualization"

MAX_IMAGES_PER_CLASS = 20  

def visualize_from_xml():

    total_images = 0

    for class_folder in ANNOT_BASE.iterdir():

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        image_folder = IMAGE_BASE / class_name
        save_folder  = SAVE_BASE / class_name
        os.makedirs(save_folder, exist_ok=True)

        count = 0

        for xml_file in class_folder.glob("*.xml"):

            if count >= MAX_IMAGES_PER_CLASS:
                break

            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.find("filename").text
            image_path = image_folder / filename

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            for obj in root.findall("object"):

                label = obj.find("name").text
                bbox  = obj.find("bndbox")

                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                text_y = ymin - 10 if ymin - 10 > 10 else ymin + 20

                cv2.putText(
                    image,
                    label,
                    (xmin, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            cv2.imwrite(str(save_folder / filename), image)

            count += 1
            total_images += 1

    print(f"Visualization completed. Total images saved: {total_images}")


if __name__ == "__main__":
    visualize_from_xml()