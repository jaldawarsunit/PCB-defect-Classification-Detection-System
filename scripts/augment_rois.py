import cv2
import numpy as np
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "output/roi_from_xml"
DEST   = ROOT / "dataset_augmented_rois"

AUG_PER_IMAGE = 3   

def rotate(img):
    angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def flip(img):
    return cv2.flip(img, 1)

def brightness(img):
    value = random.randint(-20, 20)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)

    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def noise(img):
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def zoom(img):
    h, w = img.shape[:2]
    scale = random.uniform(1.05, 1.1)

    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))

    startx = (nw - w) // 2
    starty = (nh - h) // 2

    return resized[starty:starty+h, startx:startx+w]


AUG_FUNCTIONS = [rotate, flip, brightness, noise, zoom]


def augment_dataset():

    print("Starting ROI augmentation...")

    for class_folder in SOURCE.iterdir():

        if not class_folder.is_dir():
            continue

        save_folder = DEST / class_folder.name
        os.makedirs(save_folder, exist_ok=True)

        for img_path in class_folder.glob("*.jpg"):

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            cv2.imwrite(str(save_folder / img_path.name), img)

            for i in range(AUG_PER_IMAGE):

                aug = img.copy()

                func = random.choice(AUG_FUNCTIONS)
                aug = func(aug)

                new_name = f"{img_path.stem}_aug_{i}.jpg"
                cv2.imwrite(str(save_folder / new_name), aug)

    print("ROI augmentation completed successfully.")


if __name__ == "__main__":
    augment_dataset()