import os
import random
import shutil
from pathlib import Path

SOURCE = Path("dataset_augmented_rois")
DEST = Path("dataset_split")

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for class_folder in SOURCE.iterdir():

    images = list(class_folder.glob("*.jpg"))
    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for subset, subset_images in zip(
        ["train", "val", "test"],
        [train_images, val_images, test_images]
    ):

        save_dir = DEST / subset / class_folder.name
        os.makedirs(save_dir, exist_ok=True)

        for img in subset_images:
            shutil.copy(img, save_dir)

print("Dataset split completed.")