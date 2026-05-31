# PCB Defect Detection and Classification System

## Project Overview

PCB Defect Detection and Classification System is an automated web-based
application that detects and classifies defects in Printed Circuit Boards (PCBs)
using image processing and deep learning.

The system takes two images as input — a defect-free template image and a test
PCB image. It subtracts the two images to find differences, applies thresholding
and contour detection to locate defect regions, and then uses a trained
EfficientNet-B0 deep learning model to classify each defect into one of 6
categories:

Missing Hole | Mouse Bite | Open Circuit | Short | Spur | Spurious Copper

The system achieves above 95% classification accuracy and processes results
in under 3 seconds. Results are displayed on a Flask web interface with
annotated bounding boxes, an on-screen prediction log table, and options
to download the result image and CSV log.

## Defect Categories

The model classifies defects into 6 categories:

1. Missing Hole
2. Mouse Bite
3. Open Circuit
4. Short
5. Spur
6. Spurious Copper

## Tech Stack

| Area             | Tools / Libraries                                     |
|------------------|-------------------------------------------------------|
| Image Processing | OpenCV, NumPy                                         |
| Deep Learning.   | PyTorch, TorchVision                                  |
| Model            | EfficientNet-B0 (Transfer Learning)                   |
| Dataset          | DeepPCB                                               |
| Frontend         | HTML, CSS, JavaScript                                 |
| Backend          | Python, Flask                                         |
| Evaluation       | Accuracy, Loss, Confusion Matrix                      |
| Export           | CSV, Annotated Image                                  |


## Project Structure
```text

PCB-Defect-Detection-and-Classification-System/
│
├── app.py                   # Flask backend - main application
│
├── scripts/
│   ├── predict_defects1.py  # Core prediction pipeline
│   ├── train_model.py       # Model training script
│   ├── test_model.py        # Model evaluation script
│   ├── roi_extraction.py    # ROI extraction from contours
│   ├── subtraction.py       # Image subtraction pipeline
│   ├── augment_rois.py      # Data augmentation
│   ├── split_dataset.py     # Dataset splitting
│   ├── confusion_matrix.py  # Confusion matrix generation
│   └── visualization_xml.py # Visualization utilities
│
├── static/
│   ├── css/style.css        # Frontend styling
│   ├── js/script.js         # Frontend logic
│   └── images/bg.jpg        # Background image
│
├── PCB DATASET              # Dataset From Kaggle
│
├── templates/
│   └── index.html           # Main web interface
│
└──pcb_defect_model.pth.     # Tranined Model

```

## How It Works

1. User uploads a **Template Image** (defect-free PCB) and a **Test Image** (PCB to inspect)
2. Images are subtracted to generate a **difference map**
3. **Otsu thresholding** and morphological operations isolate defect regions
4. **Contour detection** extracts individual defect ROIs with bounding boxes
5. Each ROI is classified by **EfficientNet-B0** into one of 6 defect categories
6. Annotated output image and prediction log are returned to the user

## Model

| Parameter     | Value              |
|---------------|--------------------|
| Architecture  | EfficientNet-B0    |
| Optimizer     | Adam               |
| Loss Function | Cross-Entropy Loss |
| Input Size    | 128×128            |
| Test Accuracy | Above 95%          |
| Dropout       | 0.4                |
| Classes       | 6                  |

## Features

- Upload template and test PCB image pairs via browser
- Real-time defect detection and classification
- Annotated output image with bounding boxes and defect labels
- On-screen prediction log table (Defect Label, Confidence, X, Y, Width, Height)
- Download annotated result image (JPG)
- Download prediction log as CSV file

## Installation and Usage

### 1. Clone the repository
``` bash
git clone https://github.com/jaldawarsunit/PCB-defect-Classification-Detection-System.git
cd PCB-Defect-Detection-and-Classification-System
```

### 2. Install dependencies
``` bash
pip install -r requirement.txt
```

### 3. Run the application
``` bash
python app.py
```

### 4. Open in browser
http://localhost:5000

## Dataset

While Selecting: Template: 
1. Go To PCB_dataset -> PCB_USED -> 01.jpg
similar to 10 images.

While Selecting: Raw Image:
1. Go To PCB_dataset -> images -> Missing_hole -> 01_missing_hole_01.jpg

warning: template first number should be match with defects stating number;
warning: best tranined Model Is Share, This Model Is Tranined On Mac M2
         similar to 10 images.


This project uses the **DeepPCB Dataset** which contains paired template
and test PCB images annotated with 6 defect categories.

## Author
**Sunit Jaldawar**
PCB Defect Detection and Classification System
