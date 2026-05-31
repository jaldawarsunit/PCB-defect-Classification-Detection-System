import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "pcb_defect_model.pth"

AREA_THRESHOLD = 50
PADDING = 25

classes = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once
model = models.efficientnet_b0(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 6)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def predict_defects(template_img, test_img):

    # Fresh log per call — avoids accumulation across multiple image pairs
    log_data = []

    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    gray_test = cv2.resize(gray_test, (gray_template.shape[1], gray_template.shape[0]))

    gray_template = cv2.GaussianBlur(gray_template, (5, 5), 0)
    gray_test = cv2.GaussianBlur(gray_test, (5, 5), 0)

    diff = cv2.absdiff(gray_template, gray_test)

    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = test_img.copy()
    img_h, img_w = test_img.shape[:2]

    for cnt in contours:

        if cv2.contourArea(cnt) < AREA_THRESHOLD:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if x < 10 or y < 10 or (x + w) > img_w - 10 or (y + h) > img_h - 10:
            continue

        x1 = max(x - PADDING, 0)
        y1 = max(y - PADDING, 0)
        x2 = min(x + w + PADDING, img_w)
        y2 = min(y + h + PADDING, img_h)

        roi = test_img[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        roi_tensor = transform(roi).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            prediction = model(roi_tensor)
            prob = torch.softmax(prediction, dim=1)

            confidence, predicted = torch.max(prob, 1)

        label = classes[predicted.item()]
        log_data.append([
            label,
            round(confidence.item(), 2),
            x1, y1, x2 - x1, y2 - y1
        ])

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(
            output,
            f"{label} {confidence.item():.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    return output, log_data