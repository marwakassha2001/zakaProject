import os
import uuid

import cv2
from ultralytics import YOLO


def load_model(weights_path="weights/best.pt"):
    """
    Load YOLO model from weights file.
    """
    model = YOLO(weights_path)
    return model


def run_detection(model, image_path, output_dir="static/outputs"):
    """
    Run detection on an image and save annotated result.

    Returns:
        result_path (str): path to saved annotated image
        labels (list[str]): list of detected class names
    """
    os.makedirs(output_dir, exist_ok=True)

  
    results = model(image_path)
    result = results[0]

    # Plot detections to an array (BGR image)
    im = result.plot()

    # Save annotated image
    out_name = f"{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, im)

    # Extract labels
    labels = []
    for box in result.boxes:
        cls_idx = int(box.cls)
        labels.append(model.names[cls_idx])

    # Make path web-friendly
    out_path = out_path.replace("\\", "/")
    return out_path, labels
