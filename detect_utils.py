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
    os.makedirs(output_dir, exist_ok=True)

    results = model.predict(
        source=image_path,
        imgsz=512,      # if still crashing -> 416
        conf=0.25,
        device="cpu",
        verbose=False
    )
    result = results[0]

    im = result.plot()
    im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if im.shape[-1] == 3 else im

    out_name = f"{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, im_bgr)

    labels = []
    for box in result.boxes:
        cls_idx = int(box.cls)
        labels.append(model.names[cls_idx])

    labels = sorted(set(labels))  # optional

    out_path = out_path.replace("\\", "/")
    return out_path, labels

