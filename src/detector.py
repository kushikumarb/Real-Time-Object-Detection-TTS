from ultralytics import YOLO
import torch
from utils import validate_model


class ObjectDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        if not validate_model(model_path):
            raise FileNotFoundError(f"Model {model_path} not found")

        self.model = YOLO(model_path)
        self.conf = conf_threshold

    def detect(self, frame):
        return self.model.predict(
            frame,
            conf=self.conf,
            iou=0.45,
            half=torch.cuda.is_available(),
            device="0" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )[0]
