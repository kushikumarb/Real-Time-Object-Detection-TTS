import cv2
import time
import logging
import os
from typing import Union
from urllib.request import urlretrieve


def setup_logger(name: str = "object_detection") -> logging.Logger:
    """Configure logging with file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # File handler
    fh = logging.FileHandler("detection.log")
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def calculate_fps(start_time: float, end_time: float) -> float:
    """Calculate frames per second."""
    return 1.0 / max(0.0001, (end_time - start_time))  # Prevent division by zero


def draw_fps(frame: cv2.Mat, fps: float) -> None:
    """Draw FPS counter on frame."""
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def validate_model(model_path: str) -> bool:
    """Check or download YOLO model."""
    if os.path.exists(model_path):
        return True

    try:
        model_name = os.path.basename(model_path)
        url = f"https://github.com/ultralytics/assets/releases/download/v8.1.0/{model_name}"
        urlretrieve(url, model_path)
        return True
    except Exception as e:
        logging.error(f"Model download failed: {e}")
        return False
