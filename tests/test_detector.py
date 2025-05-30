import cv2
import pytest
from src.detector import ObjectDetector


def test_detector_init():
    detector = ObjectDetector("yolov8n.pt", 0.5)
    assert detector.conf == 0.5


def test_detection():
    detector = ObjectDetector("yolov8n.pt", 0.5)
    test_img = cv2.imread("../assets/output_sample.jpg")
    results = detector.detect(test_img)
    assert len(results.boxes) > 0  
