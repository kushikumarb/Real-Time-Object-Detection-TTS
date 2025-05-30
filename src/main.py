import argparse
import cv2
import time
from detector import ObjectDetector
from tts_engine import TTSEngine
from utils import setup_logger, calculate_fps, draw_fps


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--lang", default="english", help="TTS language")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    args = parser.parse_args()

    # Initialize components
    logger = setup_logger()
    detector = ObjectDetector(args.model, args.conf)
    tts = TTSEngine(args.lang)

    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    logger.info("Starting detection loop...")
    while True:
        start_time = time.time()

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            break

        # Detection and TTS
        results = detector.detect(frame)
        tts.announce(results)

        # Display FPS
        fps = calculate_fps(start_time, time.time())
        draw_fps(results.plot(), fps)

        # Show output
        cv2.imshow("Object Detection", results.plot())
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Application stopped")


if __name__ == "__main__":
    main()
