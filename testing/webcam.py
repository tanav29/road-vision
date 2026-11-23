from pathlib import Path

import cv2
from ultralytics import YOLO

from detection import process_frame


def main():
    model = YOLO(str(Path(__file__).with_name("model.pt")))
    cap = cv2.VideoCapture("./videos/dashcam3.mp4")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = process_frame(frame, model)
            window_name = "Smart Dashcam System"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, annotated_frame)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
