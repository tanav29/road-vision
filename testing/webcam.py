import argparse
from pathlib import Path
import cv2
import time

# import threading
import numpy as np
from ultralytics import YOLO
from lane import detect_lanes, game_lane
from autodrive import Controller, WindowsKeys, decide
# from playsound import playsound

parser = argparse.ArgumentParser(description="Run the dashcam detector on a video or a browser game.")
parser.add_argument("--autodrive", action="store_true", help="Enable experimental Windows game controls (requires --screen)")
parser.add_argument("--screen", action="store_true", help="Select an area of the primary screen to capture")
parser.add_argument("--video", default="./videos/dashcam3.mp4", help="Video path when not using --screen")
parser.add_argument("--fps", type=float, default=5.0, help="Maximum inference frames per second (default: 5)")
parser.add_argument("--imgsz", type=int, default=512, help="YOLO inference image size (default: 512)")
args = parser.parse_args()
if args.autodrive and not args.screen:
    parser.error("--autodrive requires --screen")
if not 1 <= args.fps <= 60:
    parser.error("--fps must be between 1 and 60")
if args.imgsz < 32:
    parser.error("--imgsz must be at least 32")

model = YOLO(str(Path(__file__).resolve().parent / "model.pt"))
cap = None
screen_region = None

if args.screen:
    # Capture a still to let the user select the game. Pillow is already a YOLO dependency.
    from PIL import ImageGrab

    screenshot = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
    print("Select the game area, then press Enter or Space. Press Esc to cancel.")
    x, y, width, height = cv2.selectROI("Select game area", screenshot, showCrosshair=True)
    cv2.destroyWindow("Select game area")
    if width == 0 or height == 0:
        raise SystemExit("No game area selected.")
    screen_region = (x, y, x + width, y + height)
    print("Capturing the selected screen area. Keep the game visible and press q in the preview to quit.")
else:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

controller = None
if args.autodrive:
    backend = WindowsKeys()
    x1, y1, x2, y2 = screen_region
    controller = Controller(backend, backend.window_at((x1+x2)//2, (y1+y2)//2))
    print("Experimental mode: click the game, press F8 to arm; F9 stops immediately.")
    print("Focus loss or a detection timeout also stops and releases all keys.")

last_beep_time = 0


# def beep():
#     playsound("../sounds/warning.mp3")


def get_color(class_id):
    np.random.seed(class_id)
    color = np.random.randint(0, 255, 3)
    return tuple(int(c) for c in color)


try:
    while True:
        frame_started = time.monotonic()
        if screen_region is not None:
            frame = cv2.cvtColor(np.array(ImageGrab.grab(bbox=screen_region)), cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # Lane detection overlay
        lane_offset, frame_with_lanes = game_lane(frame) if controller else (None, detect_lanes(frame))

        results = model(frame, conf=0.5, imgsz=args.imgsz, verbose=False)

        annotated_frame = frame_with_lanes.copy()

        control_boxes = []
        if results[0].boxes is not None:
            threshold = 800  # Object too close
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                control_boxes.append((str(class_name), x1, y1, x2, y2))
                box_area = abs((x2 - x1) * (y2 - y1))
                frame_height, frame_width = frame.shape[:2]
                center_left = int(frame_width * 1 / 3) + 40
                center_right = int(frame_width * 2 / 3) + 40
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2

                now = time.time()

                if (
                    class_name == "car"
                    and box_area > threshold * 10
                    and x2 != x1
                    and abs(y2 - y1) / abs(x2 - x1) > 0.5
                    and center_left <= box_center_x <= center_right
                ):
                    cv2.putText(
                        annotated_frame,
                        "CAR TOO CLOSE!",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )
                    # if time.time() - last_beep_time > 1:
                    #     threading.Thread(target=beep, daemon=True).start()
                    #     last_beep_time = time.time()
                elif class_name == "person" and box_area > threshold * 3:
                    cv2.putText(
                        annotated_frame,
                        "PERSON TOO CLOSE!",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )
                    # if time.time() - last_beep_time > 1:
                    #     threading.Thread(target=beep, daemon=True).start()
                    #     last_beep_time = time.time()

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), get_color(class_id), 2)
                label = f"{class_name}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - th - 4),
                    (x1 + tw, y1),
                    get_color(class_id),
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        if controller:
            keys, action = decide(lane_offset, control_boxes, frame.shape[1], frame.shape[0], time.monotonic())
            status = controller.update(keys, action)
            cv2.putText(annotated_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (0, 255, 255), 2)

        window_name = "Smart Dashcam System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, annotated_frame)

        if cv2.getWindowProperty("Smart Dashcam System", cv2.WND_PROP_VISIBLE) < 1:
            break

        # Process window events while waiting for the next inference frame.
        remaining_ms = max(1, int((1.0 / args.fps - (time.monotonic() - frame_started)) * 1000))
        if cv2.waitKey(remaining_ms) & 0xFF == ord("q"):
            break

finally:
    if controller is not None:
        controller.close()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
