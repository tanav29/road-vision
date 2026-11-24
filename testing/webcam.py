import argparse
from pathlib import Path


def main():
    import cv2
    import time

    import numpy as np
    from ultralytics import YOLO
    from lane import detect_lanes
    from ribbon import RibbonPlanner
    from autodrive import Controller, WindowsKeys
    from selection import select_region

    parser = argparse.ArgumentParser(description="Run the dashcam detector on a video or a browser game.")
    parser.add_argument("--autodrive", action="store_true", help="Enable experimental Windows game controls (requires --screen)")
    parser.add_argument("--map-image", type=Path, help="Use this full map image for screen localization")
    parser.add_argument("--map", action="store_true", help="Show rotating minimap localization alongside screen capture")
    parser.add_argument("--screen", action="store_true", help="Select an area of the primary screen to capture")
    parser.add_argument("--video", default="./videos/dashcam3.mp4", help="Video path when not using --screen")
    parser.add_argument("--fps", type=float, default=60.0, help="Maximum inference frames per second (default: 60)")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference image size (default: 320)")
    args = parser.parse_args()
    if args.map_image:
        args.map = True
    if args.map and not args.screen:
        parser.error("--map requires --screen")
    if args.autodrive and not args.screen:
        parser.error("--autodrive requires --screen")
    if not 1 <= args.fps <= 60:
        parser.error("--fps must be between 1 and 60")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")

    model = YOLO(str(Path(__file__).resolve().parent / "model.pt"))
    cap = controller = map_preview = None
    screen_region = None
    try:

        if args.screen:
            # Capture a still to let the user select the game. Pillow is already a YOLO dependency.
            from PIL import ImageGrab

            screenshot = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
            print("Select the game area, then press Enter or Space. Press Esc to cancel.")
            x, y, width, height = select_region("Select game area", screenshot)
            if width == 0 or height == 0:
                raise SystemExit("No game area selected.")
            screen_region = (x, y, x + width, y + height)
            print("Capturing the selected screen area. Keep the game visible and press q in the preview to quit.")
        else:
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                raise SystemExit(f"Could not open video: {args.video}")

        if args.autodrive:
            backend = WindowsKeys()
            x1, y1, x2, y2 = screen_region
            capture_point = ((x1+x2)//2, (y1+y2)//2)
            print("Focus loss or a detection timeout also stops and releases all keys.")

        colors = {i: tuple(int(c) for c in np.random.default_rng(i).integers(0, 255, 3))
                  for i in model.names}

        planner = RibbonPlanner()
        window_name = "Smart Dashcam System"
        if args.map:
            from navigation import MAP_PATH, MapPreview
            map_preview = MapPreview(screenshot[y:y+height, x:x+width], args.map_image or MAP_PATH)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, 500, 500)
        previous_start = None
        while True:
            frame_started = time.monotonic()
            if screen_region is not None:
                frame = cv2.cvtColor(np.array(ImageGrab.grab(bbox=screen_region)), cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            if map_preview:
                map_preview.update(frame)

            results = model(frame, conf=0.5, imgsz=args.imgsz, verbose=False)
            frame_height, frame_width = frame.shape[:2]
            control_boxes = []
            if results[0].boxes is not None:
                for box in results[0].boxes.cpu():
                    class_id = int(box.cls[0])
                    control_boxes.append((class_id, *map(int, box.xyxy[0])))
            if args.autodrive:
                named_boxes = [(model.names[i], *coords) for i, *coords in control_boxes]
                keys, action = planner.decide(frame, named_boxes, time.monotonic())
                annotated_frame = frame.copy()
                planner.draw(annotated_frame, keys)
            else:
                annotated_frame = detect_lanes(frame)
            if control_boxes:
                threshold = 800  # Object too close
                for class_id, x1, y1, x2, y2 in control_boxes:
                    class_name = model.names[class_id]
                    box_area = abs((x2 - x1) * (y2 - y1))
                    relevant = not args.autodrive or planner.intersects(
                        (class_name, x1, y1, x2, y2), frame_width, frame_height)
                    center_left = int(frame_width / 3) + 40
                    center_right = int(frame_width * 2 / 3) + 40
                    box_center_x = (x1 + x2) // 2

                    if (
                        class_name == "car"
                        and box_area > threshold * 10
                        and x2 != x1
                        and abs(y2 - y1) / abs(x2 - x1) > 0.5
                        and relevant
                        and (args.autodrive or center_left <= box_center_x <= center_right)
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
                    elif class_name == "person" and relevant and box_area > threshold * 3:
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

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), colors[class_id], 2)
                    label = f"{class_name}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - th - 4),
                        (x1 + tw, y1),
                        colors[class_id],
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
                status = controller.update(keys, action)
                if status != action:
                    planner.reset()
                cv2.putText(annotated_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            .65, (0, 255, 255), 2)

            fps = (f"{1 / max(frame_started - previous_start, 1e-6):.1f}"
                   if previous_start is not None else "...")
            previous_start = frame_started
            cv2.putText(annotated_frame, f"Detection: {fps} FPS (target {args.fps:g})",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
            cv2.imshow(window_name, annotated_frame)

            if cv2.getWindowProperty("Smart Dashcam System", cv2.WND_PROP_VISIBLE) < 1:
                break

            # Process window events while waiting for the next inference frame.
            remaining_ms = max(1, int((1.0 / args.fps - (time.monotonic() - frame_started)) * 1000))
            if cv2.waitKey(remaining_ms) & 0xFF == ord("q"):
                break
            if map_preview and cv2.getWindowProperty(map_preview.window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.autodrive and controller is None:
                # Show and pump both preview windows before listening for the arm shortcut.
                controller = Controller(backend, capture_point=capture_point)
                print("Ready: click the game, then press Ctrl+Alt+G. Ctrl+Alt+X stops.")

    finally:
        if controller is not None:
            controller.close()
        if map_preview is not None:
            map_preview.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
        raise SystemExit(130)
