import cv2
import numpy as np
from lane import detect_lanes


def get_color(class_id):
    rng = np.random.default_rng(class_id)
    return tuple(int(c) for c in rng.integers(0, 255, 3))


def process_frame(frame, model):
    # Lane detection overlay
    frame_with_lanes = detect_lanes(frame)

    results = model(frame, conf=0.5)

    annotated_frame = frame_with_lanes.copy()

    if results[0].boxes is not None:
        threshold = 800  # Object too close
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_area = abs((x2 - x1) * (y2 - y1))
            frame_width = frame.shape[1]
            center_left = int(frame_width * 1 / 3) + 40
            center_right = int(frame_width * 2 / 3) + 40
            box_center_x = (x1 + x2) // 2

            if (
                x2 != x1
                and class_name == "car"
                and box_area > threshold * 10
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

    return annotated_frame
