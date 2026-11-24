"""Interactive capture selection that yields to Python for terminal interrupts."""
import cv2


def select_region(window, frame):
    start = end = None
    dragging = False
    h, w = frame.shape[:2]

    def mouse(event, x, y, flags, param):
        nonlocal start, end, dragging
        point = (min(max(x, 0), w-1), min(max(y, 0), h-1))
        if event == cv2.EVENT_LBUTTONDOWN:
            start = end = point
            dragging = True
        elif dragging and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
            end = point
            dragging = event != cv2.EVENT_LBUTTONUP

    cv2.namedWindow(window)
    try:
        cv2.setMouseCallback(window, mouse)
        drawn = None
        while True:
            if drawn != (start, end):
                display = frame.copy()
                if start is not None:
                    cv2.rectangle(display, start, end, (0, 255, 0), 2)
                cv2.imshow(window, display)
                drawn = (start, end)
            key = cv2.waitKey(25) & 0xff
            if key in (27, ord('c')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return (0, 0, 0, 0)
            if key in (10, 13, 32) and start is not None and not dragging:
                x, right = sorted((start[0], end[0]))
                y, bottom = sorted((start[1], end[1]))
                if right > x and bottom > y:
                    return x, y, right-x+1, bottom-y+1
    finally:
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(window)
