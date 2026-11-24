import cv2
import numpy as np


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 180, 240)
    height, width = edges.shape[:2]
    mask = np.zeros_like(edges)
    if len(edges.shape) > 2:
        channel_count = edges.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    bottom_left = [width * 0.15, height]
    top_left = [width * 0.45, height * 0.6]
    bottom_right = [width * 0.95, height]
    top_right = [width * 0.55, height * 0.6]
    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    cv2.fillPoly(mask, ver, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # masker_img !
    # Focus on the lower half of the frame
    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180, 20, np.array([]), minLineLength=20, maxLineGap=180
    )
    line_img = np.zeros_like(frame)
    line_img = slope_lines(line_img, lines)
    # houghed_img !
    return cv2.addWeighted(frame, 0.8, line_img, 0.2, 0)


def slope_lines(image, lines):
    img = image.copy()
    poly_vertices = []
    order = [0, 1, 3, 2]

    left_lines = []
    right_lines = []

    # Handle case when no lines are detected
    if lines is None:
        return image

    # OpenCV versions return either (N, 1, 4) or (N, 4).
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        if x1 == x2:
            continue
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1

        if m < 0:
            left_lines.append((m, c))
        else:
            right_lines.append((m, c))

    # Only calculate means if we have lines
    if len(left_lines) > 0:
        left_lines = np.mean(left_lines, axis=0)
    else:
        left_lines = None

    if len(right_lines) > 0:
        right_lines = np.mean(right_lines, axis=0)
    else:
        right_lines = None

    # Only process lines that exist
    valid_lines = []
    if left_lines is not None:
        valid_lines.append(left_lines)
    if right_lines is not None:
        valid_lines.append(right_lines)

    for slope, intercept in valid_lines:
        # Skip if slope is zero or very small to avoid division by zero
        if abs(slope) < 0.001:
            continue

        # getting complete height of image in y1
        rows, cols = image.shape[:2]
        y1 = int(rows)  # image.shape[0]

        # taking y2 upto 60% of actual height or 60% of y1
        y2 = int(rows * 0.6)  # int(0.6*y1)

        # we know that equation of line is y=mx +c so we can write it x=(y-c)/m
        try:
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)

            # Check if coordinates are reasonable (within image bounds)
            if -1000 <= x1 <= cols + 1000 and -1000 <= x2 <= cols + 1000:
                poly_vertices.append((x1, y1))
                poly_vertices.append((x2, y2))
                draw_lines(img, np.array([[[x1, y1, x2, y2]]]))
        except (ZeroDivisionError, OverflowError, ValueError):
            # Skip this line if there are mathematical errors
            continue

    # Only fill polygon if we have enough vertices
    if len(poly_vertices) >= 4:
        poly_vertices = [poly_vertices[i] for i in order]
        cv2.fillPoly(img, pts=np.array([poly_vertices], "int32"), color=(0, 255, 0))

    return cv2.addWeighted(image, 0.7, img, 0.4, 0.0)


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def game_lane(frame, tracker=None):
    """Estimate lane center offset using white/yellow markings in a road trapezoid.

    Returns (offset, overlay), with offset None when both boundaries are not credible.
    """
    h, w = frame.shape[:2]
    roi_top = int(.55*h)
    hsv = cv2.cvtColor(frame[roi_top:], cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 175), (180, 85, 255))
    yellow = cv2.inRange(hsv, (15, 70, 130), (40, 255, 255))
    markings = cv2.bitwise_or(white, yellow)
    edges = cv2.Canny(cv2.GaussianBlur(markings, (5, 5), 0), 50, 150)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [np.array([(int(.04*w), h-roi_top-1), (int(.38*w), 0),
                                  (int(.62*w), 0), (int(.96*w), h-roi_top-1)], np.int32)], 255)
    lines = cv2.HoughLinesP(cv2.bitwise_and(edges, mask), 1, np.pi/180,
                            threshold=max(15, w//40), minLineLength=max(15, w//25), maxLineGap=w//20)
    sides = [[], []]
    bottom, top = .90*h, .62*h
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if abs(y2-y1) < h*.025:
                continue
            dxdy = (x2-x1)/(y2-y1)
            if not .2 < abs(dxdy) < 2.5:
                continue
            y1 += roi_top
            y2 += roi_top
            xb = x1 + (bottom-y1)*dxdy
            xt = x1 + (top-y1)*dxdy
            if dxdy < 0 and -.2*w < xb < .55*w:
                sides[0].append((xb, xt, np.hypot(x2-x1, y2-y1)))
            elif dxdy > 0 and .45*w < xb < 1.2*w:
                sides[1].append((xb, xt, np.hypot(x2-x1, y2-y1)))
    overlay = frame.copy()
    if not all(sides):
        if tracker is not None:
            tracker.reset()
        return None, overlay

    def clusters(side):
        candidates = np.asarray(side)
        groups = []
        for candidate in candidates:
            group = candidates[np.abs(candidates[:, 0] - candidate[0]) < .07*w]
            line = np.average(group[:, :2], axis=0, weights=group[:, 2])
            groups.append((line, group[:, 2].sum()))
        return groups

    # Choose boundaries together: independently selecting the strongest lines can
    # combine a road edge and an unrelated marking into a fictitious lane.
    pairs = []
    left_clusters = clusters(sides[0])
    right_clusters = clusters(sides[1])
    for left, left_support in left_clusters:
        for right, right_support in right_clusters:
            bottom_width = right[0] - left[0]
            top_width = right[1] - left[1]
            if not (.25*w < bottom_width < 1.1*w and 0 < top_width < .75*bottom_width):
                continue
            if not (-.1*w < left[1] < 1.1*w and -.1*w < right[1] < 1.1*w):
                continue
            center_error = abs((left[0] + right[0])/2 - w/2) / w
            # An interior marking means this pair spans multiple lanes.
            if any(left[0]+.09*w < line[0] < right[0]-.09*w
                   for line, _ in left_clusters + right_clusters):
                continue
            score = (left_support + right_support) / (1 + 3*center_error)
            pairs.append((score, left, right))
    if not pairs:
        if tracker is not None:
            tracker.reset()
        return None, overlay
    _, left, right = max(pairs, key=lambda pair: pair[0])
    if tracker is not None:
        left, right = tracker.smooth(left, right, w)
        tracker.adjacent.clear()
        lane_width = right[0] - left[0]
        for side, sign, inner, candidates in (
                ('left', -1, left, left_clusters), ('right', 1, right, right_clusters)):
            neighbors = []
            for outer, _ in candidates:
                gap = sign * (outer - inner)
                if .45*lane_width < gap[0] < 1.55*lane_width and .05*gap[0] < gap[1] < .9*gap[0]:
                    neighbors.append((gap[0], (outer + inner) / 2))
            if neighbors:
                _, centerline = min(neighbors, key=lambda item: item[0])
                tracker.adjacent[side] = (.65*centerline[0] + .35*centerline[1] - w/2)/(w/2)
    for xb, xt in (left, right):
        cv2.line(overlay, (int(xb), int(bottom)), (int(xt), int(top)), (0, 255, 0), 3)
    center = (left[0] + right[0]) / 2
    cv2.circle(overlay, (int(center), int(bottom)), 6, (0, 255, 255), -1)
    lookahead = (left[1] + right[1]) / 2
    target = .65 * center + .35 * lookahead
    return (target - w/2)/(w/2), overlay


class LaneTracker:
    """Smooth confirmed boundaries; never carry them through a missing detection."""
    def __init__(self):
        self.adjacent = {}
        self.previous = None

    def reset(self):
        self.adjacent.clear()
        self.previous = None

    def smooth(self, left, right, width):
        current = np.array([left, right], dtype=float)
        if self.previous is not None and np.max(np.abs(current - self.previous)) < .15 * width:
            current = .35 * current + .65 * self.previous
        self.previous = current
        return current
