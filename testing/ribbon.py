"""Perspective path selection for a forward-facing GTA camera."""
import cv2
import numpy as np


class RibbonPlanner:
    # Bend is the lateral displacement at the far end, as a fraction of frame width.
    bends = (0., -.12, .12, -.24, .24)
    traffic = {'car', 'truck', 'bus', 'motorcycle', 'motorbike', 'bicycle', 'person'}
    horizon = .55

    def __init__(self):
        self.reset()
        self.rows = np.linspace(self.horizon, 1., 32)
        self.samples = []
        for bend in self.bends:
            left, right = self.limits(self.rows, bend)
            cols = left[:, None] + (right-left)[:, None]*np.linspace(.02, .98, 24)
            self.samples.append((np.rint(self.rows*179).astype(int)[:, None],
                                 np.rint(cols*319).astype(int)))

    def reset(self):
        self.bend = 0.
        self.braking = False
        self.brake_until = 0.
        self.last_frame = None

    @classmethod
    def limits(cls, y, bend):
        depth = np.clip((y-cls.horizon)/(1-cls.horizon), 0., 1.)
        center = .5 + bend*(1-depth)**1.6
        half_width = .037 + .123*depth
        return center-half_width, center+half_width

    def intersects(self, box, bend, width, height):
        _, x1, _, x2, y2 = box
        if y2 < self.horizon*height:
            return False
        left, right = self.limits(y2/height, bend)
        inset = .25*(x2-x1)
        return x1+inset < right*width and x2-inset > left*width

    def decide(self, frame, boxes, now):
        if self.last_frame is not None and now-self.last_frame > .6:
            self.reset()
        self.last_frame = now
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        smooth = cv2.GaussianBlur(small, (5, 5), 0)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        road = gray[143:169, 146:174]
        if np.median(road) < 9 or np.median(road) > 246:
            self.bend = 0.
            self.braking = False
            return set(), 'Road obscured'

        edge = cv2.dilate(cv2.Canny(gray, 35, 95), np.ones((3, 3), np.uint8)) > 0
        # Compare surface colour to the pavement immediately ahead. Lab chroma
        # keeps shadows less influential than kerbs, grass and road markings.
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.int16)
        reference = np.median(lab[143:169, 146:174], axis=(0, 1))
        delta = np.abs(lab-reference)
        different = (delta[:, :, 1:].sum(axis=2) > 28) | (delta[:, :, 0] > 55)
        evidence = .70*edge + .72*different
        traffic = [box for box in boxes if box[0].lower() in self.traffic and
                   (box[3]-box[1])*(box[4]-box[2]) > .004*width*height]
        urgent = any(self.intersects(box, 0., width, height) and
                     (box[0].lower() == 'person' or
                      (box[4] > .72*height and (box[3]-box[1])*(box[4]-box[2]) > .025*width*height))
                     for box in traffic)
        if urgent and not self.braking:
            self.brake_until = now+.18
        self.braking = urgent
        if urgent:
            self.bend = 0.
            return ({'down'} if now < self.brake_until else set()), 'Obstacle ahead'

        choices = []
        for bend, (rows, cols) in zip(self.bends, self.samples):
            if any(self.intersects(box, bend, width, height) for box in traffic):
                continue
            samples = evidence[rows, cols]
            # Include the worst band so a short intrusion is not averaged away.
            cost = .55*samples.mean() + .45*np.max(samples.reshape(8, -1).mean(axis=1))
            if cost < .46:
                choices.append((cost + .20*abs(bend) + .12*abs(bend-self.bend), bend))
        if not choices:
            self.bend = 0.
            return set(), 'Path blocked: coasting'
        _, self.bend = min(choices, key=lambda item: item[0])
        if self.bend == 0:
            return {'up'}, 'Clear path'
        side = 'left' if self.bend < 0 else 'right'
        return {'up', side}, f'Following ribbon: {side}'

    def draw(self, frame, keys):
        height, width = frame.shape[:2]
        left, right = self.limits(self.rows, self.bend)
        ys = np.rint(self.rows*(height-1))
        polygon = np.int32(np.vstack((np.column_stack((left*(width-1), ys)),
                                      np.column_stack((right*(width-1), ys))[::-1])))
        color = (70, 200, 100) if 'up' in keys else (30, 90, 230)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, .23, frame, .77, 0, dst=frame)
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
