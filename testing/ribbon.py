"""Perspective path selection for a forward-facing GTA camera."""
import cv2
import numpy as np

from lane import game_lane, LaneTracker


class RibbonPlanner:
    traffic = {'car', 'truck', 'bus', 'motorcycle', 'motorbike', 'bicycle', 'person'}
    horizon = .55

    def __init__(self):
        self.reset()
        self.rows = np.linspace(self.horizon, 1., 32)
        self.sample_rows = np.rint(self.rows*179).astype(int)[:, None]
        self.fallback_cols = {}
        for bend in (0., -.12, .12):
            depth = (self.rows-self.horizon)/(1-self.horizon)
            shift = bend*(1-depth)**1.6
            left, right = self.limits(self.rows)
            xs = (left+shift)[:, None] + (right-left)[:, None]*np.linspace(.02, .98, 24)
            self.fallback_cols[bend] = np.rint(xs*319).astype(int)

    def reset(self):
        self.tracker = LaneTracker()
        self.bend = 0.
        self.steering = None
        self.braking = False
        self.brake_until = 0.
        self.last_frame = None
        self.crossing_clear_since = None
        self.yielding = False
        self.crosswalk = None
        self.wall_since = None
        self.reversing = False
        self.road_since = None

    def limits(self, y):
        depth = np.clip((y-self.horizon)/(1-self.horizon), 0., 1.)
        half_width = .037 + .123*depth
        if self.tracker.previous is None:
            center = .5 + self.bend*(1-depth)**1.6
            return center-half_width, center+half_width
        # The tracker measures x at 90% and 62% of the 320px-wide image.
        t = (np.maximum(y, .62)-.62)/.28
        left, right = [(top + t*(bottom-top))/320 for bottom, top in self.tracker.previous]
        center = (left+right)/2
        half_width = np.minimum(half_width, (right-left)*.40)
        return center-half_width, center+half_width

    def intersects(self, box, width, height):
        _, x1, _, x2, y2 = box
        if y2 < self.horizon*height:
            return False
        left, right = self.limits(y2/height)
        inset = .25*(x2-x1)
        return x1+inset < right*width and x2-inset > left*width

    def facade_ahead(self, gray, different, cols):
        """Check for an upright surface extending into the current lane."""
        # Long upright window frames and masonry joints can remain visible even
        # when a grey facade matches asphalt or fills the pavement reference.
        dx = np.abs(cv2.Sobel(gray, cv2.CV_16S, 1, 0))
        dy = np.abs(cv2.Sobel(gray, cv2.CV_16S, 0, 1))
        upright = ((dx > 32) & (dx > 2 * dy)).astype(np.uint8)
        upright = cv2.morphologyEx(upright, cv2.MORPH_OPEN, np.ones((13, 1), np.uint8))
        ys = np.r_[80, 87, 94, self.sample_rows[:9, 0]][:, None]
        xs = np.vstack((np.repeat(cols[:1], 3, axis=0), cols[:9]))
        surface = different[ys, xs].mean(axis=1)
        structure = upright[ys, xs].mean(axis=1)
        return np.all(surface >= .65) or np.all(structure >= .12)

    @staticmethod
    def crossing_markings(gray):
        """Find a band of repeated bright zebra stripes across the forward path."""
        bright = gray > np.clip(np.median(gray[143:169, 146:174]) + 30, 145, 220)
        groups = []
        for y in range(100, 176, 3):
            transitions = np.flatnonzero(np.diff(np.r_[False, bright[y], False]))
            runs = transitions.reshape(-1, 2)
            chain = []
            band = None
            for start, end in runs:
                if chain and not 2 <= start - chain[-1][1] <= 22:
                    chain = []
                if not 3 <= end - start <= 30:
                    chain = []
                    continue
                chain.append((start, end))
                if len(chain) >= 4 and chain[0][0] < 160 < end:
                    band = (y, chain[0][0], end)
            if band is not None:
                groups.append(band)
        band = []
        for row in reversed(groups):
            y, left, right = row
            if band:
                prev_y, prev_left, prev_right = band[-1]
                if prev_y-y > 3 or min(right, prev_right)-max(left, prev_left) < 24:
                    if len(band) >= 3:
                        break
                    band = []
            band.append(row)
        if len(band) >= 3:
            return (min(lo for _, lo, _ in band)/320, (band[-1][0]-2)/180,
                    max(hi for _, _, hi in band)/320, (band[0][0]+3)/180)
        return None

    def decide(self, frame, boxes, now):
        if self.last_frame is not None and now-self.last_frame > .6:
            waiting = self.yielding
            recovering = self.reversing
            self.reset()
            self.yielding = waiting
            self.reversing = recovering
        self.last_frame = now
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        smooth = cv2.GaussianBlur(small, (5, 5), 0)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        self.crosswalk = self.crossing_markings(gray)
        road = gray[143:169, 146:174]
        if np.median(road) < 9 or (np.median(road) > 246 and self.crosswalk is None):
            self.steering = None
            self.braking = False
            self.crossing_clear_since = self.road_since = self.wall_since = None
            return set(), 'Road obscured'

        offset, _ = game_lane(small, self.tracker)
        self.bend = 0.
        if offset is None and self.crosswalk is None:
            # Follow an incomplete road edge using the earlier ribbon costs.
            # Limit corrections to the two shallow bends on unmarked pavement.
            edges = cv2.dilate(cv2.Canny(gray, 35, 95), np.ones((3, 3), np.uint8)) > 0
            costs = {}
            for bend, sample_cols in self.fallback_cols.items():
                samples = edges[self.sample_rows, sample_cols]
                costs[bend] = (.55*samples.mean() + .45*samples.reshape(8, -1).mean(axis=1).max()
                               + abs(bend)*.20)
            self.bend = min(costs, key=costs.get)
            offset = self.bend
        left_path, right_path = self.limits(self.rows)
        xs = left_path[:, None] + (right_path-left_path)[:, None]*np.linspace(.02, .98, 24)
        cols = np.clip(np.rint(xs*319).astype(int), 0, 319)
        # Compare surface colour to the pavement immediately ahead. Lab chroma
        # keeps shadows less influential than kerbs, grass and road markings.
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.int16)
        reference = np.median(lab[143:169, 146:174], axis=(0, 1))
        delta = np.abs(lab-reference)
        different = delta[:, :, 1:].sum(axis=2) > 28
        if self.crosswalk:
            left, top, right, bottom = self.crosswalk

        for box in boxes:
            if box[0].lower() not in {'person', 'pedestrian'}:
                continue
            _, x1, _, x2, foot = box
            on_crossing = (self.crosswalk is not None and x2/width > left-.035
                           and x1/width < right+.035 and top-.035 <= foot/height <= bottom+.035)
            if on_crossing or self.intersects(box, width, height):
                self.road_since = self.wall_since = None
                if not self.yielding:
                    self.brake_until = now + .18
                self.yielding = True
                self.crossing_clear_since = None
                self.steering = None
                return ({'down'} if now < self.brake_until else set()), 'Yielding to pedestrian'

        if self.yielding:
            if self.crossing_clear_since is None:
                self.crossing_clear_since = now
            if now - self.crossing_clear_since < .8:
                return ({'down'} if now < self.brake_until else set()), 'Waiting for crossing to clear'
            self.yielding = False
            self.crossing_clear_since = None
        traffic = [box for box in boxes if box[0].lower() in self.traffic and
                   (box[3]-box[1])*(box[4]-box[2]) > .004*width*height]
        urgent = any(self.intersects(box, width, height) and
                     (box[0].lower() == 'person' or
                      (box[4] > .72*height and (box[3]-box[1])*(box[4]-box[2]) > .025*width*height))
                     for box in traffic)
        if urgent and not self.braking:
            self.brake_until = now+.18
        self.braking = urgent
        if urgent:
            self.road_since = self.wall_since = None
            self.steering = None
            return ({'down'} if now < self.brake_until else set()), 'Obstacle ahead'

        # Sample across the horizon and foreground. Flat asphalt alone only
        # fills the lower view. Sparse HUD graphics may overlay a solid wall.
        scene = smooth[10:165:5, 24:296:5].astype(np.int16)
        color = np.median(scene, axis=(0, 1))
        similar = np.max(np.abs(scene-color), axis=2) <= 16
        wall = similar[:13].mean() >= .85 and similar[13:].mean() >= .85
        # Plain concrete can match the road's hue. Require the light or dark
        # surface to span the horizon and the approach, as for coloured walls.
        facade = self.facade_ahead(gray, different | (delta[:, :, 0] > 45), cols)
        if wall or facade:
            self.road_since = None
            self.steering = None
            if self.wall_since is None:
                self.wall_since = now
            if now - self.wall_since >= .15:
                self.reversing = True
            if self.reversing:
                return {'down'}, 'Wall ahead: reversing'
            return ({'down'} if facade else set()), 'Checking for wall'
        self.wall_since = None

        # Require a broad, continuous off-colour surface. Thin paint, asphalt
        # texture and changes in brightness alone are common on clear lanes.
        covered = different[self.sample_rows, cols].mean(axis=1) > .70
        offroad = np.any(np.convolve(covered.astype(int), np.ones(5), 'valid') == 5)
        traffic_ahead = any(self.intersects(box, width, height) for box in traffic)
        if self.reversing:
            if traffic_ahead:
                self.road_since = None
                return set(), 'Waiting for lane to clear'
            if offroad:
                self.road_since = None
                return {'down'}, 'Reversing: looking for road'
            if self.road_since is None:
                self.road_since = now
            if now - self.road_since < .20:
                return set(), 'Checking recovered road'
            self.reversing = False
            self.road_since = None
        if traffic_ahead:
            self.steering = None
            return set(), 'Waiting for traffic ahead'
        if offroad:
            self.steering = None
            return set(), 'Road surface differs: coasting'
        if offset is None or abs(offset) < .035:
            self.steering = None
        elif abs(offset) > .09:
            self.steering = 'right' if offset > 0 else 'left'
        elif (offset > 0) != (self.steering == 'right'):
            self.steering = None
        if self.steering:
            return {'up', self.steering}, f'Centering in lane: {self.steering}'
        return {'up'}, 'Holding course'

    def draw(self, frame, keys):
        height, width = frame.shape[:2]
        if self.crosswalk:
            left, top, right, bottom = self.crosswalk
            cv2.rectangle(frame, (int(left*width), int(top*height)),
                          (int(right*width), int(bottom*height)), (255, 200, 60), 2)
        left, right = self.limits(self.rows)
        ys = np.rint(self.rows*(height-1))
        polygon = np.int32(np.vstack((np.column_stack((left*(width-1), ys)),
                                      np.column_stack((right*(width-1), ys))[::-1])))
        color = (70, 200, 100) if 'up' in keys else (30, 90, 230)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, .23, frame, .77, 0, dst=frame)
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
