"""Rotation-aware minimap localization for the GTA screen capture."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import ImageGrab
from selection import select_region

MAP_PATH = Path(__file__).resolve().parent.parent / 'map.webp'


class MinimapMatcher:
    def __init__(self, world):
        gray = cv2.cvtColor(world, cv2.COLOR_BGR2GRAY)
        self.roads = (gray < 80).astype(np.float32)
        self.coarse = cv2.resize(self.roads, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
        self.norms = {}
        self.pose = None
        self.status = 'Searching the full map'
        self.sift = cv2.SIFT_create(contrastThreshold=.015)
        features, descriptors = self.sift.detectAndCompute(gray, None)
        self.reference_points = np.float32([point.pt for point in features])
        self.index = None
        if descriptors is not None and len(descriptors) >= 2:
            self.index = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
            self.index.add([descriptors])
            self.index.train()

    def landmark_fix(self, crop):
        if self.index is None:
            return None
        gray = cv2.cvtColor(cv2.resize(crop, (160, 160)), cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        cv2.circle(mask, (80, 80), 64, 255, -1)
        cv2.circle(mask, (80, 80), 13, 0, -1)
        points, desc = self.sift.detectAndCompute(gray, mask)
        if desc is None:
            return None
        # Require distinct reference landmarks before fitting rotation and zoom.
        matches = {}
        for first, second in self.index.knnMatch(desc, k=2):
            if first.distance < .74 * second.distance:
                previous = matches.get(first.trainIdx)
                if previous is None or first.distance < previous.distance:
                    matches[first.trainIdx] = first
        if len(matches) < 6:
            return None
        src = np.float32([points[m.queryIdx].pt for m in matches.values()])
        dst = self.reference_points[list(matches)]
        matrix, inliers = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=2.5)
        if matrix is None:
            return None
        support = src[inliers.ravel().astype(bool)]
        if len(support) < max(6, .65*len(src)) or cv2.contourArea(cv2.convexHull(support)) < 450:
            return None
        x, y = matrix @ np.array([79.5, 79.5, 1])
        if not (0 <= x < self.roads.shape[1] and 0 <= y < self.roads.shape[0]):
            return None
        angle = np.degrees(np.arctan2(matrix[0, 1], matrix[0, 0])) % 360
        self.status = f'Tracking {len(support)} landmarks'
        return round(x), round(y), round(angle) % 360, len(support)/len(src)

    @staticmethod
    def search(world, template, mask, side, angle):
        # Zero-mean correlation within the radar ring penalizes solid map features.
        center = (side - 1) / 2
        resized = cv2.resize(template, (side, side), interpolation=cv2.INTER_AREA)
        matrix = cv2.getRotationMatrix2D((center, center), angle, 1)
        patch = cv2.warpAffine(resized, matrix, (side, side))
        pixels = mask.sum()
        kernel = (patch - np.sum(patch * mask) / pixels) * mask
        energy = np.sum(kernel * kernel)
        if energy < .005 * pixels:
            return None
        return cv2.matchTemplate(world, kernel, cv2.TM_CCORR) / np.sqrt(energy)

    @staticmethod
    def normalization(world, side):
        y, x = np.ogrid[:side, :side]
        r2 = (x-(side-1)/2)**2 + (y-(side-1)/2)**2
        mask = ((r2 < (.39*side)**2) & (r2 > (.07*side)**2)).astype(np.float32)
        sums = cv2.matchTemplate(world, mask, cv2.TM_CCORR)
        squares = cv2.matchTemplate(world * world, mask, cv2.TM_CCORR)
        variance = squares - sums*sums / mask.sum()
        return mask, 1 / np.sqrt(np.maximum(variance, .005 * mask.sum()))

    @staticmethod
    def radar_layers(crop):
        # Brightness varies with HUD opacity. Isolate narrow bright and dark
        # strokes relative to their local background, ignoring the rim and arrow.
        gray = cv2.cvtColor(cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        yy, xx = np.ogrid[:96, :96]
        ring = (xx-47.5)**2 + (yy-47.5)**2
        valid = (ring < 38**2) & (ring > 7**2)
        layers = {}
        for mode in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
            contrast = cv2.morphologyEx(gray, mode, kernel)
            level, _ = cv2.threshold(contrast[valid], 0, 255, cv2.THRESH_OTSU)
            strokes = contrast > max(6, level)
            if .012 < strokes[valid].mean() < .60:
                layers[mode] = strokes.astype(np.float32)
        return layers

    def refine(self, layer, x, y, angle, side, factors, angles, radius):
        best = None
        for factor in factors:
            length = round(side * factor)
            pad = length//2 + radius
            x0, y0 = max(0, x-pad), max(0, y-pad)
            region = self.roads[y0:y+pad+1, x0:x+pad+1]
            if min(region.shape) < length:
                continue
            mask, norm = self.normalization(region, length)
            for delta in angles:
                rotation = (angle + delta) % 360
                scores = self.search(region, layer, mask, length, rotation)
                if scores is None:
                    continue
                _, score, _, (px, py) = cv2.minMaxLoc(scores * norm)
                candidate = (score, x0+px+length//2, y0+py+length//2, rotation, length)
                if best is None or candidate > best:
                    best = candidate
        return best

    def match(self, minimap):
        h, w = minimap.shape[:2]
        size = min(h, w)
        layers = {}
        if size >= 30:
            x, y = (w-size)//2, (h-size)//2
            crop = minimap[y:y+size, x:x+size]
            if self.pose is None:
                fix = self.landmark_fix(crop)
                if fix is not None:
                    return fix
            layers = self.radar_layers(crop)
        if not layers:
            self.pose = None
            self.status = 'Too little radar detail - check the selection or move'
            return None

        if self.pose:
            x, y, angle, side, mode = self.pose
            if mode in layers:
                near = self.refine(layers[mode], x, y, angle, side,
                                   (1., .96, 1.04), range(-18, 19, 3), max(12, side//6))
                if near and near[0] >= .72:
                    return self.accept(near, mode)

        self.pose = None
        peaks = []
        # Search footprints in reference-map pixels. Screen DPI and HUD zoom
        # can change the radar's screen diameter independently of its coverage.
        for side in (10, 13, 17, 21, 25, 30, 35, 40, 45, 54, 65):
            if side > min(self.coarse.shape):
                continue
            if side not in self.norms:
                self.norms[side] = self.normalization(self.coarse, side)
            mask, norm = self.norms[side]
            for mode, layer in layers.items():
                for angle in range(0, 360, 12):
                    scores = self.search(self.coarse, layer, mask, side, angle)
                    if scores is None:
                        continue
                    scores *= norm
                    for _ in range(2):
                        _, score, _, (px, py) = cv2.minMaxLoc(scores)
                        if score < .36:
                            break
                        # Small footprints correlate by chance more often.
                        rank = score - 2.8/side
                        peaks.append((rank, 4*px+2*side, 4*py+2*side, angle, 4*side, mode))
                        radius = max(3, side//3)
                        scores[max(0,py-radius):py+radius+1, max(0,px-radius):px+radius+1] = -2

        # Compare distinct locations after refinement. A coarse-grid tie can
        # resolve once the reference is searched at native resolution.
        finalists = []
        for _, x, y, angle, side, mode in sorted(peaks, reverse=True):
            if any(np.hypot(x-p[1], y-p[2]) < max(6, side*.15) for p, _ in finalists):
                continue
            candidate = self.refine(layers[mode], x, y, angle, side,
                                    (.92, 1., 1.08), range(-6, 7, 2), 8)
            if candidate:
                finalists.append((candidate, mode))
            if len(finalists) == 5:
                break
        finalists.sort(reverse=True)
        if finalists:
            best, mode = finalists[0]
            runner = max((p[0] for p, _ in finalists[1:]
                          if np.hypot(best[1]-p[1], best[2]-p[2]) > best[4]*.2), default=0.)
            if best[0] >= .48 and best[0]-runner >= .045:
                return self.accept(best, mode)
            self.status = ('Similar locations - move toward a junction' if best[0] >= .48
                           else 'Weak road match - check radar crop and map')
        else:
            self.status = 'No matching roads - check radar crop and map'
        return None

    def accept(self, candidate, mode):
        score, x, y, angle, side = candidate
        self.pose = (x, y, angle, side, mode)
        self.status = 'Tracking'
        return x, y, angle, score


def match_minimap(minimap, world):
    return MinimapMatcher(world).match(minimap)


class MapPreview:
    window = 'Destination map'

    def __init__(self, game, map_path=MAP_PATH):
        self.world = cv2.imread(str(map_path))
        if self.world is None:
            raise SystemExit(f'Cannot load map: {map_path}')
        longest = max(self.world.shape[:2])
        if longest > 1000:
            scale = 1000/longest
            self.world = cv2.resize(self.world, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print('Select the circular minimap, then press Enter.')
        self.roi = select_region('Select minimap', game)
        if not all(self.roi[2:]):
            raise SystemExit('No minimap selected.')
        self.matcher = MinimapMatcher(self.world)
        self.destination = self.candidate = self.future = None
        self.started = 0
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.click)
        self.worker = ThreadPoolExecutor(max_workers=1)

    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.destination = (x, y)

    def update(self, game):
        now = time.monotonic()
        if self.future is not None and self.future.done():
            self.candidate = self.future.result()
            self.captured = self.started
            self.future = None
        if self.future is None and now-self.started >= .12:
            x, y, w, h = self.roi
            self.started = now
            self.future = self.worker.submit(self.matcher.match, game[y:y+h,x:x+w].copy())
        display = self.world.copy()
        label = self.matcher.status
        if self.candidate and now-self.captured < 3:
            x, y, angle, score = self.candidate
            cv2.circle(display, (x, y), 7, (255, 100, 0), -1)
            label = f'Match {score:.2f}   radar rotation {angle} deg'
        if self.destination:
            cv2.drawMarker(display, self.destination, (0, 0, 255))
        for color, thickness in [((0,0,0), 3), ((255,255,255), 1)]:
            cv2.putText(display, label, (10,25), cv2.FONT_HERSHEY_SIMPLEX, .6, color, thickness)
        cv2.imshow(self.window, display)

    def close(self):
        self.worker.shutdown(wait=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Track the game radar on a reference map.")
    parser.add_argument("--map-image", default=MAP_PATH, help="Full map image from your game edition")
    args = parser.parse_args()
    screenshot = cv2.cvtColor(np.asarray(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
    print('Select the game area, then press Enter.')
    x, y, w, h = select_region('Select game area', screenshot)
    if not w or not h:
        raise SystemExit('No game area selected.')
    preview = MapPreview(screenshot[y:y+h,x:x+w], args.map_image)
    try:
        while True:
            frame = cv2.cvtColor(np.asarray(ImageGrab.grab(bbox=(x,y,x+w,y+h))), cv2.COLOR_RGB2BGR)
            preview.update(frame)
            if cv2.waitKey(50) & 0xff == ord('q') or cv2.getWindowProperty(preview.window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        preview.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped by Ctrl+C.')
        raise SystemExit(130)
