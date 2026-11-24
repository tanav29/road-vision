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
        self.coarse = cv2.resize(self.roads, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)

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

    def match(self, minimap):
        h, w = minimap.shape[:2]
        size = min(h, w)
        if size < 30:
            return None
        y, x = (h-size)//2, (w-size)//2
        gray = cv2.cvtColor(minimap[y:y+size, x:x+size], cv2.COLOR_BGR2GRAY)
        roads = (gray > 185).astype(np.float32)
        if roads.max() == roads.min():
            return None
        peaks = []
        for scale in (.7, 1., 1.4, 1.8):
            side = round(size * scale / 2)
            if side > min(self.coarse.shape):
                continue
            mask, norm = self.normalization(self.coarse, side)
            for angle in range(0, 360, 10):
                scores = self.search(self.coarse, roads, mask, side, angle)
                if scores is None:
                    continue
                scores *= norm
                # Retain a geographically distinct runner-up, even at the same rotation.
                for _ in range(2):
                    _, score, _, (px, py) = cv2.minMaxLoc(scores)
                    peaks.append((score, 2*px+side, 2*py+side, angle, side))
                    radius = max(4, side//3)
                    scores[max(0,py-radius):py+radius+1, max(0,px-radius):px+radius+1] = -2
        if not peaks:
            return None
        best, bx, by, angle, coarse_side = max(peaks)
        if best < .38 or any(score > best-.04 and np.hypot(px-bx, py-by) > coarse_side*.8
                              for score, px, py, _, _ in peaks):
            return None
        # Refine scale, translation and rotation near the winning coarse pose.
        refined = None
        for factor in (.92, 1., 1.08):
            side = round(2*coarse_side*factor)
            pad = side//2 + 8
            x0, y0 = max(0,bx-pad), max(0,by-pad)
            region = self.roads[y0:by+pad+1, x0:bx+pad+1]
            if min(region.shape) < side:
                continue
            mask, norm = self.normalization(region, side)
            for delta in range(-5, 6):
                rotation = (angle+delta) % 360
                scores = self.search(region, roads, mask, side, rotation)
                if scores is None:
                    continue
                _, score, _, (px, py) = cv2.minMaxLoc(scores * norm)
                candidate = (score, px+x0+side//2, py+y0+side//2, rotation)
                if refined is None or candidate > refined:
                    refined = candidate
        if refined and refined[0] >= .55:
            score, x, y, rotation = refined
            return x, y, rotation, score


def match_minimap(minimap, world):
    return MinimapMatcher(world).match(minimap)


class MapPreview:
    window = 'Destination map'

    def __init__(self, game):
        self.world = cv2.imread(str(MAP_PATH))
        if self.world is None:
            raise SystemExit(f'Cannot load map: {MAP_PATH}')
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
        if self.future is None and now-self.started >= .5:
            x, y, w, h = self.roi
            self.started = now
            self.future = self.worker.submit(self.matcher.match, game[y:y+h,x:x+w].copy())
        display = self.world.copy()
        label = 'Searching for position'
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
    screenshot = cv2.cvtColor(np.asarray(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
    print('Select the game area, then press Enter.')
    x, y, w, h = select_region('Select game area', screenshot)
    if not w or not h:
        raise SystemExit('No game area selected.')
    preview = MapPreview(screenshot[y:y+h,x:x+w])
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
