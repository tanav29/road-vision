"""Experimental minimap localization and destination picker for the browser game."""
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import ImageGrab

MAP_PATH = Path(__file__).resolve().parent.parent / 'map.webp'


def match_minimap(minimap, world):
    """Return an unverified (x, y, angle, score) candidate, or None.

    The game shows light roads; the supplied map draws roads dark. Matching
    uses a circular mask to exclude the minimap border and center marker.
    """
    h, w = minimap.shape[:2]
    if min(h, w) < 30:
        return None
    size = min(h, w)
    crop = minimap[(h-size)//2:(h+size)//2, (w-size)//2:(w+size)//2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    roads = (gray > 185).astype(np.uint8) * 255
    yy, xx = np.ogrid[:size, :size]
    radius = np.sqrt((xx-(size-1)/2)**2 + (yy-(size-1)/2)**2)
    mask = ((radius < size*.39) & (radius > size*.07)).astype(np.uint8) * 255
    world_gray = cv2.cvtColor(world, cv2.COLOR_BGR2GRAY)
    world_roads = (world_gray < 80).astype(np.uint8) * 255
    best = None
    for scale in (0.7, 1.0, 1.4, 1.8):
        side = max(12, round(size * scale))
        if side >= min(world.shape[:2]):
            continue
        template = cv2.resize(roads, (side, side), interpolation=cv2.INTER_NEAREST)
        scaled_mask = cv2.resize(mask, (side, side), interpolation=cv2.INTER_NEAREST)
        for angle in range(0, 360, 15):
            matrix = cv2.getRotationMatrix2D(((side-1)/2, (side-1)/2), angle, 1)
            rotated = cv2.warpAffine(template, matrix, (side, side), flags=cv2.INTER_NEAREST)
            rotated_mask = cv2.warpAffine(scaled_mask, matrix, (side, side), flags=cv2.INTER_NEAREST)
            if cv2.countNonZero(cv2.bitwise_and(rotated, rotated_mask)) < 20:
                continue
            result = cv2.matchTemplate(world_roads, rotated, cv2.TM_CCORR_NORMED, mask=rotated_mask)
            result = np.nan_to_num(result, nan=-1, posinf=-1, neginf=-1)
            _, score, _, point = cv2.minMaxLoc(result)
            candidate = (point[0] + side//2, point[1] + side//2, angle, score)
            if best is None or score > best[3]:
                best = candidate
    return best


def main():
    world = cv2.imread(str(MAP_PATH))
    if world is None:
        raise SystemExit(f'Cannot load map: {MAP_PATH}')
    screenshot = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
    print('Select the game area, then press Enter. Escape cancels.')
    gx, gy, gw, gh = cv2.selectROI('Select game area', screenshot, showCrosshair=True)
    cv2.destroyWindow('Select game area')
    if not gw or not gh:
        raise SystemExit('No game area selected.')
    game = screenshot[gy:gy+gh, gx:gx+gw]
    print('Select the entire circular minimap inside the game area, then press Enter.')
    mx, my, mw, mh = cv2.selectROI('Select minimap', game, showCrosshair=True)
    cv2.destroyWindow('Select minimap')
    if not mw or not mh:
        raise SystemExit('No minimap selected.')
    bbox = (gx+mx, gy+my, gx+mx+mw, gy+my+mh)
    destination = [None]
    window = 'Destination map - click to set destination; q to quit'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            destination[0] = (x, y)

    cv2.setMouseCallback(window, on_mouse)
    print('Click the map to set a destination. The blue position is an unverified candidate.')
    try:
        while True:
            start = time.monotonic()
            minimap = cv2.cvtColor(np.array(ImageGrab.grab(bbox=bbox)), cv2.COLOR_RGB2BGR)
            candidate = match_minimap(minimap, world)
            display = world.copy()
            if candidate:
                x, y, angle, score = candidate
                cv2.circle(display, (x, y), 8, (255, 60, 0), -1)
                cv2.putText(display, f'UNVERIFIED match {score:.2f}, rotation {angle} deg',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
                cv2.putText(display, f'UNVERIFIED match {score:.2f}, rotation {angle} deg',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            if destination[0]:
                cv2.drawMarker(display, destination[0], (0, 0, 255), cv2.MARKER_CROSS, 22, 3)
            cv2.imshow(window, display)
            if cv2.waitKey(max(1, int((1 - (time.monotonic()-start))*1000))) & 0xff == ord('q'):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
