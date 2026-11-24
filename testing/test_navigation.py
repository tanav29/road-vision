import unittest

import cv2
import numpy as np

from navigation import MAP_PATH, MinimapMatcher


class NavigationTests(unittest.TestCase):
    def test_rotated_rescaled_radar_on_supplied_map(self):
        matcher = MinimapMatcher(cv2.imread(str(MAP_PATH)))
        for rotation, side in [(47, 100), (176, 140), (289, 70), (339, 180)]:
            with self.subTest(rotation=rotation, side=side):
                half = side // 2
                crop = matcher.roads[430-half:430+half, 300-half:300+half]
                radar = cv2.resize(crop * 255, (100, 100))
                transform = cv2.getRotationMatrix2D((49.5, 49.5), -rotation, 1)
                radar = cv2.warpAffine(radar, transform, (100, 100)).astype(np.uint8)
                cv2.circle(radar, (50, 50), 5, 255, -1)
                cv2.circle(radar, (50, 50), 47, 255, 3)
                result = matcher.match(cv2.cvtColor(radar, cv2.COLOR_GRAY2BGR))
                self.assertIsNotNone(result)
                x, y, angle, _ = result
                self.assertLessEqual(np.hypot(x-300, y-430), 2)
                self.assertLessEqual(abs((angle-rotation+180) % 360-180), 3)

    def test_repeated_positions_and_empty_radar(self):
        rng = np.random.default_rng(1)
        tile = np.full((100, 100, 3), 255, np.uint8)
        for x, y in rng.integers(10, 90, (30, 2)):
            cv2.line(tile, (x, y), (x+6, y+8), (0, 0, 0), 3)
        matcher = MinimapMatcher(np.tile(tile, (2, 2, 1)))
        self.assertIsNone(matcher.match(255-tile))
        self.assertIsNone(matcher.match(np.zeros_like(tile)))
