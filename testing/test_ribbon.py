import unittest

import cv2
import numpy as np

from ribbon import RibbonPlanner


class RibbonTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((400, 640, 3), 100, np.uint8)
        self.frame[:180] = (180, 140, 100)
        self.planner = RibbonPlanner()

    def test_clear_unmarked_road_and_visible_projection(self):
        for i in range(4):
            keys, _ = self.planner.decide(self.frame, [], i*.1)
            self.assertEqual(keys, {'up'})
        overlay = self.frame.copy()
        self.planner.draw(overlay, keys)
        covered = np.any(overlay != self.frame, axis=2)
        self.assertGreater(covered[395].sum(), 3*covered[225].sum())
        self.assertFalse(covered[:210].any())

    def test_steers_away_from_edge_on_either_side_then_straightens(self):
        for flip, side in [(False, 'right'), (True, 'left')]:
            frame = self.frame.copy()
            cv2.line(frame, (300, 224), (275, 296), (245, 245, 245), 4)
            if flip:
                frame = cv2.flip(frame, 1)
            for i in range(4):
                self.assertEqual(self.planner.decide(frame, [], i*.1)[0], {'up', side})
            self.assertEqual(self.planner.decide(self.frame, [], .4)[0], {'up'})

    def test_uses_obstacle_footprint_and_only_passes_on_open_path(self):
        shoulder = ('person', 75, 220, 125, 380)
        leader = ('car', 300, 195, 340, 250)
        self.assertEqual(self.planner.decide(self.frame, [shoulder], 0)[0], {'up'})
        keys, _ = self.planner.decide(self.frame, [leader], .1)
        self.assertNotEqual(self.planner.bend, 0.)
        self.assertIn('up', keys)
        barrier = ('truck', 0, 190, 639, 250)
        self.assertEqual(self.planner.decide(self.frame, [barrier], .2)[0], set())
        close = ('car', 265, 225, 375, 370)
        self.assertEqual(self.planner.decide(self.frame, [close], .3)[0], {'down'})
        self.assertEqual(self.planner.decide(self.frame, [close], .55)[0], set())

    def test_obscured_and_nonroad_surface(self):
        self.assertFalse(self.planner.decide(np.zeros_like(self.frame), [], 0)[0])
        self.frame[225:295] = (25, 170, 30)
        self.assertFalse(self.planner.decide(self.frame, [], .1)[0])
