import sys
import time
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from autodrive import Controller, decide, CTRL, ALT, ARM, STOP
from lane import game_lane


class FakeKeys:
    def __init__(self):
        self.focus = 1
        self.pressed_keys = set()
        self.events = []
    def foreground(self): return self.focus
    def pressed(self, key): return key in self.pressed_keys
    def key(self, name, down): self.events.append((name, down))


class DrivingTests(unittest.TestCase):
    def test_decision_obstacle_and_missing_lane(self):
        self.assertEqual(decide(None, [], 1000, 600, 0)[0], {'down'})
        self.assertEqual(decide(0, [('car', 400, 300, 600, 550)], 1000, 600, 0)[0], {'down'})
        self.assertEqual(decide(.3, [], 1000, 600, 0)[0], {'up', 'right'})
        self.assertEqual(decide(-.3, [], 1000, 600, 0)[0], {'up', 'left'})

    def test_synthetic_lane_and_blank(self):
        frame = np.zeros((600, 1000, 3), np.uint8)
        self.assertIsNone(game_lane(frame)[0])
        cv2.line(frame, (130, 590), (430, 330), (255, 255, 255), 8)
        cv2.line(frame, (870, 590), (570, 330), (255, 255, 255), 8)
        offset, _ = game_lane(frame)
        self.assertIsNotNone(offset)
        self.assertLess(abs(offset), .08)

    def test_watchdog_stop_and_release(self):
        backend = FakeKeys()
        ctl = Controller(backend, 1, timeout=.15)
        try:
            backend.pressed_keys.update({CTRL, ALT, ARM})
            time.sleep(.05)
            backend.pressed_keys.clear()
            ctl.update({'up', 'left'}, 'drive')
            backend.pressed_keys.update({CTRL, ALT, STOP})
            time.sleep(.06)
            self.assertFalse(ctl.armed)
            self.assertIn(('up', False), backend.events)
            self.assertIn(('left', False), backend.events)
            backend.pressed_keys.clear()
            time.sleep(.04)
            backend.pressed_keys.update({CTRL, ALT, ARM})
            time.sleep(.05)
            backend.pressed_keys.clear()
            ctl.update({'up'}, 'drive')
            backend.focus = 2
            time.sleep(.05)
            self.assertFalse(ctl.armed)
            self.assertFalse(ctl.held)
        finally:
            ctl.close()

    def test_timeout_and_close_release(self):
        backend = FakeKeys()
        ctl = Controller(backend, 1, timeout=.1)
        try:
            backend.pressed_keys.update({CTRL, ALT, ARM})
            time.sleep(.05)
            backend.pressed_keys.clear()
            ctl.update({'up'}, 'drive')
            time.sleep(.16)
            self.assertFalse(ctl.armed)
            self.assertIn(('up', False), backend.events)
        finally:
            ctl.close()

if __name__ == '__main__':
    unittest.main()
