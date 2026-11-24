import sys
import time
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from autodrive import Controller, DrivePlanner, decide, CTRL, ALT, ARM, STOP
from lane import game_lane, LaneTracker


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
        self.assertEqual(decide(None, [], 1000, 600, 0)[0], set())
        self.assertEqual(decide(0, [('car', 400, 300, 600, 550)], 1000, 600, 0)[0], {'down'})
        self.assertEqual(decide(.3, [], 1000, 600, 0)[0], {'up', 'right'})
        self.assertEqual(decide(-.3, [], 1000, 600, 0)[0], {'up', 'left'})

    def test_close_objects_only_trigger_in_projected_lane(self):
        # At the bottom of the frame the lane bends left of image center.
        bounds = ((170, 420), (420, 590))
        for name in ('person', 'car'):
            for x1, x2, expected in [(397, 595, {'up'}), (210, 340, {'down'})]:
                planner = DrivePlanner()
                keys, _ = planner.decide(0, [(name, x1, 290, x2, 540)],
                                         1000, 600, 0, {}, bounds)
                self.assertEqual(keys, expected)

    def test_synthetic_lane_and_blank(self):
        frame = np.zeros((600, 1000, 3), np.uint8)
        self.assertIsNone(game_lane(frame)[0])
        cv2.line(frame, (130, 590), (430, 330), (255, 255, 255), 8)
        cv2.line(frame, (870, 590), (570, 330), (255, 255, 255), 8)
        offset, _ = game_lane(frame)
        self.assertIsNotNone(offset)
        self.assertLess(abs(offset), .08)

    def test_lane_tracking_resets_when_markings_disappear(self):
        tracker = LaneTracker()
        frame = np.zeros((600, 1000, 3), np.uint8)
        for start, end in [((130, 590), (430, 330)), ((870, 590), (570, 330))]:
            cv2.line(frame, start, end, (255, 255, 255), 8)
        self.assertIsNotNone(game_lane(frame, tracker)[0])
        self.assertIsNotNone(tracker.previous)
        self.assertIsNone(game_lane(np.zeros_like(frame), tracker)[0])
        self.assertIsNone(tracker.previous)

    def test_steady_inputs_and_steering_hysteresis(self):
        planner = DrivePlanner()
        for tick in range(40):
            self.assertEqual(planner.decide(.01, [], 1000, 600, tick/30, {})[0], {'up'})
        for i, offset in enumerate((.11, .08, .055, .085)):
            self.assertEqual(planner.decide(offset, [], 1000, 600, 1.4+i*.05, {})[0],
                             {'up', 'right'})
        self.assertEqual(planner.decide(.02, [], 1000, 600, 1.65, {})[0], {'up'})

    def test_marked_lane_pass_and_abort(self):
        frame = np.zeros((600, 1000, 3), np.uint8)
        for start, end in [((310,599),(450,330)), ((690,599),(550,330)), ((990,599),(582,330))]:
            cv2.line(frame, start, end, (255,255,255), 6)
        tracker = LaneTracker()
        offset, _ = game_lane(frame, tracker)
        self.assertEqual(set(tracker.adjacent), {'right'})
        planner = DrivePlanner()
        leader = [('car', 460, 290, 535, 370)]
        def tick(now, boxes):
            return planner.decide(offset, boxes, 1000, 600, now, tracker.adjacent)[0]
        self.assertEqual(tick(0, leader), set())
        self.assertEqual(tick(.18, leader), {'up', 'right'})
        self.assertEqual(tick(.24, []), {'up', 'right'})
        self.assertEqual(tick(.3, leader + [('truck', 700, 320, 880, 510)]), set())
        self.assertEqual(tick(.4, leader), set())
        self.assertEqual(tick(.5, [('person', 445, 290, 525, 440)]), {'down'})

    def test_pass_hands_back_to_lane_center_after_crossing(self):
        planner = DrivePlanner()
        lead = [('car', 460, 290, 535, 370)]
        for now in (0., .2):
            planner.decide(0., lead, 1000, 600, now, {'left': -.44})
        self.assertEqual(planner.side, 'left')
        planner.decide(-.24, [], 1000, 600, .3, {'left': -.7})
        self.assertIsNone(planner.side)
        self.assertEqual(planner.decide(0., [], 1000, 600, .4, {})[0], {'up'})

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

    def test_first_arm_binds_visible_game_and_rearm_retains_target(self):
        backend = FakeKeys()
        visible = [None]
        backend.capture_target = lambda point: visible[0]
        ctl = Controller(backend, capture_point=(400, 250))
        def chord():
            backend.pressed_keys.update({CTRL, ALT, ARM})
            time.sleep(.06)
            backend.pressed_keys.clear()
            time.sleep(.06)
        try:
            chord()
            self.assertFalse(ctl.armed)
            visible[0] = backend.focus = 2
            chord()
            self.assertTrue(ctl.armed)
            ctl.update({'up'}, 'drive')
            visible[0] = backend.focus = 3
            time.sleep(.06)
            self.assertIn(('up', False), backend.events)
            chord()
            self.assertFalse(ctl.armed)
            visible[0] = backend.focus = 2
            chord()
            self.assertTrue(ctl.armed)
        finally:
            ctl.close()

    def test_arm_modifiers_must_be_released_before_driving(self):
        backend = FakeKeys()
        ctl = Controller(backend, 1)
        try:
            backend.pressed_keys.update({CTRL, ALT, ARM})
            time.sleep(.05)
            self.assertTrue(ctl.armed)
            self.assertEqual(ctl.update({'up'}, 'drive'), 'ARMED: release Ctrl and Alt')
            self.assertEqual(backend.events, [])
            backend.pressed_keys.clear()
            ctl.update({'up'}, 'drive')
            self.assertIn(('up', True), backend.events)
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
