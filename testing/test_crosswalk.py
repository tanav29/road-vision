import unittest

from crosswalk import simulate


class CrosswalkTests(unittest.TestCase):
    def test_yield_and_resume(self):
        events, result = simulate("pedestrian", 8, 0.2)
        self.assertEqual(result, "PASS")
        self.assertAlmostEqual(events["first detected"].time - events["first visible"].time, 0.2)
        self.assertLess(events["stopped"].position, 0)
        self.assertEqual(events["stopped"].speed, 0)
        self.assertGreater(events["resume commanded"].time, events["crosswalk clear"].time)
        self.assertIn("route complete", events)

    def test_empty_crossing(self):
        events, result = simulate("empty", 8, 0.2)
        self.assertEqual(result, "PASS")
        self.assertEqual(list(events), ["route complete"])
        self.assertEqual(events["route complete"].speed, 8)

    def test_van_delays_visibility(self):
        clear, _ = simulate("pedestrian", 8, 0)
        hidden, result = simulate("van", 8, 0)
        self.assertEqual(result, "PASS")
        self.assertGreater(hidden["first visible"].time, clear["first visible"].time)
        self.assertEqual(hidden["first visible"].time, hidden["first detected"].time)

    def test_late_detection_fails(self):
        events, result = simulate("van", 12, 1)
        self.assertIn("entered before clear", events)
        self.assertTrue(result.startswith("FAIL"))
