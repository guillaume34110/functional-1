from collections import Counter
import unittest

from diagnose_interventions import aggregate_step, finish


class DiagnosticAggregationTest(unittest.TestCase):
    def test_unknown_targets_are_masked_and_never_pass(self):
        counts = Counter()
        aggregate_step(counts, "validation/teacher/a", "first_effect", 2, 2, 0, -1, 0, True, 0)
        key = "validation/teacher/a/first_effect/joint_visible"
        report = finish(counts)
        self.assertEqual(report[key + "/unknown"], 1)
        self.assertEqual(report[key + "/total"], 0)
        self.assertEqual(report[key + "/correct"], 0)
        self.assertIsNone(report[key + "/rate"])

    def test_late_requested_step_does_not_count_earlier_correct_predictions(self):
        counts = Counter()
        prefix = "validation/autonomous/b"
        aggregate_step(counts, prefix, "last", 0, 7, 203, 203, 1, True, 0)
        self.assertFalse(counts)
        aggregate_step(counts, prefix, "last", 7, 7, 204, 203, 2, True, 2)
        report = finish(counts)
        self.assertEqual(report[prefix + "/last/shape/rate"], 1.)
        self.assertEqual(report[prefix + "/last/joint/rate"], 0.)
        self.assertEqual(report[prefix + "/last/joint_visible/rate"], 0.)
        self.assertEqual(report[prefix + "/last/joint/total"], 1)


if __name__ == "__main__":
    unittest.main()
