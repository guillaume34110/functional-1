from collections import defaultdict
import unittest

import numpy as np
import torch

from diagnose_scene import audit, confidence, count_metric, finish
from scene_model import ObjectSceneDynamics
from test_scene_model import assets


class DiagnosticCountersTest(unittest.TestCase):
    def test_audit_accepts_saved_annotation_integer_dtypes(self):
        torch.set_num_threads(1)
        model = ObjectSceneDynamics(*assets())
        frames = np.zeros((35, 18, 20), dtype=np.uint8)
        actions = np.zeros(34, dtype=np.uint8)
        labels = {"cells": np.full((35, 4, 2), -1, dtype=np.int16),
                  "shape_id": np.full(35, -1, dtype=np.int16),
                  "evidence": np.full(35, 3, dtype=np.uint8)}
        result = audit(model, [(frames, actions)], [labels], 34)
        for protocol in ("teacher", "rollout8"):
            self.assertEqual(result[protocol]["steps"], 2)
            self.assertEqual(result[protocol]["current/visibility_false_positive/denominator"], 2)

    def test_masked_rates_and_empty_subset(self):
        stats = defaultdict(float)
        count_metric(stats, "error", torch.tensor([[True, False], [True, True]]),
                     torch.tensor([[True], [False]]))
        count_metric(stats, "empty", torch.tensor([True]), torch.tensor([False]))
        result = finish(stats)
        self.assertEqual(result["error/numerator"], 1)
        self.assertEqual(result["error/denominator"], 2)
        self.assertEqual(result["error/rate"], .5)
        self.assertIsNone(result["empty/rate"])

    def test_confidence_bins_partition_known_labels(self):
        stats = defaultdict(float)
        confidence(stats, "visible", torch.tensor([0., .1, .5, .9, 1.]),
                   torch.tensor([False, False, True, True, True]),
                   torch.tensor([True, True, False, True, True]))
        self.assertEqual(sum(v for k, v in stats.items() if k.endswith("/count")), 4)
        self.assertEqual(sum(v for k, v in stats.items() if k.endswith("/positive_sum")), 2)
        self.assertEqual(stats["visible/bin-0.9-1.01/count"], 2)


if __name__ == "__main__":
    unittest.main()
