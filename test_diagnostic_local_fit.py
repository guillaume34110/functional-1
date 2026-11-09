import unittest

import numpy as np
import torch
from functools import partial

from diagnostic_local_fit import blocks, first_failing, fixed_context, metrics, select_bundle
from intervention_training import branch_objective
from test_intervention_training import fixture
from train_scene import scene_loss


class LocalFitDiagnosticTest(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_first_failing_predicate_checks_teacher_or_autonomous_and_stops(self):
        candidates = [{"record": {"parent_file": "train/000", "anchor": i}} for i in range(4)]
        calls = []
        def evaluate(candidate):
            index = candidate["record"]["anchor"]; calls.append(index)
            return index, [{"board_exact": True}], [{"board_exact": index != 1}]
        chosen, context, examined = first_failing(iter(candidates), evaluate)
        self.assertEqual(context, 1)
        self.assertEqual(chosen["record"]["anchor"], 1)
        self.assertEqual(calls, [0, 1])
        self.assertEqual(examined[-1]["baseline_board_errors"], {"teacher": 0, "autonomous": 1})

    def test_first_failing_scan_is_bounded_and_does_not_fit_without_failure(self):
        candidates = [{"record": {"parent_file": "train/000", "anchor": i}} for i in range(100)]
        calls = []
        def perfect(candidate):
            calls.append(candidate["record"]["anchor"])
            return None, [{"board_exact": True}], [{"board_exact": True}]
        with self.assertRaisesRegex(ValueError, "no fit run"):
            first_failing(iter(candidates), perfect)
        self.assertEqual(len(calls), 64)

    def test_selection_uses_only_targets_and_rejects_unknown_without_relaxation(self):
        model, bundle = fixture()
        model.scene_details = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("prediction forbidden during selection"))
        key, chosen = select_bundle({(1, 40): bundle, (0, 32): bundle}, model)
        self.assertEqual(key, (0, 32)); self.assertIs(chosen, bundle)
        bundle["labels"]["evidence"][5, 8] = 0
        with self.assertRaises(ValueError): select_bundle({(0, 32): bundle}, model)

    def test_exact_blocks_masked_metrics_and_fixed_detached_context(self):
        model, bundle = fixture(); model.eval()
        tokens = np.repeat(bundle["tokens"][0, :1], 4, axis=0)
        context = fixed_context(model, tokens, np.zeros(3, dtype=np.uint8), 3)
        self.assertFalse(context.requires_grad)
        initial = context.clone()
        loss_fn = partial(scene_loss, mode_weights=[.3, 20., 40.])
        components = blocks(model, bundle, context, loss_fn)
        actual = branch_objective(model, bundle, context, loss_fn)
        torch.testing.assert_close(components["scene"] + components["first_effect"], actual)
        actual.backward()
        self.assertTrue(torch.equal(context, initial))
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        bundle["labels"]["evidence"][0, 1] = 0
        rows, _ = metrics(model, bundle, context, False)
        row = next(r for r in rows if r["step"] == 0 and r["button"] == "noop")
        self.assertFalse(row["next"]["known"])
        self.assertIsNone(row["next"]["joint"])


if __name__ == "__main__": unittest.main()
