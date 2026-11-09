import unittest

import torch

from audit_overlay_context import OverlayContext
from scene_model import ObjectSceneDynamics
from test_scene_model import assets, chosen, synthetic_details


class OverlayContextTest(unittest.TestCase):
    def scenario(self, layer=False):
        model = ObjectSceneDynamics(*assets())
        context = OverlayContext(model, layer=layer)
        tokens = torch.zeros(1, 18, 20, dtype=torch.long)
        tokens[:, 2:4, 4:6] = 2  # Existing canvas under future X overlay.
        tokens[:, 7:9, 7:9] = 2  # Y: parser will wrongly identify this as current object.
        first = synthetic_details(model)
        first["current"] = chosen(model, 0, 0, 0, visible=False)
        first["next"] = chosen(model, 0, 2, 2)
        rendered, _ = context.decode(tokens, first)
        second = synthetic_details(model)
        second["current"] = chosen(model, 0, 7, 5)
        second["next"] = chosen(model, 0, 12, 5)
        return model, context, rendered, second

    @torch.inference_mode()
    def test_exact_output_reuses_x_preserving_y_and_external_input_is_original(self):
        model, context, rendered, second = self.scenario()
        prediction, evidence = context.decode(rendered, second)
        self.assertTrue(bool(evidence["matched"][0]))
        self.assertTrue(torch.all(prediction[:, 2:4, 4:6] == 0))
        self.assertTrue(torch.all(prediction[:, 7:9, 7:9] == 2))
        self.assertTrue(torch.all(evidence["original"][:, 7:9, 7:9] == 0))
        external = prediction.clone(); external[:, 0, 1] = 2
        actual, evidence = context.decode(external, second)
        self.assertFalse(bool(evidence["matched"][0]))
        torch.testing.assert_close(actual, model.decode_scene(external, second, hard=True))

    @torch.inference_mode()
    def test_layer_restores_underlying_canvas_and_compose_retains_deposit(self):
        model, context, rendered, second = self.scenario(layer=True)
        prediction, _ = context.decode(rendered, second)
        self.assertTrue(torch.all(prediction[:, 2:4, 4:6] == 2))
        self.assertTrue(torch.all(prediction[:, 7:9, 7:9] == 2))
        model, context, rendered, second = self.scenario(layer=True)
        second["mode"][:] = -20; second["mode"][:, 1] = 20
        deposited, _ = context.decode(rendered, second)
        self.assertTrue(torch.all(deposited[:, 2:4, 4:6] == 1))
        external = deposited.clone(); external[:, 0, 1] = 1
        actual, evidence = context.decode(external, second)
        self.assertFalse(bool(evidence["matched"][0]))
        torch.testing.assert_close(actual, model.decode_scene(external, second, hard=True))


if __name__ == "__main__":
    unittest.main()
