import unittest

from audit_long import classify_identity, possible_delayed_rotation, summarize
from reclassify_latency import reclassify
import copy


class LongAuditTest(unittest.TestCase):
    def test_saved_latency_reclassification_is_narrow_immutable_and_absorbing(self):
        old = [[2, 3], [2, 4], [2, 5], [3, 4]]
        right = [[y, x + 1] for y, x in old]
        events = [{"status": "motion", "button": 2, "reason": None},
                  {"status": "violation", "button": 0, "reason": "uncommanded-horizontal-motion", "old_cells": old, "new_cells": right},
                  {"status": "motion", "button": 3, "reason": None},
                  {"status": "violation", "button": 3, "reason": "uncommanded-horizontal-motion", "old_cells": old, "new_cells": right},
                  {"status": "violation", "button": 3, "reason": "proposed-overlay-intersects-visible-background"}]
        original = copy.deepcopy(events)
        changed = reclassify(events, "noop")
        self.assertEqual(events, original)
        self.assertEqual(changed[1]["status"], "unknown")
        self.assertEqual(changed[1]["reason"], "possible-delayed-horizontal")
        self.assertEqual(changed[3]["status"], "violation")
        self.assertEqual(changed[4], original[4])
        result = summarize(changed)
        self.assertEqual(result["chain_status"], "inconclusive")
        self.assertEqual(result["known_geometry_prefix_steps"], 1)
        # Seven held-down steps do not carry the old horizontal command forward.
        held = [{"status": "motion", "button": 3} for _ in range(7)] + [events[3]]
        self.assertEqual(reclassify(held, "right")[-1]["status"], "violation")
    def test_proven_deposit_resets_identity_but_collision_or_unproved_style_does_not(self):
        deposit = {"status": "deposit", "collision": False, "reason": "old-canvas-retained-and-old-object-supported"}
        event, color = classify_identity(deposit, 7, 3)
        self.assertEqual(event["status"], "deposit")
        self.assertTrue(event["identity_reset"])
        self.assertEqual(color, 7)
        following, color = classify_identity({"status": "motion"}, 7, color)
        self.assertEqual(following["status"], "motion")
        collision, color = classify_identity({**deposit, "collision": True}, 7, 3)
        self.assertEqual(collision["status"], "violation")
        self.assertEqual(color, 3)
        unknown, color = classify_identity({"status": "motion"}, 7, 3)
        self.assertEqual(unknown["status"], "unknown")
        self.assertEqual(color, 3)
        summary = summarize([event, following])
        self.assertEqual(summary["counts"], {"deposit": 1, "motion": 1})
        self.assertEqual(summary["known_geometry_prefix_steps"], 2)
        self.assertEqual(summary["chain_status"], "no-observed-violation-gameplay-unverified")

    def test_noop_after_rotation_is_unknown_only_if_orientation_changes(self):
        old = [[2, 3], [2, 4], [2, 5], [3, 4]]
        rotated = [[2, 4], [3, 3], [3, 4], [4, 4]]
        translated = [[3, 3], [3, 4], [3, 5], [4, 4]]
        self.assertTrue(possible_delayed_rotation(old, rotated, 0, 4))
        self.assertTrue(possible_delayed_rotation(old, rotated, 0, 5))
        self.assertFalse(possible_delayed_rotation(old, translated, 0, 4))
        self.assertFalse(possible_delayed_rotation(old, rotated, 0, None))
        self.assertFalse(possible_delayed_rotation(old, rotated, 3, 4))
        self.assertFalse(possible_delayed_rotation(old, rotated, 0, 1))

    def test_unknown_breaks_chain_even_when_later_motion_looks_valid(self):
        result = summarize([{"status": "motion"}, {"status": "unknown", "reason": "spawn"},
                            {"status": "motion"}, {"status": "violation", "reason": "collision"}])
        self.assertEqual(result["known_geometry_prefix_steps"], 1)
        self.assertEqual(result["first_failure"]["step"], 1)
        self.assertEqual(result["first_violation"]["step"], 3)
        self.assertFalse(result["first_violation"]["before_first_unknown"])
        self.assertEqual(result["chain_status"], "inconclusive")

    def test_geometric_deposit_at_uncertain_spawn_never_becomes_a_valid_chain(self):
        result = summarize([{"status": "motion"}, {"status": "unknown", "reason": "style-change",
                            "previous_oracle": {"status": "deposit"}}, {"status": "motion"}])
        self.assertEqual(result["geometric_counts_including_uncertain_boundaries"]["deposit"], 1)
        self.assertEqual(result["counts"], {"motion": 2, "unknown": 1})
        self.assertEqual(result["known_geometry_prefix_steps"], 1)
        self.assertEqual(result["chain_status"], "inconclusive")

    def test_certain_violation_before_unknown_remains_a_failure(self):
        result = summarize([{"status": "motion"}, {"status": "violation", "reason": "collision"},
                            {"status": "unknown", "reason": "spawn"}, {"status": "motion"}])
        self.assertTrue(result["first_violation"]["before_first_unknown"])
        self.assertEqual(result["first_failure"]["step"], 1)
        self.assertEqual(result["known_geometry_prefix_steps"], 1)
        self.assertEqual(result["chain_status"], "violation")


if __name__ == "__main__":
    unittest.main()
