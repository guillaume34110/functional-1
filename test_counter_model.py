import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from collect_counters import GLYPHS
from counter_model import CounterReadout


class CounterPulseTest(unittest.TestCase):
    def test_deferred_receipt_is_consumed_once_alongside_a_new_receipt(self):
        glyphs = GLYPHS[:, 1::2, 1::2].copy()
        glyphs[8:10] = GLYPHS[8:10, ::2, ::2]
        addends = torch.zeros((3, 2, 5), dtype=torch.long)
        addends[1, 0, -1] = 4
        addends[2, 1, -1] = 1
        head = CounterReadout(torch.from_numpy(glyphs), torch.arange(11), addends, pulsed=True)
        with torch.no_grad():
            head.delay.weight[:, 0].copy_(torch.tensor([-1., -1., 1.]))
            head.level.weight.zero_()
            head.level.bias.fill_(-1)
        digits = torch.tensor([[10, 10, 10, 10, 0, 10, 10, 10, 10, 0, 10, 1]])

        def exact_digit_logits(source, delta, carry):
            total = source.masked_fill(source == 10, 0) + delta + carry
            updated = (total % 10).masked_fill((source == 10) & (total == 0), 10)
            return F.one_hot(updated + (total // 10) * 11, 22)

        with patch.object(head, "digit_logits", side_effect=exact_digit_logits):
            waiting, pending = head.consume(digits, torch.tensor([0]), torch.tensor([2]))
            torch.testing.assert_close(waiting, digits)
            self.assertEqual(pending.tolist(), [2])
            scored, pending = head.consume(waiting, pending, torch.tensor([1]))
            self.assertEqual(scored.tolist(), [[10, 10, 10, 10, 4, 10, 10, 10, 10, 1, 10, 1]])
            stable, pending = head.consume(scored, pending, torch.tensor([0]))
            torch.testing.assert_close(stable, scored)
            self.assertEqual(pending.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
