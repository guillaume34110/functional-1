from http.cookiejar import CookieJar
import json
import threading
import unittest
from urllib.request import build_opener, HTTPCookieProcessor, ProxyHandler, Request

import numpy as np
import torch

from play import handler_for, HTTPServer, NeuralSession, RELEASE
from functional_two import FunctionalTwoSession
from model import ACTIONS
from object_annotations import classify_board
from run import ROOT
from token_model import decode_tokens
from train_native_render import SCHEDULES, capture


class PublicPlayTest(unittest.TestCase):
    def test_zigzag_keeps_its_rotation_phase_at_both_walls(self):
        torch.set_num_threads(1)
        model = FunctionalTwoSession(RELEASE.with_name("functional-2")).model
        rom = ROOT / "roms/adjustris-v1.1.gb"
        turns = range(33, 70, 4)
        with torch.inference_mode():
            for wall in ("left_wall", "right_wall"):
                buttons = SCHEDULES[wall]
                frames = capture(rom, 4001, buttons)[:, 1::2, 1::2]
                start = next(t for t, frame in enumerate(frames) if (classify_board(frame) >= 0).sum() == 4)
                state = None
                for t in range(start, turns[-1] + 1):
                    frame = torch.from_numpy(frames[t:t+1, None].copy())
                    action = torch.tensor([ACTIONS.index(buttons[t])])
                    if t in turns:
                        with self.subTest(wall=wall, step=t):
                            actual, _ = model.predict_frame(frame, action, state)
                            np.testing.assert_array_equal(actual[0, 0, :, 8:48], frames[t+1, :, 8:48])
                    state = model.advance_observed_frame(frame, action, state)

    def test_bar_rotation_respects_both_walls(self):
        torch.set_num_threads(1)
        game = FunctionalTwoSession(RELEASE.with_name("functional-2"))
        model = game.model
        shapes = model.object_shapes.tolist()
        horizontal = shapes.index([[0, 0], [0, 1], [0, 2], [0, 3]])
        vertical = shapes.index([[0, 0], [1, 0], [2, 0], [3, 0]])

        def frame(kind, row, column):
            tokens = model.frame_tokens(game.frame)
            board = tokens[0, :, 2:12]
            board.fill_(model.blank_id)
            mask = model.shape_masks[kind, row * 10 + column].bool()
            board[mask] = model.object_tiles[kind]
            return decode_tokens(tokens, model.codebook)

        with torch.inference_mode():
            # These two poses give opposite outcomes in the ROM when B is pressed.
            for before, after in (((vertical, 1, 1), (vertical, 1, 1)),
                                  ((horizontal, 2, 6), (vertical, 1, 7))):
                with self.subTest(before=before):
                    actual, _ = model.predict_frame(frame(*before), torch.tensor([ACTIONS.index("b")]))
                    self.assertTrue(torch.equal(actual[:, :, :, 8:48], frame(*after)[:, :, :, 8:48]))

    def test_selected_level_changes_the_hud_and_the_opening_drop(self):
        torch.set_num_threads(1)
        game = FunctionalTwoSession(RELEASE.with_name("functional-2"))

        def first_drop(level):
            game.reset()
            for _ in range(abs(game.level - level)):
                game.step("right" if game.level < level else "left")
            self.assertEqual(game.level, level)
            game.step("start")
            self.assertEqual(int(game.model.counter.read(game.frame)[0, -1]), level)
            initial = classify_board(game.frame[0, 0].numpy())
            for tick in range(1, 25):
                game.step("noop")
                if not np.array_equal(initial, classify_board(game.frame[0, 0].numpy())):
                    return tick
            self.fail("opening piece did not move")

        self.assertGreater(first_drop(1), first_drop(9))
        game.reset()
        self.assertEqual(game.step("up")["level"], 9)

    def test_color_level_ten_starts_with_two_visible_digits(self):
        torch.set_num_threads(1)
        game = FunctionalTwoSession(RELEASE.with_name("functional-2"))
        game.step("start")
        game.frame = game.frame_for_level(10)
        self.assertEqual(game.model.counter.read(game.frame)[0, -2:].tolist(), [1, 0])
        game.step("noop")
        self.assertEqual(game.model.counter.read(game.frame)[0, -2:].tolist(), [1, 0])

    def test_two_visitors_can_play_and_restart_independently(self):
        torch.set_num_threads(1)
        seed = NeuralSession(RELEASE / "model.pt", RELEASE / "first-piece.npz")
        current = FunctionalTwoSession(RELEASE.with_name("functional-2"))
        with HTTPServer(("127.0.0.1", 0), handler_for(current, public=True, comparison=seed)) as server:
            url = f"http://127.0.0.1:{server.server_port}"
            worker = threading.Thread(target=server.serve_forever)
            worker.start()
            try:
                a, b = (build_opener(ProxyHandler({}), HTTPCookieProcessor(CookieJar())) for _ in range(2))

                def post(client, path, payload):
                    request = Request(url + path, json.dumps(payload).encode(),
                                      {"Content-Type": "application/json"})
                    with client.open(request, timeout=5) as response:
                        return json.load(response)

                with a.open(url, timeout=5) as response:
                    self.assertIn(b"Functional 2", response.read())
                self.assertEqual(post(a, "/v1/step", {"action": "right"})["steps"], 1)
                self.assertFalse(post(a, "/step", {"action": "start"})["menu"])
                self.assertEqual(post(a, "/step", {"action": "noop"})["steps"], 1)
                self.assertEqual(post(a, "/v1/step", {"action": "right"})["steps"], 2)
                self.assertEqual(post(b, "/v1/step", {"action": "left"})["steps"], 1)
                self.assertEqual(post(a, "/v1/reset", {}), seed.result(0))
                self.assertEqual(post(a, "/step", {"action": "noop"})["steps"], 2)
                self.assertEqual(post(b, "/v1/step", {"action": "down"})["steps"], 2)
                self.assertEqual(post(a, "/v1/step", {"action": "a"})["steps"], 1)
            finally:
                server.shutdown()
                worker.join()
