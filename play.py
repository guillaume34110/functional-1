"""Local keyboard interface backed exclusively by the trained neural dynamics."""
import argparse
import base64
from collections import OrderedDict
from copy import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
from http.cookies import CookieError, SimpleCookie
import json
from pathlib import Path
from secrets import token_urlsafe
import time

import torch
from model import ACTIONS, NeuralTetris
from run import read_episode


PAGE = Path(__file__).with_name("web").joinpath("index.html").read_text(encoding="utf-8")
RELEASE = Path(__file__).with_name("releases") / "functional-1"


class NeuralSession:
    def __init__(self, weights, seed):
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.tiles = "codebook" in state_dict
        if self.tiles:
            from token_model import CalibratedEventDynamics, TileDynamics, EventTileDynamics, ObservedHistoryDynamics
            if "_object_scene" in state_dict:
                from scene_model import scene_from_weights
                self.model = scene_from_weights(state_dict).eval()
                model_type = None
            elif "_observed_history" in state_dict:
                model_type = ObservedHistoryDynamics
            elif "gate_logit_offset" in state_dict:
                model_type = CalibratedEventDynamics
            else:
                model_type = EventTileDynamics if "change_head.weight" in state_dict else TileDynamics
            if model_type is not None:
                self.model = model_type(state_dict["codebook"]).eval()
        else:
            self.model = NeuralTetris().eval()
        if not self.tiles or model_type is not None:
            self.model.load_state_dict(state_dict)
        self.frames, self.actions, metadata = read_episode(seed)
        if metadata.get("source") != "capture":
            raise ValueError("initial prefix must be a reference capture")
        self.reset()

    def reset(self):
        self.state = None
        advance = getattr(self.model, "advance_observed_frame", None)
        with torch.inference_mode():
            for frame, action in zip(self.frames[:-1], self.actions):
                frame = torch.from_numpy(frame.copy())[None, None]
                action = torch.tensor([int(action)])
                if advance is None:
                    _, self.state = self.predict(frame, action, self.state)
                else:
                    self.state = advance(frame, action, self.state)
        self.frame = torch.from_numpy(self.frames[-1].copy())[None, None]
        self.steps = 0
        return self.result(0.0)

    def result(self, elapsed):
        return {"frame": base64.b64encode(self.frame[0, 0].numpy().tobytes()).decode(),
                "steps": self.steps, "inference_ms": elapsed}

    def predict(self, frame, action, state):
        if self.tiles:
            return self.model.predict_frame(frame, action, state)
        logits, state = self.model(frame, action, state)
        return logits.argmax(1, keepdim=True).to(torch.uint8), state

    def step(self, action):
        if action not in ACTIONS:
            raise ValueError("unknown action")
        started = time.perf_counter()
        with torch.inference_mode():
            self.frame, self.state = self.predict(self.frame, torch.tensor([ACTIONS.index(action)]), self.state)
        self.steps += 1
        return self.result((time.perf_counter() - started) * 1000)


def handler_for(session, *, public=False):
    games = OrderedDict()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path not in ("/", "/health"):
                self.send_error(404); return
            body = b"ok\n" if self.path == "/health" else PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            try:
                if self.path not in ("/step", "/reset"):
                    self.send_error(404)
                    return
                if public and self.headers.get_content_type() != "application/json":
                    self.send_error(415)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 256:
                    raise ValueError("request too large")
                payload = json.loads(self.rfile.read(length))
                action = payload["action"] if self.path == "/step" else None
                if self.path == "/step" and action not in ACTIONS:
                    raise ValueError("unknown action")
                game, cookie = self.game()
                if self.path == "/step":
                    result = game.step(action)
                else:
                    result = game.reset()
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                if cookie:
                    self.send_header("Set-Cookie", cookie)
                self.end_headers(); self.wfile.write(body)
            except (ValueError, KeyError, TypeError, CookieError):
                self.send_error(400, "invalid command")

        def game(self):
            if not public:
                return session, None
            now = time.monotonic()
            while games and now - next(iter(games.values()))[0] > 300:
                games.popitem(last=False)
            token = SimpleCookie(self.headers.get("Cookie", "")).get("functional1")
            if token and token.value in games:
                _, game = games.pop(token.value)
                games[token.value] = now, game
                return game, None
            if len(games) >= 32:
                games.popitem(last=False)
            key = token_urlsafe(24)
            game = copy(session)
            game.reset()
            games[key] = now, game
            cookie = f"functional1={key}; Path=/; HttpOnly; SameSite=Strict"
            if self.headers.get("X-Forwarded-Proto") == "https":
                cookie += "; Secure"
            return game, cookie
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=RELEASE / "model.pt")
    parser.add_argument("--seed", type=Path, default=RELEASE / "first-piece.npz")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    torch.set_num_threads(1)
    session = NeuralSession(args.weights, args.seed)
    server = HTTPServer((args.host, args.port), handler_for(session, public=True))
    print(f"http://{args.host}:{args.port} — Ctrl-C to close", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
