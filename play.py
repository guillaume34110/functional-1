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


def version_page(path, *, stream=False, browser=False):
    v1 = path == "/v1"
    page = PAGE if v1 else PAGE.replace("Functional 1", "Functional 2")
    v1_current = ' aria-current="page"' if v1 else ""
    v2_current = "" if v1 else ' aria-current="page"'
    page = page.replace("<!-- versions -->", f'<nav aria-label="Game version">'
                        f'<a href="/"{v2_current}>V2</a> '
                        f'<a href="/v1"{v1_current}>V1</a></nav>')
    if browser:
        return page.replace("const browserLocal=false", "const browserLocal=true")
    return page.replace("const streaming=false", "const streaming=true") if stream and not v1 else page


class GameSite:
    def __init__(self, session, comparison):
        self.games = OrderedDict()
        self.active = set()
        self.versions = {"": (session, "tetris_v2" if comparison is not None else "functional1")}
        if comparison is not None:
            self.versions["/v1"] = comparison, "tetris_v1"

    def page(self, path, *, stream=False):
        if "/v1" not in self.versions:
            return PAGE
        return version_page(path, stream=stream)

    def game(self, prefix, headers):
        template, name = self.versions[prefix]
        now = time.monotonic()
        for key, (seen, game) in tuple(self.games.items()):
            if now - seen > 300 and game not in self.active:
                del self.games[key]
        token = SimpleCookie(headers.get("Cookie", "")).get(name)
        key = (name, token.value) if token else None
        if key in self.games:
            _, game = self.games.pop(key)
            self.games[key] = now, game
            return game, None
        if len(self.games) >= 32:
            for key, (_, game) in self.games.items():
                if game not in self.active:
                    del self.games[key]
                    break
            else:
                raise OverflowError("too many games")
        token = token_urlsafe(24)
        game = copy(template)
        self.games[name, token] = now, game
        cookie = f"{name}={token}; Path={prefix or '/'}; HttpOnly; SameSite=Strict"
        if headers.get("X-Forwarded-Proto") == "https":
            cookie += "; Secure"
        return game, cookie


def handler_for(session, *, public=False, comparison=None):
    site = GameSite(session, comparison)
    versions = site.versions

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            if path != "/health" and path not in versions:
                self.send_error(404); return
            body = b"ok\n" if path == "/health" else site.page(path).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            try:
                prefix, _, command = self.path.rpartition("/")
                if prefix not in versions or command not in ("step", "reset"):
                    self.send_error(404)
                    return
                if public and self.headers.get_content_type() != "application/json":
                    self.send_error(415)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 256:
                    raise ValueError("request too large")
                payload = json.loads(self.rfile.read(length))
                action = payload["action"] if command == "step" else None
                if command == "step" and action not in ACTIONS:
                    raise ValueError("unknown action")
                game, cookie = self.game(prefix)
                if command == "step":
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

        def game(self, prefix):
            if not public:
                return versions[prefix][0], None
            game, cookie = site.game(prefix, self.headers)
            if cookie:
                game.reset()
            return game, cookie
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=RELEASE / "model.pt")
    parser.add_argument("--seed", type=Path, default=RELEASE / "first-piece.npz")
    parser.add_argument("--functional-2", type=Path, metavar="RELEASE")
    parser.add_argument("--compare", action="store_true", help="serve Functional 1 at /v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.compare and args.functional_2 is None:
        parser.error("--compare requires --functional-2")
    if args.functional_2 is None:
        session = NeuralSession(args.weights, args.seed)
    else:
        from functional_two import FunctionalTwoSession

        session = FunctionalTwoSession(args.functional_2)
    comparison = NeuralSession(args.weights, args.seed) if args.compare else None
    if args.compare:
        from live import serve
        serve(GameSite(session, comparison), args.host, args.port)
        return
    server = HTTPServer((args.host, args.port), handler_for(session, public=True, comparison=comparison))
    print(f"http://{args.host}:{args.port} — Ctrl-C to close", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
