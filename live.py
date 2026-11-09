"""Serve the comparison site with a server-clocked Functional 2 stream."""
import asyncio
from collections import deque
from http.cookies import CookieError
import json
import time

from aiohttp import WSMsgType, web

from model import ACTIONS


TICK = 1 / 15
HEADERS = {"Cache-Control": "no-store"}


def serve(site, host, port):
    async def page(request):
        if request.path == "/health":
            return web.Response(text="ok\n", headers=HEADERS)
        return web.Response(text=site.page(request.path.rstrip("/"), stream=True),
                            content_type="text/html", headers=HEADERS)

    def acquire(prefix, request):
        try:
            game, cookie = site.game(prefix, request.headers)
        except CookieError:
            raise web.HTTPBadRequest()
        except OverflowError:
            raise web.HTTPServiceUnavailable()
        if game in site.active:
            raise web.HTTPConflict()
        site.active.add(game)
        return game, cookie

    def release(game):
        site.active.remove(game)
        for key, (_, cached) in site.games.items():
            if cached is game:
                site.games.move_to_end(key)
                site.games[key] = time.monotonic(), game
                break

    async def command(request):
        prefix, _, name = request.path.rpartition("/")
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            payload = await request.json()
            action = payload["action"] if name == "step" else None
            if name == "step" and action not in ACTIONS:
                raise ValueError()
        except (KeyError, ValueError, TypeError):
            raise web.HTTPBadRequest()
        game, cookie = acquire(prefix, request)
        try:
            if name == "reset":
                result = await asyncio.to_thread(game.reset)
            else:
                if cookie:
                    await asyncio.to_thread(game.reset)
                result = await asyncio.to_thread(game.step, action)
            headers = {**HEADERS, "Set-Cookie": cookie} if cookie else HEADERS
            return web.json_response(result, headers=headers)
        finally:
            release(game)

    async def stream(request):
        if request.headers.get("Origin") not in (f"https://{request.host}", f"http://{request.host}"):
            raise web.HTTPForbidden()
        game, cookie = acquire("", request)
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=256)
        if cookie:
            ws.headers["Set-Cookie"] = cookie
        loop = asyncio.get_running_loop()
        held = "noop"
        pulses = deque(maxlen=16)
        paused = False
        previous = None

        async def send(result):
            nonlocal previous
            frame = result["frame"]
            if frame == previous:
                result.pop("frame")
            previous = frame
            await asyncio.wait_for(ws.send_json(result), timeout=5)

        try:
            await ws.prepare(request)
            if cookie:
                await send(await asyncio.to_thread(game.reset))
            else:
                await send(game.result(0.0))
            deadline = loop.time()
            while not ws.closed:
                ticking = not paused and (game.menu_screen is None or pulses or held != "noop")
                remaining = deadline - loop.time()
                if ticking and remaining <= 0:
                    deadline = loop.time() + TICK
                    result = await asyncio.to_thread(game.step, pulses.popleft() if pulses else held)
                    await send(result)
                    continue
                try:
                    message = await ws.receive(timeout=remaining if ticking else None)
                except TimeoutError:
                    continue
                if message.type != WSMsgType.TEXT:
                    break
                try:
                    data = json.loads(message.data)
                    if (data["held"] not in ACTIONS or data.get("pulse", "noop") not in ACTIONS
                            or type(data["paused"]) is not bool or type(data.get("reset", False)) is not bool):
                        raise ValueError()
                except (KeyError, ValueError, TypeError):
                    await ws.close(code=1008, message=b"invalid command")
                    break
                held, paused = data["held"], data["paused"]
                if paused or data.get("reset"):
                    pulses.clear()
                if data.get("reset"):
                    await send(await asyncio.to_thread(game.reset))
                if "pulse" in data and not paused:
                    pulses.append(data["pulse"])
        except (ConnectionError, TimeoutError):
            pass
        finally:
            release(game)
        return ws

    app = web.Application(client_max_size=256)
    app.add_routes([web.get(path, page) for path in ("/", "/v1", "/v1/", "/health")])
    app.add_routes([web.post(prefix + "/" + name, command)
                    for prefix in site.versions for name in ("step", "reset")])
    app.router.add_get("/stream", stream)
    web.run_app(app, host=host, port=port)
