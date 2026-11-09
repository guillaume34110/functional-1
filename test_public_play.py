from http.cookiejar import CookieJar
import json
import threading
import unittest
from urllib.request import build_opener, HTTPCookieProcessor, ProxyHandler, Request

import torch

from play import handler_for, HTTPServer, NeuralSession, RELEASE
from functional_two import FunctionalTwoSession


class PublicPlayTest(unittest.TestCase):
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
