from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

from treesem_adapter.model import RequestValidationError
from treesem_adapter.server import TreeSemAdapterServer


class FakePredictor:
    metadata = {"service": "treeSem-model-adapter", "input_dim": 2}

    def predict(self, payload):
        if "sample_index" not in payload:
            raise RequestValidationError("sample_index is required")
        return {"model": "treeSem", "prediction": {"label": 1}}


class ServerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = TreeSemAdapterServer(("127.0.0.1", 0), FakePredictor())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_health(self):
        status, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_prediction(self):
        status, body = self.request("/v1/predict", {"sample_index": 0})
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], "treeSem")

    def test_validation_error(self):
        status, body = self.request("/v1/predict", {})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
