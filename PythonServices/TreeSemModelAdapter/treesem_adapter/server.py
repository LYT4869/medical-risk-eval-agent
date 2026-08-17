from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .model import RequestValidationError, ServingBundlePredictor, TreeSemPredictor


LOGGER = logging.getLogger("treesem_adapter")
MAX_REQUEST_BYTES = 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RequestValidationError("duplicate JSON object key")
        result[key] = value
    return result


class TreeSemAdapterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], predictor: Any) -> None:
        super().__init__(address, TreeSemRequestHandler)
        self.predictor = predictor


class TreeSemRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", **self.server.predictor.metadata},
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        try:
            payload = self._read_json()
            result = self.server.predictor.predict(payload)
            self._send_json(HTTPStatus.OK, result)
        except RequestValidationError as error:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "message": str(error)},
            )
        except json.JSONDecodeError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_json"},
            )
        except Exception:
            LOGGER.exception("treeSem prediction failed")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "model_inference_failed"},
            )

    def _read_json(self) -> Any:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise RequestValidationError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise RequestValidationError("invalid Content-Length") from error
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise RequestValidationError("request body size is invalid")
        return json.loads(
            self.rfile.read(length).decode("utf-8"),
            object_pairs_hook=_unique_object,
        )

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status.value, status.phrase)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("client=%s " + format, self.client_address[0], *args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a trusted treeSem artifact")
    parser.add_argument("--bundle")
    parser.add_argument("--artifact")
    parser.add_argument("--code-root")
    parser.add_argument("--raw-file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.bundle:
        if any((args.artifact, args.code_root, args.raw_file)):
            raise SystemExit("--bundle cannot be combined with legacy artifact options")
        LOGGER.info("loading treeSem serving bundle")
        predictor = ServingBundlePredictor(args.bundle)
    else:
        if not all((args.artifact, args.code_root, args.raw_file)):
            raise SystemExit(
                "provide --bundle or all of --artifact, --code-root and --raw-file"
            )
        LOGGER.info("loading trusted legacy treeSem artifact: %s", args.artifact)
        predictor = TreeSemPredictor(
            artifact_path=args.artifact,
            code_root=args.code_root,
            raw_file=args.raw_file,
            device=args.device,
        )
    server = TreeSemAdapterServer((args.host, args.port), predictor)
    LOGGER.info("treeSem model adapter listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
