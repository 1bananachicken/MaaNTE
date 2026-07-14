from __future__ import annotations

import importlib.util
import json
import os
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any

from .logger import logger

_HOST = "127.0.0.1"
_DEFAULT_PORT = 14515
_MODULE_NAME = "nte_coordinate_api"
_API_VERSION = "1.2.0"
_MODULE_LOCK = threading.Lock()
_SERVER_LOCK = threading.Lock()
_MODULE: Any | None = None
_SERVER: socketserver.ThreadingTCPServer | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bridge_port() -> int:
    value = os.getenv("MAANTE_COORDINATE_BRIDGE_PORT", str(_DEFAULT_PORT))
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError("MAANTE_COORDINATE_BRIDGE_PORT must be between 1 and 65535")
    return port


def _load_coordinate_module() -> Any:
    global _MODULE
    with _MODULE_LOCK:
        if _MODULE is not None:
            return _MODULE

        thirdparty = _project_root() / "thirdparty"
        candidates = sorted(thirdparty.glob("nte_coordinate_api*.pyd"))
        if not candidates:
            raise RuntimeError("protected coordinate module is not installed")
        path_text = str(thirdparty)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

        candidate = candidates[-1]
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, candidate)
        if spec is None or spec.loader is None:
            raise RuntimeError("protected coordinate module spec is unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            if sys.modules.get(_MODULE_NAME) is module:
                sys.modules.pop(_MODULE_NAME, None)
            raise

        api_version = getattr(module, "API_VERSION", None)
        if api_version != _API_VERSION:
            raise RuntimeError(
                "protected coordinate API %s is required, got %s"
                % (_API_VERSION, api_version or "<unknown>")
            )
        if not callable(getattr(module, "CoordinateCapture", None)):
            raise RuntimeError("protected coordinate module has no CoordinateCapture")
        _MODULE = module
        return module


class _CoordinateHandler(socketserver.StreamRequestHandler):
    capture: Any | None = None

    def _reply(self, **payload: Any) -> None:
        self.wfile.write(
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
        self.wfile.flush()

    def _close_capture(self) -> None:
        capture = self.capture
        self.capture = None
        if capture is not None:
            capture.close()

    def handle(self) -> None:
        try:
            while True:
                raw = self.rfile.readline(1024 * 1024)
                if not raw:
                    break
                try:
                    request = json.loads(raw.decode("utf-8"))
                    request_type = str(request.get("type", ""))
                    if request_type == "start":
                        self._close_capture()
                        backend = str(request.get("backend", "")).strip().lower()
                        if backend not in {"pcap", "pktmon"}:
                            raise ValueError("capture backend must be pcap or pktmon")
                        module = _load_coordinate_module()
                        capture = module.CoordinateCapture(
                            refresh_rate=0,
                            capture_backend=backend,
                        )
                        capture.start()
                        self.capture = capture
                        self._reply(ok=True)
                    elif request_type == "read":
                        if self.capture is None:
                            raise RuntimeError("coordinate capture is not started")
                        pose = self.capture.read(max_age=float(request.get("max_age", 1.0)))
                        self._reply(
                            ok=True,
                            pose=None if pose is None else [float(value) for value in pose],
                        )
                    elif request_type == "stats":
                        stats = self.capture.stats() if self.capture is not None else {}
                        self._reply(ok=True, stats=stats)
                    elif request_type == "close":
                        self._close_capture()
                        self._reply(ok=True)
                        break
                    else:
                        raise ValueError("unknown coordinate bridge request")
                except Exception as exc:
                    logger.warning("Coordinate bridge request failed: %s", exc)
                    self._reply(ok=False, error=str(exc))
        finally:
            try:
                self._close_capture()
            except Exception as exc:
                logger.warning("Coordinate bridge cleanup failed: %s", exc)


class _CoordinateServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_coordinate_bridge() -> None:
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is not None:
            return
        server = _CoordinateServer((_HOST, _bridge_port()), _CoordinateHandler)
        thread = threading.Thread(
            target=server.serve_forever,
            name="coordinate-bridge",
            daemon=True,
        )
        thread.start()
        _SERVER = server
        logger.info("Coordinate bridge listening on %s:%d", _HOST, server.server_address[1])
