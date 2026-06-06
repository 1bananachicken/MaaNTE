import asyncio
import inspect
import json
import threading
import time
import logging
from typing import Any

from ..Common.logger import get_logger

logger = get_logger(__name__)

get_logger("websockets").setLevel(logging.WARNING)
get_logger("websockets.server").setLevel(logging.WARNING)
get_logger("websockets.client").setLevel(logging.WARNING)
get_logger("websockets.protocol").setLevel(logging.WARNING)


class NavigationWebSocketPublisher:
    def __init__(self, host="0.0.0.0", port="14514") -> None:
        self._host = str(host)
        self._port = int(port)
        self._state_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._started = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any | None = None
        self._ready_event = threading.Event()
        self._start_error: BaseException | None = None
        self._clients: set[Any] = set()
        self._state = {
            "type": "navi-state",
            "version": 1,
            "position": None,
            "angle": None,
            "angleConfidence": 0.0,
            "timestamp": 0.0,
        }

    def start(self, timeout: float = 5.0) -> None:
        should_wait = False
        with self._start_lock:
            if self._started:
                if self._start_error is not None:
                    raise RuntimeError(
                        "Navigation WebSocket failed to start"
                    ) from self._start_error
                return
            self._started = True
            self._ready_event.clear()
            self._start_error = None
            self._thread = threading.Thread(
                target=self._run_server,
                name="navi-websocket-server",
                daemon=True,
            )
            self._thread.start()
            should_wait = True

        if should_wait:
            if not self._ready_event.wait(timeout):
                self.stop()
                raise TimeoutError(
                    f"Navigation WebSocket start timed out: "
                    f"ws://{self._host}:{self._port}"
                )
            if self._start_error is not None:
                self.stop()
                raise RuntimeError(
                    f"Navigation WebSocket failed to start: "
                    f"ws://{self._host}:{self._port}"
                ) from self._start_error

    def stop(self, timeout: float = 3.0) -> None:
        with self._start_lock:
            if not self._started:
                return
            loop = self._loop
            thread = self._thread

        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._close_server(), loop)
            try:
                future.result(timeout=timeout)
            except Exception as exc:
                logger.warning(f"Navigation WebSocket close timed out: {exc}")
            loop.call_soon_threadsafe(loop.stop)

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

        with self._start_lock:
            self._started = False
            self._thread = None
            self._loop = None
            self._server = None
            self._clients.clear()
        logger.info(f"Navigation WebSocket stopped: ws://{self._host}:{self._port}")

    def publish_state(
        self,
        point: tuple[int, int] | None,
        *,
        score: float,
        mode: str,
        source_size: tuple[int, int] = (11264, 11264),
        angle: float | None,
        angle_confidence: float,
    ) -> None:
        self.start()
        with self._state_lock:
            self._state["position"] = (
                {
                    "pixelX": int(point[0]),
                    "pixelY": int(point[1]),
                    "score": float(score),
                    "mode": mode,
                    "sourceWidth": int(source_size[0]),
                    "sourceHeight": int(source_size[1]),
                }
                if point is not None
                else None
            )
            self._state["angle"] = float(angle) if angle is not None else None
            self._state["angleConfidence"] = float(angle_confidence)
            self._state["timestamp"] = time.time()
        self._schedule_broadcast()

    def _serialize_state(self) -> str:
        with self._state_lock:
            return json.dumps(self._state, ensure_ascii=False)

    def _schedule_broadcast(self) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_latest(), loop)

    async def _handle_client(self, websocket, _path=None) -> None:
        self._clients.add(websocket)
        try:
            await websocket.send(self._serialize_state())
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)

    async def _broadcast_latest(self) -> None:
        if not self._clients:
            return
        payload = self._serialize_state()
        clients = list(self._clients)
        results = await asyncio.gather(
            *(client.send(payload) for client in clients),
            return_exceptions=True,
        )
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                self._clients.discard(client)

    async def _close_server(self) -> None:
        clients = list(self._clients)
        if clients:
            await asyncio.gather(
                *(self._close_client(client) for client in clients),
                return_exceptions=True,
            )
        self._clients.clear()

        server = self._server
        if server is not None:
            server.close()
            try:
                await asyncio.wait_for(server.wait_closed(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("Navigation WebSocket server close timed out")
            self._server = None

    async def _close_client(self, client: Any) -> None:
        close_result = client.close()
        if inspect.isawaitable(close_result):
            await close_result

        wait_closed = getattr(client, "wait_closed", None)
        if wait_closed is not None:
            wait_result = wait_closed()
            if inspect.isawaitable(wait_result):
                await wait_result

    def _run_server(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            self._start_error = RuntimeError(
                "Navigation WebSocket is unavailable. Install requirements.txt and retry."
            )
            logger.error(str(self._start_error))
            self._ready_event.set()
            return

        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        async def start_server():
            return await serve(self._handle_client, self._host, self._port)

        try:
            self._server = loop.run_until_complete(start_server())
            self._ready_event.set()
            loop.run_forever()
        except Exception as exc:
            self._start_error = exc
            logger.error(f"Navigation WebSocket failed to start: {exc}")
            self._ready_event.set()
        finally:
            if self._server is not None:
                try:
                    loop.run_until_complete(self._close_server())
                except Exception as exc:
                    logger.warning(f"Navigation WebSocket cleanup failed: {exc}")
            self._loop = None
            loop.close()
