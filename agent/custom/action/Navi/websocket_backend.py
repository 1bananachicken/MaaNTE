import asyncio
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
        self._host = host
        self._port = port
        self._state_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._started = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[Any] = set()
        self._state = {
            "type": "navi-state",
            "version": 1,
            "position": None,
            "angle": None,
            "angleConfidence": 0.0,
            "timestamp": 0.0,
        }

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._started = True
            thread = threading.Thread(
                target=self._run_server,
                name="navi-websocket-server",
                daemon=True,
            )
            thread.start()

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

    def _run_server(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            logger.error(
                "Navigation WebSocket is unavailable. Install requirements.txt and retry."
            )
            return

        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        async def start_server():
            return await serve(self._handle_client, self._host, self._port)

        try:
            loop.run_until_complete(start_server())
            logger.info(
                f"Navigation WebSocket listening: ws://{self._host}:{self._port}"
            )
            loop.run_forever()
        except Exception as exc:
            logger.error(f"Navigation WebSocket failed to start: {exc}")
        finally:
            self._loop = None
            loop.close()
