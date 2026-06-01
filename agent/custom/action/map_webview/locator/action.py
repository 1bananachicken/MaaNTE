import base64
import importlib.util
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from ...Common.logger import get_logger
from ...Navi.map_locator_ncc import MapLocationNccResult, MapLocatorNcc
from ...Navi.predict_angle import AnglePredictionResult, AnglePredictor
from ..calibration.action import MapCoordinateTransform, load_transform, parse_params

logger = get_logger(__name__)

DEFAULT_MAP_URL = "https://www.ghzs666.com/yh-map#/"

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_RESOURCE_ROOT = (
    _PROJECT_ROOT / "assets" if (_PROJECT_ROOT / "assets").exists() else _PROJECT_ROOT
)
DEFAULT_POINTER_PATH = (
    _RESOURCE_ROOT / "resource/base/image/map/map_webview_pointer.png"
)


def _positive_float(value: Any, default: float) -> float:
    try:
        return max(0.01, float(value))
    except (TypeError, ValueError):
        return default


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _location_payload(
    result: MapLocationNccResult,
    angle_result: AnglePredictionResult,
    transform: MapCoordinateTransform | None,
) -> dict[str, Any]:
    online_point = transform.apply(result.point) if transform and result.point else None
    return {
        "onlinePoint": online_point,
        "angle": angle_result.angle if angle_result.found else None,
    }


def _build_overlay_script(pointer_path: Path) -> str:
    pointer = base64.b64encode(pointer_path.read_bytes()).decode("ascii")
    script = Path(__file__).with_name("overlay.js").read_text(encoding="utf-8")
    return script.replace(
        "__MAANTE_POINTER_DATA_URL__",
        json.dumps(f"data:image/png;base64,{pointer}"),
    )


def _fetch_and_patch_map(
    map_url: str, overlay_script: str, state_url: str = ""
) -> bytes:
    """Fetch the upstream map page and inject overlay + state-polling script."""
    fetch_url = map_url.split("#")[0]
    parts = urlsplit(fetch_url)
    base_origin = f"{parts.scheme}://{parts.netloc}"

    req = urllib.request.Request(
        fetch_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=15.0) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    # Inject <base href="upstream_origin/"> so that all relative resource paths
    # (e.g. static/js/app.js, ./assets/...) resolve to the upstream origin rather
    # than to our local proxy server.  Root-relative paths (starting with /) are
    # NOT affected by <base>, so we still rewrite those explicitly below.
    base_tag = f'<base href="{base_origin}/">'
    if "<head>" in html:
        html = html.replace("<head>", f"<head>\n{base_tag}", 1)
    elif "<head " in html:
        # <head with attributes
        html = re.sub(
            r"(<head[^>]*>)", lambda m: m.group(1) + "\n" + base_tag, html, count=1
        )
    else:
        html = base_tag + "\n" + html

    # Rewrite all root-relative resource URLs (src=/ href=/) to full upstream URLs.
    # <base> does NOT affect absolute-path URLs (starting with /), so without this
    # the browser would try to fetch scripts/styles from our local server and get 404.
    def _rewrite_root_relative(m: re.Match) -> str:
        attr = m.group(1)
        quote = m.group(2)  # opening quote char (may be empty string)
        path = m.group(3)  # path starting with /
        # Skip protocol-relative URLs (//cdn.example.com/...)
        if path.startswith("//"):
            return m.group(0)
        # The closing quote (if any) was NOT consumed by the group — it remains in
        # the source HTML as-is, so we only emit attr=QUOTE origin+path here.
        return f"{attr}={quote}{base_origin}{path}"

    # Matches: src=/foo  href=/foo  src="/foo  src='/foo  (closing quote left in place)
    html = re.sub(
        r'(src|href)=(["\']?)(/[^"\'>\s]*)',
        _rewrite_root_relative,
        html,
    )

    # Polling script uses the absolute state_url so the fetch always reaches the
    # local proxy server regardless of what <base href> is set to.
    _state_url = state_url or "/state.json"
    inject = f"""
<script>
// MaaNTE overlay
{overlay_script}
</script>
<script>
(function () {{
  var _stateUrl = {json.dumps(_state_url)};
  function poll() {{
    fetch(_stateUrl, {{cache: 'no-store'}})
      .then(function (r) {{ return r.json(); }})
      .then(function (d) {{
        if (typeof window.__maanMapLocatorUpdate === 'function') {{
          window.__maanMapLocatorUpdate(d);
        }}
      }})
      .catch(function () {{}})
      .finally(function () {{ setTimeout(poll, 100); }});
  }}
  // Give the page JS time to initialise Leaflet before first call.
  document.addEventListener('DOMContentLoaded', function () {{ setTimeout(poll, 800); }});
  setTimeout(poll, 2000); // safety fallback if DOMContentLoaded already fired
}})();
</script>
"""
    if "</body>" in html:
        html = html.replace("</body>", inject + "\n</body>", 1)
    elif "</html>" in html:
        html = html.replace("</html>", inject + "\n</html>", 1)
    else:
        html += inject

    return html.encode("utf-8")


_BROWSER_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f1117; color: #e0e0e0; font-family: 'Segoe UI', sans-serif;
         display: flex; flex-direction: column; align-items: center;
         justify-content: center; min-height: 100vh; gap: 24px; }}
  h1 {{ font-size: 1.3rem; color: #7eb3ff; letter-spacing: .05em; }}
  .card {{ background: #1a1d27; border: 1px solid #2e3248; border-radius: 12px;
           padding: 28px 40px; min-width: 320px; text-align: center; }}
  .label {{ font-size: .75rem; color: #888; text-transform: uppercase;
            letter-spacing: .08em; margin-bottom: 6px; }}
  .value {{ font-size: 1.6rem; font-weight: 600; font-variant-numeric: tabular-nums;
            color: #fff; margin-bottom: 20px; }}
  .value.null {{ color: #555; font-size: 1rem; font-weight: 400; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
          background: #555; margin-right: 8px; vertical-align: middle;
          transition: background .3s; }}
  .dot.live {{ background: #4caf50; box-shadow: 0 0 6px #4caf5088; }}
  .map-btn {{ display: inline-block; margin-top: 8px; padding: 10px 28px;
              background: #2563eb; color: #fff; border-radius: 8px;
              text-decoration: none; font-size: .95rem; transition: background .2s; }}
  .map-btn:hover {{ background: #1d4ed8; }}
  .hint {{ font-size: .75rem; color: #666; }}
</style>
</head>
<body>
<h1>&#127757; {title}</h1>
<div class="card">
  <div class="label"><span class="dot" id="dot"></span>\u5b9e\u65f6\u5750\u6807</div>
  <div class="value" id="coord">\u7b49\u5f85\u6570\u636e\u2026</div>
  <div class="label">\u671d\u5411\u89d2\u5ea6</div>
  <div class="value" id="angle">\u2014</div>
  <a class="map-btn" href="{map_url}" target="_blank" rel="noopener">\u6253\u5f00\u5728\u7ebf\u5730\u56fe</a>
</div>
<div class="hint">\u6570\u636e\u6bcf 100\u202fms \u81ea\u52a8\u5237\u65b0 &nbsp;&middot;&nbsp; \u8bf7\u4fdd\u6301\u672c\u9875\u9762\u5f00\u542f</div>
<script>
async function refresh() {{
  try {{
    const r = await fetch('/state.json', {{ cache: 'no-store' }});
    const d = await r.json();
    const dot = document.getElementById('dot');
    const coord = document.getElementById('coord');
    const angle = document.getElementById('angle');
    dot.className = 'dot live';
    if (d.onlinePoint) {{
      coord.className = 'value';
      coord.textContent = 'X\u2009' + d.onlinePoint[0].toFixed(1) +
                          '\u2002Y\u2009' + d.onlinePoint[1].toFixed(1);
    }} else {{
      coord.className = 'value null';
      coord.textContent = '\u5750\u6807\u672a\u8bc6\u522b';
    }}
    angle.className = d.angle != null ? 'value' : 'value null';
    angle.textContent = d.angle != null ? d.angle.toFixed(1) + '\u00b0' : '\u2014';
  }} catch (_) {{
    document.getElementById('dot').className = 'dot';
  }}
}}
setInterval(refresh, 100);
refresh();
</script>
</body>
</html>
"""


class _MapStateHandler(BaseHTTPRequestHandler):
    server: "_MapStateServer"

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/state.json":
            content = self.server.read_state()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
        elif path in ("/map", "/map/"):
            content, error = self.server.get_proxy_html()
            if error:
                self.send_error(502, f"Failed to fetch upstream map: {error}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
        elif path in ("/", "/index.html"):
            content = self.server.read_page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class _MapStateServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        transform: MapCoordinateTransform | None,
        map_url: str = DEFAULT_MAP_URL,
        title: str = "MaaNTE Online Map",
        overlay_script: str = "",
    ):
        self._state_lock = threading.Lock()
        self._transform = transform
        self._state = b'{"onlinePoint": null, "angle": null}'
        self._page = _BROWSER_PAGE_TEMPLATE.format(title=title, map_url=map_url).encode(
            "utf-8"
        )
        self._map_url = map_url
        self._overlay_script = overlay_script
        self._proxy_lock = threading.Lock()
        self._proxy_html: bytes | None = None
        self._proxy_error: str | None = None
        super().__init__(("127.0.0.1", 0), _MapStateHandler)

    @property
    def state_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}/state.json"

    def update_location(
        self,
        result: MapLocationNccResult,
        angle_result: AnglePredictionResult,
    ) -> None:
        with self._state_lock:
            self._state = json.dumps(
                _location_payload(result, angle_result, self._transform),
                ensure_ascii=False,
            ).encode("utf-8")

    def read_state(self) -> bytes:
        with self._state_lock:
            return self._state

    def read_page(self) -> bytes:
        return self._page

    def get_proxy_html(self) -> tuple[bytes, str | None]:
        """Return (html_bytes, error_str). Fetches once and caches."""
        with self._proxy_lock:
            if self._proxy_html is not None or self._proxy_error is not None:
                return self._proxy_html or b"", self._proxy_error
            try:
                self._proxy_html = _fetch_and_patch_map(
                    self._map_url, self._overlay_script, self.state_url
                )
                self._proxy_error = None
            except Exception as exc:
                self._proxy_html = None
                self._proxy_error = str(exc)
                logger.error(f"Map proxy fetch failed: {exc}")
            return self._proxy_html or b"", self._proxy_error

    @property
    def page_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}/"

    @property
    def map_proxy_url(self) -> str:
        """Local URL that serves the upstream map with overlay injected."""
        host, port = self.server_address
        # Preserve the hash fragment (e.g. #/) from the original map_url
        fragment = self._map_url.partition("#")[2]
        suffix = f"#{fragment}" if fragment else ""
        return f"http://{host}:{port}/map{suffix}"


def _start_browser_viewer(server: _MapStateServer) -> None:
    """Open the proxied map page (with overlay injected) in the system browser."""
    webbrowser.open(server.map_proxy_url)


def _start_viewer(
    server: _MapStateServer,
    map_url: str,
    params: dict[str, Any],
) -> subprocess.Popen:
    command = [
        sys.executable,
        str(Path(__file__).with_name("window.py")),
        "--url",
        map_url,
        "--state-url",
        server.state_url,
        "--title",
        str(params.get("title") or "MaaNTE Online Map"),
        "--width",
        str(_positive_int(params.get("width"), 1280)),
        "--height",
        str(_positive_int(params.get("height"), 820)),
    ]
    if params.get("webview_debug"):
        command.append("--debug")
    if params.get("pointer_image"):
        command.extend(["--pointer-path", str(params["pointer_image"])])

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
    return subprocess.Popen(command, creationflags=creationflags)


def _stop_viewer(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


@AgentServer.custom_action("map_webview_locator")
class MapWebViewLocatorAction(CustomAction):
    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        params = parse_params(argv.custom_action_param)
        viewer_mode = str(params.get("viewer_mode") or "webview").strip().lower()
        use_browser = viewer_mode == "browser"

        if not use_browser and importlib.util.find_spec("webview") is None:
            logger.error(
                "Map webview requires pywebview. Install requirements.txt and retry."
                " (Or set viewer_mode=browser to use the system browser instead.)"
            )
            return CustomAction.RunResult(success=False)

        try:
            locator = MapLocatorNcc(
                big_map_path=params.get("big_map_path") or params.get("map_path"),
                debug=False,
            )
            predictor = AnglePredictor(
                backend=params.get("angle_backend") or params.get("backend"),
                pointer_roi=params.get("pointer_roi") or None,
                threshold=float(params.get("angle_threshold", 0.0)),
                debug=False,
            )
            angle_provider = predictor.provider_name()
            transform = load_transform(params)
            _map_url = str(params.get("map_url") or DEFAULT_MAP_URL)
            _title = str(params.get("title") or "MaaNTE Online Map")
            _pointer_path = (
                Path(str(params["pointer_image"])).expanduser()
                if params.get("pointer_image")
                else DEFAULT_POINTER_PATH
            )
            _overlay_script = _build_overlay_script(_pointer_path)
            server = _MapStateServer(
                transform,
                map_url=_map_url,
                title=_title,
                overlay_script=_overlay_script,
            )
        except Exception as exc:
            logger.error(f"Map webview locator init failed: {exc}")
            return CustomAction.RunResult(success=False)

        server_thread = threading.Thread(
            target=server.serve_forever,
            name="map-webview-state-server",
            daemon=True,
        )
        server_thread.start()

        if use_browser:
            try:
                _start_browser_viewer(server)
            except Exception as exc:
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=2.0)
                logger.error(f"Map browser viewer failed to open: {exc}")
                return CustomAction.RunResult(success=False)

            logger.info(
                f"Map browser locator started: map={locator.big_map_path}, "
                f"calibrated={transform is not None}, angle_provider={angle_provider}, "
                f"page={server.page_url}"
            )
            controller = context.tasker.controller
            update_interval = _positive_float(params.get("update_interval"), 0.1)
            try:
                while not context.tasker.stopping:
                    started = time.perf_counter()
                    frame = controller.post_screencap().wait().get()
                    if frame is not None:
                        result = locator.locate(frame)
                        angle_result = predictor.predict(frame)
                        server.update_location(result, angle_result)
                        logger.debug(
                            f"Map browser location: point={result.point}, "
                            f"score={result.score:.3f}, angle={angle_result.angle}"
                        )
                    sleep_time = update_interval - (time.perf_counter() - started)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            except Exception as exc:
                logger.error(f"Map browser locator failed: {exc}")
                return CustomAction.RunResult(success=False)
            finally:
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=2.0)
            return CustomAction.RunResult(success=True)

        try:
            process = _start_viewer(
                server,
                server.map_proxy_url,
                params,
            )
        except Exception as exc:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2.0)
            logger.error(f"Map webview process failed to start: {exc}")
            return CustomAction.RunResult(success=False)

        logger.info(
            f"Map webview locator started: map={locator.big_map_path}, "
            f"calibrated={transform is not None}, angle_provider={angle_provider}"
        )
        controller = context.tasker.controller
        update_interval = _positive_float(params.get("update_interval"), 0.1)
        exit_code = None
        try:
            while not context.tasker.stopping and process.poll() is None:
                started = time.perf_counter()
                frame = controller.post_screencap().wait().get()
                if frame is not None:
                    result = locator.locate(frame)
                    angle_result = predictor.predict(frame)
                    server.update_location(result, angle_result)
                    logger.debug(
                        f"Map webview location: point={result.point}, "
                        f"score={result.score:.3f}, angle={angle_result.angle}"
                    )

                sleep_time = update_interval - (time.perf_counter() - started)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            exit_code = process.poll()
        except Exception as exc:
            logger.error(f"Map webview locator failed: {exc}")
            return CustomAction.RunResult(success=False)
        finally:
            _stop_viewer(process)
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2.0)

        if exit_code not in (None, 0):
            logger.error(f"Map webview process exited unexpectedly: code={exit_code}")
            return CustomAction.RunResult(success=False)
        return CustomAction.RunResult(success=True)
