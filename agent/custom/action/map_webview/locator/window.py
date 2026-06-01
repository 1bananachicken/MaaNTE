import argparse
import base64
import json
import threading
import time
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[5]
RESOURCE_ROOT = (
    PROJECT_ROOT / "assets" if (PROJECT_ROOT / "assets").exists() else PROJECT_ROOT
)
DEFAULT_POINTER_PATH = RESOURCE_ROOT / "resource/base/image/map/map_webview_pointer.png"
POLL_INTERVAL = 0.1
PAGE_LOAD_TIMEOUT = 20.0
MAX_PAGE_RELOADS = 2


def _resolve_pointer_path(value: str | None) -> Path:
    if not value:
        return DEFAULT_POINTER_PATH
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _overlay_script(pointer_path: Path) -> str:
    pointer = base64.b64encode(pointer_path.read_bytes()).decode("ascii")
    script = Path(__file__).with_name("overlay.js").read_text(encoding="utf-8")
    return script.replace(
        "__MAANTE_POINTER_DATA_URL__",
        json.dumps(f"data:image/png;base64,{pointer}"),
    )


_LOADING_STYLE = (
    "document.documentElement.style.background='#1a1a1a';"
    "document.body&&(document.body.style.background='#1a1a1a');"
)


def _reload_page(window, page_url: str) -> None:
    try:
        window.load_url(page_url)
    except Exception:
        pass


def _poll_state(
    window,
    state_url: str,
    page_url: str,
    closed: threading.Event,
    script: str,
) -> None:
    page_loaded = threading.Event()

    def on_loaded() -> None:
        page_loaded.set()

    window.events.loaded += on_loaded

    load_deadline = time.monotonic() + PAGE_LOAD_TIMEOUT
    reloads = 0
    while not closed.is_set():
        if not page_loaded.is_set():
            if time.monotonic() >= load_deadline and reloads < MAX_PAGE_RELOADS:
                page_loaded.clear()
                _reload_page(window, page_url)
                reloads += 1
                load_deadline = time.monotonic() + PAGE_LOAD_TIMEOUT
            else:
                try:
                    window.evaluate_js(_LOADING_STYLE)
                except Exception:
                    pass
            closed.wait(POLL_INTERVAL)
            continue
        try:
            with urlopen(state_url, timeout=1.0) as response:
                payload = json.load(response)
            window.evaluate_js(
                f"{script}\nwindow.__maanMapLocatorUpdate("
                f"{json.dumps(payload, ensure_ascii=False)});"
            )
        except Exception:
            pass
        closed.wait(POLL_INTERVAL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--state-url", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--pointer-path")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        import webview
    except ImportError:
        return 1

    closed = threading.Event()
    window = webview.create_window(
        args.title, url=args.url, width=args.width, height=args.height
    )
    window.events.closed += closed.set
    webview.start(
        _poll_state,
        (
            window,
            args.state_url,
            args.url,
            closed,
            _overlay_script(_resolve_pointer_path(args.pointer_path)),
        ),
        debug=args.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
