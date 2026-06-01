import argparse
import json
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

POLL_INTERVAL = 0.1
PAGE_LOAD_TIMEOUT = 20.0
MAX_PAGE_RELOADS = 2
_DRAIN_SCRIPT = """
(() => {
  const queue = window.__maanMapCalibrationQueue || [];
  window.__maanMapCalibrationQueue = [];
  return queue;
})()
"""


def _overlay_script() -> str:
    return Path(__file__).with_name("overlay.js").read_text(encoding="utf-8")


def _post(state_url: str, suffix: str, payload: dict) -> None:
    request = Request(
        state_url.rsplit("/", 1)[0] + suffix,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=1.0):
        pass


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
            closed.wait(POLL_INTERVAL)
            continue
        try:
            with urlopen(state_url, timeout=1.0) as response:
                payload = json.load(response)
            window.evaluate_js(
                f"{script}\nwindow.__maanMapCalibrationUpdate("
                f"{json.dumps(payload, ensure_ascii=False)});"
            )
            for item in window.evaluate_js(_DRAIN_SCRIPT) or []:
                if item.get("reset"):
                    _post(state_url, "/calibration/reset.json", {})
                else:
                    _post(state_url, "/calibration.json", item)
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
        (window, args.state_url, args.url, closed, _overlay_script()),
        debug=args.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
