import argparse
import base64
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen


_OVERLAY_SCRIPT = r"""
(() => {
  if (window.__maanMapLocatorUpdate) return;
  window.__maanMapLocatorCalibrationQueue = window.__maanMapLocatorCalibrationQueue || [];
  const pointerDataUrl = __MAANTE_POINTER_DATA_URL__;

  const state = {
    map: null,
    marker: null,
    latest: null,
    clickHooked: false,
    displayAngle: null,
  };

  const style = document.createElement('style');
  style.textContent = `
    .maan-player-marker {
      align-items: center;
      display: flex;
      height: 35px;
      justify-content: center;
      width: 30px;
    }
    .maan-player-marker img {
      filter: drop-shadow(0 2px 3px rgba(0, 0, 0, .65));
      height: 35px;
      image-rendering: pixelated;
      transform-origin: 50% 50%;
      transition: transform .1s linear;
      width: 30px;
    }
    #maan-map-locator-status {
      background: rgba(12, 18, 28, .86);
      border-radius: 5px;
      bottom: 10px;
      color: #eef4ff;
      font: 12px/1.5 Consolas, "Microsoft YaHei", sans-serif;
      left: 10px;
      max-width: min(720px, calc(100vw - 20px));
      padding: 5px 8px;
      pointer-events: none;
      position: fixed;
      z-index: 2147483647;
    }
  `;
  document.head.appendChild(style);

  const status = document.createElement('div');
  status.id = 'maan-map-locator-status';
  status.textContent = 'MaaNTE: 正在查找网页地图...';
  document.body.appendChild(status);

  function isLeafletMap(value) {
    return value
      && typeof value === 'object'
      && value._container
      && typeof value.addLayer === 'function'
      && typeof value.mouseEventToLatLng === 'function'
      && typeof value.latLngToLayerPoint === 'function';
  }

  function scan(root) {
    if (!root || typeof root !== 'object') return null;
    const queue = [root];
    const seen = new Set();
    let inspected = 0;
    while (queue.length && inspected < 6000) {
      const value = queue.shift();
      if (!value || typeof value !== 'object' || seen.has(value)) continue;
      seen.add(value);
      inspected += 1;
      if (isLeafletMap(value)) return value;
      let keys;
      try {
        keys = Object.keys(value);
      } catch (_error) {
        continue;
      }
      for (const key of keys) {
        if (key === '$parent' || key === '$root' || key === '_watcher') continue;
        let child;
        try {
          child = value[key];
        } catch (_error) {
          continue;
        }
        if (child && typeof child === 'object' && !seen.has(child)) queue.push(child);
      }
    }
    return null;
  }

  function findMap() {
    const roots = [];
    document.querySelectorAll('*').forEach(element => {
      if (element.__vue__) roots.push(element.__vue__);
    });
    for (const root of roots) {
      const map = scan(root);
      if (map) return map;
    }
    return null;
  }

  function hookLeaflet() {
    if (!window.L || !window.L.Map || window.__maanLeafletHooked) return;
    window.__maanLeafletHooked = true;
    ['invalidateSize', 'setView', 'panBy', 'flyTo', '_move', '_resetView'].forEach(name => {
      const original = window.L.Map.prototype[name];
      if (typeof original !== 'function') return;
      window.L.Map.prototype[name] = function(...args) {
        if (isLeafletMap(this)) state.map = this;
        return original.apply(this, args);
      };
    });
    window.dispatchEvent(new Event('resize'));
  }

  function calibrationPair(event) {
    if (!event.shiftKey) return;
    if (event.ctrlKey) {
      window.__maanMapLocatorCalibrationQueue.push({ reset: true });
      status.textContent = '正在清空旧标定点...';
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (!state.latest || !state.latest.point) return;
    const latLng = state.map.mouseEventToLatLng(event);
    const pair = {
      local: state.latest.point,
      online: [latLng.lat, latLng.lng],
    };
    const text = JSON.stringify(pair);
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).catch(() => {});
    }
    window.__maanMapLocatorCalibrationQueue.push(pair);
    status.textContent = `已采集标定点，正在保存: ${text}`;
    console.info('[MaaNTE calibration pair]', text);
    event.preventDefault();
    event.stopPropagation();
  }

  function ensureMap() {
    hookLeaflet();
    if (!state.map) state.map = findMap();
    if (!state.map) {
      status.textContent = 'MaaNTE: 正在查找网页 Leaflet 地图...';
      return false;
    }
    if (!state.clickHooked) {
      state.map._container.addEventListener('click', calibrationPair, true);
      state.clickHooked = true;
    }
    return true;
  }

  function ensureMarker() {
    if (state.marker || !window.L || !state.map) return;
    const icon = window.L.divIcon({
      className: '',
      html: `<div class="maan-player-marker"><img src="${pointerDataUrl}" alt=""></div>`,
      iconSize: [30, 35],
      iconAnchor: [15, 18],
    });
    state.marker = window.L.marker([0, 0], {
      icon,
      interactive: false,
      zIndexOffset: 1000000,
    }).addTo(state.map);
  }

  function updateMarkerAngle(angle) {
    if (!state.marker || !Number.isFinite(angle)) return;
    if (state.displayAngle === null) {
      state.displayAngle = angle;
    } else {
      const delta = ((angle - state.displayAngle + 540) % 360) - 180;
      state.displayAngle += delta;
    }
    const image = state.marker.getElement()?.querySelector('.maan-player-marker img');
    if (image) image.style.transform = `rotate(${state.displayAngle}deg)`;
  }

  function angleText(payload) {
    return payload.angleFound && Number.isFinite(payload.angle)
      ? `angle=${payload.angle.toFixed(1)}° conf=${payload.angleConfidence.toFixed(3)}`
      : `angle=未识别 conf=${payload.angleConfidence.toFixed(3)}`;
  }

  window.__maanMapLocatorUpdate = payload => {
    state.latest = payload;
    if (!ensureMap()) return;
    if (!payload.point) {
      if (state.marker) state.marker.setOpacity(0);
      status.textContent = `MaaNTE: 暂未定位 | score=${payload.score.toFixed(3)} | ${payload.mode} | ${angleText(payload)}`;
      return;
    }
    if (!payload.onlinePoint) {
      if (state.marker) state.marker.setOpacity(0);
      status.textContent = payload.calibrationIssue
        ? `MaaNTE: 标定质量不合格 | ${payload.calibrationIssue} | 按 Ctrl+Shift 点击地图清空旧标定`
        : `MaaNTE: 标定点=${payload.calibrationCount || 0}/3 | map.jpg=(${payload.point[0]}, ${payload.point[1]}) | 按住 Shift 点击网页中的同一地标`;
      return;
    }
    ensureMarker();
    state.marker.setOpacity(1);
    state.marker.setLatLng(payload.onlinePoint);
    updateMarkerAngle(payload.angle);
    status.textContent = `MaaNTE: map.jpg=(${payload.point[0]}, ${payload.point[1]}) | Leaflet=(${payload.onlinePoint[0].toFixed(5)}, ${payload.onlinePoint[1].toFixed(5)}) | 标定有效点=${payload.calibrationInliers}/${payload.calibrationCount} | rmse=${payload.calibrationRmse.toFixed(3)} | score=${payload.score.toFixed(3)} | ${angleText(payload)}`;
  };

  setInterval(ensureMap, 1000);
})();
"""

_CALIBRATION_DRAIN_SCRIPT = r"""
(() => {
  const queue = window.__maanMapLocatorCalibrationQueue || [];
  window.__maanMapLocatorCalibrationQueue = [];
  return queue;
})()
"""


def _resolve_pointer_path(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[3] / path

    return Path(__file__).with_name("map_webview_pointer.png")


def _pointer_data_url(pointer_path: Path) -> str:
    data = base64.b64encode(pointer_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _overlay_script(pointer_data_url: str) -> str:
    return _OVERLAY_SCRIPT.replace(
        "__MAANTE_POINTER_DATA_URL__",
        json.dumps(pointer_data_url),
    )


def _calibration_url(state_url: str) -> str:
    return state_url.rsplit("/", 1)[0] + "/calibration.json"


def _submit_calibration_pair(state_url: str, pair: dict) -> None:
    data = json.dumps(pair).encode("utf-8")
    request = Request(
        _calibration_url(state_url),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=1.0):
        pass


def _reset_calibration(state_url: str) -> None:
    request = Request(
        state_url.rsplit("/", 1)[0] + "/calibration/reset.json",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=1.0):
        pass


def _poll_state(
    window,
    state_url: str,
    closed: threading.Event,
    overlay_script: str,
) -> None:
    while not closed.is_set():
        try:
            with urlopen(state_url, timeout=1.0) as response:
                payload = json.load(response)
            script = (
                overlay_script
                + "\nwindow.__maanMapLocatorUpdate("
                + json.dumps(payload, ensure_ascii=False)
                + ");"
            )
            window.evaluate_js(script)
            pairs = window.evaluate_js(_CALIBRATION_DRAIN_SCRIPT) or []
            for pair in pairs:
                if pair.get("reset"):
                    _reset_calibration(state_url)
                else:
                    _submit_calibration_pair(state_url, pair)
        except Exception:
            pass
        closed.wait(0.1)


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
    overlay_script = _overlay_script(
        _pointer_data_url(_resolve_pointer_path(args.pointer_path))
    )
    window = webview.create_window(args.title, url=args.url, width=args.width, height=args.height)
    window.events.closed += closed.set
    webview.start(
        _poll_state,
        (window, args.state_url, closed, overlay_script),
        debug=args.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
