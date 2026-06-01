(() => {
  if (window.__maanMapCalibrationUpdate) return;
  if (!document.head || !document.body) return;

  window.__maanMapCalibrationQueue = window.__maanMapCalibrationQueue || [];
  const state = {
    map: null,
    latest: null,
    clickContainer: null,
  };

  if (!document.querySelector('#maan-map-calibration-style')) {
    const style = document.createElement('style');
    style.id = 'maan-map-calibration-style';
    style.textContent = `
      #maan-map-calibration-panel {
        background: rgba(12, 18, 28, .9);
        border-radius: 6px;
        bottom: 12px;
        color: #eef4ff;
        font: 14px/1.5 "Microsoft YaHei", sans-serif;
        left: 12px;
        max-width: min(720px, calc(100vw - 24px));
        padding: 8px 10px;
        pointer-events: none;
        position: fixed;
        z-index: 2147483647;
      }
      #maan-map-calibration-panel strong {
        display: block;
        margin-bottom: 2px;
      }
    `;
    document.head.appendChild(style);
  }

  const panel = document.createElement('div');
  panel.id = 'maan-map-calibration-panel';
  document.body.appendChild(panel);

  function show(message) {
    panel.innerHTML = `<strong>MaaNTE 地图标定</strong>${message}`;
  }

  function isLeafletMap(value) {
    return value
      && typeof value === 'object'
      && value._container
      && typeof value.addLayer === 'function'
      && typeof value.mouseEventToLatLng === 'function';
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

  function collectPair(event) {
    if (!event.shiftKey) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.ctrlKey) {
      window.__maanMapCalibrationQueue.push({ reset: true });
      show('正在清空旧标定点...');
      return;
    }
    if (!state.latest?.point || !state.map) {
      show('尚未识别到游戏位置，请回到游戏地图界面后重试。');
      return;
    }
    const latLng = state.map.mouseEventToLatLng(event);
    window.__maanMapCalibrationQueue.push({
      local: state.latest.point,
      online: [latLng.lat, latLng.lng],
    });
    show('标定点已提交，等待保存...');
  }

  function adoptMap(map) {
    if (!isLeafletMap(map) || state.map === map) return;
    if (state.clickContainer) {
      state.clickContainer.removeEventListener('click', collectPair, true);
    }
    state.map = map;
    state.clickContainer = map._container;
    state.clickContainer.addEventListener('click', collectPair, true);
  }

  function hookLeaflet() {
    if (!window.L || !window.L.Map || window.__maanCalibrationLeafletHooked) return;
    window.__maanCalibrationLeafletHooked = true;
    if (typeof window.L.Map.addInitHook === 'function') {
      window.L.Map.addInitHook(function() {
        adoptMap(this);
      });
    }
    const createMap = window.L.map;
    if (typeof createMap === 'function') {
      window.L.map = function(...args) {
        const map = createMap.apply(this, args);
        adoptMap(map);
        return map;
      };
    }
    ['invalidateSize', 'setView', 'panBy', 'flyTo', '_move', '_resetView'].forEach(name => {
      const original = window.L.Map.prototype[name];
      if (typeof original !== 'function') return;
      window.L.Map.prototype[name] = function(...args) {
        adoptMap(this);
        return original.apply(this, args);
      };
    });
    window.dispatchEvent(new Event('resize'));
  }

  function ensureMap() {
    if (state.map && state.map._container?.isConnected === false) {
      state.map = null;
      state.clickContainer = null;
    }
    hookLeaflet();
    if (!state.map) adoptMap(findMap());
    return Boolean(state.map);
  }

  window.__maanMapCalibrationUpdate = payload => {
    state.latest = payload;
    if (!ensureMap()) {
      show('正在等待在线地图加载...');
      return;
    }
    if (!payload.point) {
      show('尚未识别到游戏位置，请回到游戏地图界面。');
      return;
    }
    if (payload.calibrationIssue) {
      show(`已保存 ${payload.calibrationCount} 个点，当前拟合未通过。继续采点，或按住 Ctrl + Shift 点击地图清空重来。`);
      return;
    }
    if (payload.calibrated) {
      show(`已保存 ${payload.calibrationCount} 个点，标定可用。移动到下一个地标后，可继续按住 Shift 点击对应位置。`);
      return;
    }
    show(`已保存 ${payload.calibrationCount}/3 个点。按住 Shift 点击当前游戏位置对应的网页地图地标。`);
  };

  show('正在等待在线地图加载...');
  ensureMap();
  setInterval(ensureMap, 1000);
})();
