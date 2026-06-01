(() => {
  if (window.__maanMapLocatorUpdate) return;
  if (!document.head) return;

  const pointerDataUrl = __MAANTE_POINTER_DATA_URL__;
  const state = {
    map: null,
    marker: null,
    displayAngle: null,
  };

  if (!document.querySelector('#maan-map-locator-style')) {
    const style = document.createElement('style');
    style.id = 'maan-map-locator-style';
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
    `;
    document.head.appendChild(style);
  }

  function isLeafletMap(value) {
    return value
      && typeof value === 'object'
      && value._container
      && typeof value.addLayer === 'function'
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

  function findMapInContainer(container) {
    // Leaflet stamps its container element with _leaflet_id and stores the map
    // instance in L.map's internal map keyed by that id (L.Map._leafletMaps or
    // accessible via the container's __maanLeafletMap tag we set in adoptMap).
    if (container.__maanLeafletMap && isLeafletMap(container.__maanLeafletMap)) {
      return container.__maanLeafletMap;
    }
    // Walk up the DOM for any Vue 2/3 instance that holds the map
    let el = container;
    while (el) {
      if (el.__vue__) {
        const m = scan(el.__vue__);
        if (m) return m;
      }
      const vue3 = el._vueParentComponent || el.__vueParentComponent || el._component;
      if (vue3) {
        const m = scan(vue3);
        if (m) return m;
      }
      el = el.parentElement;
    }
    return null;
  }

  function findMap() {
    // Strategy 1: start from known .leaflet-container elements (fastest, most reliable)
    const containers = document.querySelectorAll('.leaflet-container');
    for (const container of containers) {
      const m = findMapInContainer(container);
      if (m) return m;
    }
    // Strategy 2: scan all Vue 2 roots (original behaviour)
    const roots = [];
    document.querySelectorAll('*').forEach(element => {
      if (element.__vue__) roots.push(element.__vue__);
    });
    for (const root of roots) {
      const map = scan(root);
      if (map) return map;
    }
    // Strategy 3: scan window's own enumerable top-level properties directly
    for (const key of Object.keys(window)) {
      try {
        const val = window[key];
        if (isLeafletMap(val)) return val;
        if (val && typeof val === 'object') {
          const m = scan(val);
          if (m) return m;
        }
      } catch (_) { /* skip non-readable properties */ }
    }
    return null;
  }

  function adoptMap(map) {
    if (!isLeafletMap(map) || state.map === map) return;
    if (state.marker && typeof state.marker.remove === 'function') state.marker.remove();
    state.map = map;
    state.marker = null;
    // Tag the container so findMap() can locate this instance quickly next time
    try { map._container.__maanLeafletMap = map; } catch (_) {}
  }

  function hookLeaflet() {
    if (!window.L || !window.L.Map || window.__maanLeafletHooked) return;
    window.__maanLeafletHooked = true;
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
      if (state.marker && typeof state.marker.remove === 'function') state.marker.remove();
      state.map = null;
      state.marker = null;
    }
    hookLeaflet();
    if (!state.map) adoptMap(findMap());
    return Boolean(state.map);
  }

  function readiness() {
    if (!window.L) return 'waiting-for-leaflet';
    if (!state.map) return 'waiting-for-map';
    return 'ready';
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

  window.__maanMapLocatorUpdate = payload => {
    if (!ensureMap()) return readiness();
    if (!payload.onlinePoint) {
      if (state.marker) state.marker.setOpacity(0);
      return 'ready';
    }
    ensureMarker();
    state.marker.setOpacity(1);
    state.marker.setLatLng(payload.onlinePoint);
    updateMarkerAngle(payload.angle);
    return 'ready';
  };

  ensureMap();
  setInterval(ensureMap, 1000);
})();
