"""Open UE5 movement-packet coordinate decoder.

The game serializes client movement as a bit-packed timestamp, acceleration,
location and compressed control rotation. The movement block is not at a
stable bit offset across builds, so this module discovers the block and then
locks onto it using time, location and rotation continuity.
"""

from __future__ import annotations

import ipaddress
import ctypes
import math
import struct
import sys
import threading
import time
from typing import Any

__all__ = ("API_VERSION", "CoordinateCapture")

API_VERSION = "1.3.0"
_Vector3 = tuple[float, float, float]
_Pose = tuple[float, float, float, float, float]
_Flow = tuple[str, int, str, int, str]
_Candidate = tuple[float, int, _Vector3, _Vector3]
_NORTH = (-0.013752068070295848, -0.9999054358407049, 0.0)
_EAST = (0.9999054358407049, -0.01375206807029585, 0.0)
_MAX_LOCATION_ABS = 2_000_000.0
_MAX_ROTATION_ABS = 180.001


def _bits(data: bytes, offset: int, count: int) -> int:
    if offset < 0 or count < 0 or offset + count > len(data) * 8:
        raise ValueError("bit range is outside payload")
    first_byte = offset // 8
    last_byte = (offset + count + 7) // 8
    value = int.from_bytes(data[first_byte:last_byte], "little")
    return (value >> (offset % 8)) & ((1 << count) - 1)


def _vector(
    data: bytes,
    offset: int,
    scale: int,
) -> tuple[_Vector3, int, int, bool]:
    header = _bits(data, offset, 7)
    offset += 7
    width = header & 63
    scaled = bool(header >> 6)
    if width == 0:
        raise ValueError("unsupported full-precision vector")

    values: list[float] = []
    sign = 1 << (width - 1)
    modulus = 1 << width
    for _ in range(3):
        value = _bits(data, offset, width)
        offset += width
        if value & sign:
            value -= modulus
        values.append(value / scale if scaled else float(value))
    return (values[0], values[1], values[2]), offset, width, scaled


def _rotator(data: bytes, offset: int) -> tuple[_Vector3, int]:
    """Read FRotator::SerializeCompressedShort (Pitch, Yaw, Roll)."""
    values: list[float] = []
    for _ in range(3):
        present = _bits(data, offset, 1)
        offset += 1
        compressed = _bits(data, offset, 16) if present else 0
        if present:
            offset += 16
        angle = compressed * 360.0 / 65536.0
        if angle > 180.0:
            angle -= 360.0
        values.append(angle)
    return (values[0], values[1], values[2]), offset


def _has_valid_rotation(data: bytes, offset: int) -> bool:
    """Check the serialized control rotation without retaining duplicate data."""
    try:
        flags: list[bool] = []
        cursor = offset
        for _ in range(3):
            present = bool(_bits(data, cursor, 1))
            flags.append(present)
            cursor += 1 + (16 if present else 0)
        rotation, end = _rotator(data, offset)
    except (ValueError, OverflowError):
        return False
    if end > len(data) * 8:
        return False
    # ControlRotation serializes pitch and yaw, while roll is normally omitted
    # by the client. This presence pattern removes common random alignments.
    return (
        flags[1]
        and not flags[2]
        and abs(rotation[0]) <= 90.001
        and all(
            math.isfinite(value) and abs(value) <= _MAX_ROTATION_ABS
            for value in rotation
        )
    )


def _dot(left: _Vector3, right: _Vector3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _pose(location: _Vector3, rotation: _Vector3) -> _Pose:
    pitch = rotation[0]
    pitch_radians = math.radians(pitch)
    yaw_radians = math.radians(rotation[1])
    view_direction = (
        math.cos(pitch_radians) * math.cos(yaw_radians),
        math.cos(pitch_radians) * math.sin(yaw_radians),
        math.sin(pitch_radians),
    )
    north = _dot(view_direction, _NORTH)
    east = _dot(view_direction, _EAST)
    heading = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    return location[0], location[1], location[2], pitch, heading


def _distance_sq(left: _Vector3, right: _Vector3) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right))


def _is_localish_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Return whether an address is likely to belong to the local client side.

    ``ipaddress.is_private`` intentionally covers more than RFC1918/ULA ranges in
    modern Python versions. That is useful here because packet capture adapters
    may expose local traffic through non-global ranges such as 198.18.0.0/15.
    """
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
    )


def _packet_direction(source: str, destination: str) -> str:
    """Classify packet direction as ``c2s``, ``s2c`` or ``unknown``.

    The old payload dumper marked packets as ``unknown`` because it did not have
    a stable direction helper. The only case we can identify confidently from IP
    addresses alone is local-ish <-> global. Ambiguous cases are deliberately
    left as ``unknown`` instead of being guessed.
    """
    if not source or not destination:
        return "unknown"
    try:
        source_ip = ipaddress.ip_address(source)
        destination_ip = ipaddress.ip_address(destination)
    except ValueError:
        return "unknown"

    source_local = _is_localish_address(source_ip)
    destination_local = _is_localish_address(destination_ip)
    if source_local and not destination_local:
        return "c2s"
    if destination_local and not source_local:
        return "s2c"
    return "unknown"


def _is_windows_admin() -> bool:
    if sys.platform != "win32":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


class _Decoder:
    __slots__ = (
        "_flow",
        "_last_offset",
        "_last_capture",
        "_last_location",
        "_last_time",
        "_pending_at",
        "_pending_candidate",
        "_pending_flow",
        "_pending_seen",
    )

    def __init__(self) -> None:
        self._flow: _Flow | None = None
        self._last_offset: int | None = None
        self._last_capture: float | None = None
        self._last_time: float | None = None
        self._last_location: _Vector3 | None = None
        self._pending_flow: _Flow | None = None
        self._pending_candidate: _Candidate | None = None
        self._pending_seen = 0
        self._pending_at: float | None = None

    def decode(self, payload: bytes, timestamp: float, flow: _Flow) -> _Pose | None:
        candidates = self._candidates(payload)
        if not candidates:
            return None

        if self._flow is not None and flow != self._flow:
            candidate = self._new_flow_candidate(candidates)
            if candidate is None:
                return None
            selected = self._confirm_flow(flow, candidate, timestamp)
            if selected is None:
                return None
            self._clear_pending()
            self._flow = flow
        elif self._last_time is None or self._last_capture is None:
            selected = self._new_flow_candidate(candidates)
            if selected is None:
                return None
            self._clear_pending()
            self._flow = flow
        else:
            gap = max(0.0, timestamp - self._last_capture)
            expected = self._last_time + gap
            aligned = [item for item in candidates if item[1] == self._last_offset]
            tracking_candidates = aligned or candidates
            selected = min(
                tracking_candidates,
                key=lambda item: self._tracking_key(item, expected),
            )
            time_error = abs(selected[0] - expected)
            if time_error > 1.0:
                plausible = self._reacquire_candidates(tracking_candidates)
                if not plausible:
                    return None
                selected = self._fresh(plausible)
                self._clear_pending()
                self._flow = flow
            else:
                self._clear_pending()

        client_time, bit_offset, _, location = selected
        try:
            _, cursor, _, _ = _vector(payload, bit_offset + 32, 10)
            _, cursor, _, _ = _vector(payload, cursor, 100)
            rotation, _ = _rotator(payload, cursor)
        except (ValueError, OverflowError):
            return None
        self._last_time = client_time
        self._last_offset = bit_offset
        self._last_capture = timestamp
        self._last_location = location
        return _pose(location, rotation)

    def _candidates(self, payload: bytes) -> list[_Candidate]:
        output: list[_Candidate] = []

        # The movement block is bit-packed and can shift when optional replicated
        # fields are added. Known builds use offsets 197, 213, 230, 233, 236
        # and 252, but scanning every bit also handles an unseen field insertion.
        search_end = min(512, len(payload) * 8 - 60)
        for offset in range(190, search_end):
            try:
                client_time = struct.unpack(
                    "<f", _bits(payload, offset, 32).to_bytes(4, "little")
                )[0]
                acceleration, cursor, acceleration_bits, acceleration_scaled = _vector(
                    payload, offset + 32, 10
                )
                location, _, location_bits, location_scaled = _vector(
                    payload, cursor, 100
                )
            except (ValueError, OverflowError):
                continue
            if not math.isfinite(client_time) or not 0 <= client_time < 100_000:
                continue
            if not acceleration_scaled or not location_scaled:
                continue
            if not 1 <= acceleration_bits <= 16 or not 20 <= location_bits <= 32:
                continue
            if max(map(abs, acceleration)) >= 50_000:
                continue
            if max(map(abs, location)) > _MAX_LOCATION_ABS:
                continue
            # A false bit alignment can satisfy the two vector headers while
            # producing arbitrary trailing bits. Real movement records always
            # carry a valid compressed control rotation immediately afterwards.
            location_end = cursor + 7 + location_bits * 3
            if not _has_valid_rotation(payload, location_end):
                continue
            output.append((client_time, offset, acceleration, location))
        return output

    def _tracking_key(
        self, item: _Candidate, expected: float
    ) -> tuple[float, float, float]:
        time_error = abs(item[0] - expected)
        if self._last_location is None:
            return time_error, time_error, 0.0
        distance_sq = _distance_sq(item[3], self._last_location)
        # Time remains the primary signal, while the spatial term prevents a
        # different valid-looking record in a large packet from being selected.
        spatial_penalty = min(distance_sq / (5_000.0 * 5_000.0), 100.0)
        return time_error + spatial_penalty, time_error, distance_sq

    def _fresh(self, candidates: list[_Candidate]) -> _Candidate:
        if self._last_location is None:
            return max(candidates, key=lambda item: item[0])
        return min(
            candidates,
            key=lambda item: _distance_sq(item[3], self._last_location),
        )

    def _confirm_stream(
        self,
        flow: _Flow,
        candidate: _Candidate,
        timestamp: float,
    ) -> _Candidate | None:
        """Confirm a restarted movement stream before replacing valid state."""
        return self._confirm_flow(flow, candidate, timestamp)

    @staticmethod
    def _reacquire_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
        return [
            item
            for item in candidates
            if item[0] >= 0.01
            and item[1] <= 512
            and max(map(abs, item[2])) <= 10_000
            and max(map(abs, item[3])) <= _MAX_LOCATION_ABS
        ]

    @classmethod
    def _new_flow_candidate(cls, candidates: list[_Candidate]) -> _Candidate | None:
        valid = cls._reacquire_candidates(candidates)
        return max(valid, key=lambda item: item[0]) if valid else None

    def _confirm_flow(
        self,
        flow: _Flow,
        candidate: _Candidate,
        timestamp: float,
    ) -> _Candidate | None:
        if self._pending_flow != flow or self._pending_candidate is None:
            self._pending_flow = flow
            self._pending_candidate = candidate
            self._pending_seen = 1
            self._pending_at = timestamp
            return None

        previous = self._pending_candidate
        gap = max(0.0, timestamp - (self._pending_at or timestamp))
        time_delta = candidate[0] - previous[0]
        time_ok = time_delta >= 0.001 and abs(time_delta - gap) <= 0.5
        offset_ok = candidate[1] == previous[1]
        step_ok = _distance_sq(candidate[3], previous[3]) <= 6_400_000_000.0
        self._pending_seen = (
            self._pending_seen + 1 if time_ok and offset_ok and step_ok else 1
        )
        self._pending_candidate = candidate
        self._pending_at = timestamp
        return candidate if self._pending_seen >= 2 else None

    def _clear_pending(self) -> None:
        self._pending_flow = None
        self._pending_candidate = None
        self._pending_seen = 0
        self._pending_at = None


class CoordinateCapture:
    """Return the latest raw coordinate or camera pose from passive traffic."""

    __slots__ = (
        "_decoder",
        "_filter",
        "_interface",
        "_interval",
        "_capture_backend",
        "_lock",
        "_sample",
        "_sample_at",
        "_packet_count",
        "_payload_count",
        "_s2c_count",
        "_sample_count",
        "_last_packet_wall",
        "_last_payload_wall",
        "_last_sample_wall",
        "_sniffer",
    )

    def __init__(
        self,
        interface: str | None = None,
        packet_filter: str = "tcp port 30031 or udp",
        refresh_rate: float = 30.0,
        capture_backend: str = "pcap",
    ) -> None:
        normalized_backend = capture_backend.strip().lower()
        if normalized_backend == "scapy":
            normalized_backend = "pcap"
        if normalized_backend not in {"pcap", "pktmon"}:
            raise ValueError("capture_backend must be pcap or pktmon")
        self._interface = interface
        self._filter = packet_filter
        self._interval = 1.0 / refresh_rate if refresh_rate > 0 else 0.0
        self._capture_backend = normalized_backend
        self._decoder = _Decoder()
        self._lock = threading.Lock()
        self._sample: _Pose | None = None
        self._sample_at = 0.0
        self._packet_count = 0
        self._payload_count = 0
        self._s2c_count = 0
        self._sample_count = 0
        self._last_packet_wall = 0.0
        self._last_payload_wall = 0.0
        self._last_sample_wall = 0.0
        self._sniffer: Any | None = None

    def start(self) -> None:
        if self._sniffer is not None:
            return
        if self._capture_backend == "pktmon":
            self._start_pktmon()
            return
        self._start_pcap()

    def _accept_packet(
        self,
        payload: bytes,
        timestamp: float,
        flow: _Flow,
    ) -> None:
        direction = _packet_direction(flow[0], flow[2])
        now = time.time()
        with self._lock:
            self._packet_count += 1
            self._last_packet_wall = now
            if payload:
                self._payload_count += 1
                self._last_payload_wall = now
            if direction == "s2c":
                self._s2c_count += 1
        if not payload:
            return
        if direction == "s2c":
            return

        sample = self._decoder.decode(payload, timestamp, flow)
        if sample is None:
            return
        with self._lock:
            if now - self._last_sample_wall < self._interval:
                return
            self._sample = sample
            self._sample_at = timestamp
            self._sample_count += 1
            self._last_sample_wall = now

    def _start_pcap(self) -> None:
        try:
            from scapy.all import AsyncSniffer, IP, IPv6, Raw, TCP, UDP, conf
        except ImportError as exc:
            raise RuntimeError("coordinate capture requires scapy") from exc

        conf.use_pcap = True
        if not conf.use_pcap:
            raise RuntimeError("scapy libpcap provider is unavailable")
        layers = (IP, IPv6, Raw, TCP, UDP)

        def on_packet(packet: Any) -> None:
            if not packet.haslayer(layers[2]):
                return
            payload = bytes(packet[layers[2]].load)
            if not payload:
                return

            source = ""
            destination = ""
            for ip_type in layers[:2]:
                if packet.haslayer(ip_type):
                    source = str(packet[ip_type].src)
                    destination = str(packet[ip_type].dst)
                    break

            sport = 0
            dport = 0
            protocol = ""
            if packet.haslayer(layers[3]):
                transport = packet[layers[3]]
                sport = int(transport.sport)
                dport = int(transport.dport)
                protocol = "TCP"
            elif packet.haslayer(layers[4]):
                transport = packet[layers[4]]
                sport = int(transport.sport)
                dport = int(transport.dport)
                protocol = "UDP"

            timestamp = float(getattr(packet, "time", time.time()))
            self._accept_packet(
                payload,
                timestamp,
                (source, sport, destination, dport, protocol),
            )

        sniffer = AsyncSniffer(
            iface=self._interface or str(conf.iface),
            filter=self._filter,
            prn=on_packet,
            store=False,
        )
        sniffer.start()
        self._sniffer = sniffer

    def _start_pktmon(self) -> None:
        if not _is_windows_admin():
            raise RuntimeError(
                "pktmon capture requires administrator privileges; "
                "restart MaaNTE as administrator or use visual positioning"
            )
        try:
            from pktmon_interface import PktmonSniffer
        except ImportError as exc:
            raise RuntimeError("coordinate capture requires pktmon-interface") from exc

        def on_packet(packet: Any) -> None:
            protocol = str(getattr(packet, "protocol_name", ""))
            self._accept_packet(
                bytes(getattr(packet, "payload", b"")),
                float(getattr(packet, "timestamp", time.time())),
                (
                    str(getattr(packet, "source", "")),
                    int(getattr(packet, "sport", 0)),
                    str(getattr(packet, "destination", "")),
                    int(getattr(packet, "dport", 0)),
                    protocol if protocol in {"TCP", "UDP"} else "",
                ),
            )

        sniffer_kwargs = {
            "filter": self._filter,
            "prn": on_packet,
            "store": False,
            "read_timeout_ms": 20,
            "queue_size": 64,
            "native_queue_capacity": 256,
            "buffer_size_multiplier": 4,
            "truncation_size": 9000,
            "include_empty_payloads": False,
            "drain_batch_size": 512,
            "callback_batch_size": 8,
        }
        try:
            sniffer = PktmonSniffer(**sniffer_kwargs)
        except TypeError:
            sniffer = PktmonSniffer(
                filter=self._filter,
                prn=on_packet,
                store=False,
            )
        sniffer.start()
        self._sniffer = sniffer

    def stats(self) -> dict[str, float | int | str]:
        now = time.time()
        with self._lock:
            callback_error = getattr(self._sniffer, "callback_error", None)
            return {
                "backend": self._capture_backend,
                "packet_count": self._packet_count,
                "payload_count": self._payload_count,
                "s2c_count": self._s2c_count,
                "sample_count": self._sample_count,
                "packet_age": (
                    now - self._last_packet_wall if self._last_packet_wall else -1.0
                ),
                "payload_age": (
                    now - self._last_payload_wall if self._last_payload_wall else -1.0
                ),
                "sample_age": (
                    now - self._last_sample_wall if self._last_sample_wall else -1.0
                ),
                "callback_error": (
                    type(callback_error).__name__ if callback_error is not None else ""
                ),
            }
    def read(self, max_age: float = 1.0) -> _Pose | None:
        """Return (x, y, z, raw_pitch, compass_heading).

        ``compass_heading`` is mapped to the navigation frame with north at
        0 degrees and normalized to the half-open range [0, 360).
        """
        with self._lock:
            if self._sample is None or time.time() - self._last_sample_wall > max_age:
                return None
            return self._sample

    def close(self) -> None:
        sniffer = self._sniffer
        self._sniffer = None
        if sniffer is not None and getattr(sniffer, "running", False):
            sniffer.stop(join=True)
