from __future__ import annotations

import os

import mido


class MidiProcessor:
    def parse(self, file_path: str, tracks: str | list[int] = "all") -> dict:
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".mid", ".midi"]:
            return self._parse_midi_with_mido(file_path, tracks)

        raise ValueError(
            f"Unsupported file format: {ext}. Only .mid and .midi are supported."
        )

    def _parse_midi_with_mido(self, file_path: str, tracks: str | list[int]) -> dict:
        mid = mido.MidiFile(file_path, clip=True)

        # 收集所有消息及其绝对 tick 位置和轨道索引
        all_events = []
        for idx, track in enumerate(mid.tracks):
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                all_events.append((abs_tick, idx, msg))

        all_events.sort(key=lambda x: x[0])

        # 确定要解析的轨道索引集合
        track_indices = set(self._resolve_track_indices(mid, tracks))

        # 全局跟踪 tempo，并将 tick 正确转换为秒
        tempo = 500000  # 默认 120 BPM
        last_tick = 0
        current_sec = 0.0
        active_notes: dict[tuple[int, int, int], tuple[float, int]] = {}
        pending_note_offs: set[tuple[int, int, int]] = set()
        sustain_pedals: dict[tuple[int, int], bool] = {}
        notes = []
        note_sequence = 0

        def close_note(key: tuple[int, int, int], end_sec: float) -> None:
            active_note = active_notes.pop(key, None)
            pending_note_offs.discard(key)
            if active_note is None:
                return
            start, order = active_note
            notes.append({
                "t": start,
                "p": key[2],
                "d": max(0.0, end_sec - start),
                "_order": order,
            })

        for abs_tick, idx, msg in all_events:
            # tick -> second 转换
            if abs_tick > last_tick:
                delta_tick = abs_tick - last_tick
                current_sec += mido.tick2second(delta_tick, mid.ticks_per_beat, tempo)
                last_tick = abs_tick

            if msg.type == "set_tempo":
                tempo = msg.tempo
                continue

            # 只处理选中的轨道
            if idx not in track_indices:
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                ch = getattr(msg, "channel", 0)
                if ch == 9:
                    continue
                key = (idx, ch, msg.note)
                # 如果同轨道同名音符已激活，先结束它（处理重叠）
                if key in active_notes:
                    close_note(key, current_sec)
                active_notes[key] = (current_sec, note_sequence)
                note_sequence += 1

            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                ch = getattr(msg, "channel", 0)
                if ch == 9:
                    continue
                key = (idx, ch, msg.note)
                if key not in active_notes:
                    continue
                if sustain_pedals.get((idx, ch), False):
                    pending_note_offs.add(key)
                else:
                    close_note(key, current_sec)

            elif msg.type == "control_change" and msg.control == 64:
                ch = getattr(msg, "channel", 0)
                if ch == 9:
                    continue
                pedal_key = (idx, ch)
                is_down = msg.value >= 64
                was_down = sustain_pedals.get(pedal_key, False)
                sustain_pedals[pedal_key] = is_down
                if was_down and not is_down:
                    for key in list(pending_note_offs):
                        if key[0] == idx and key[1] == ch:
                            close_note(key, current_sec)

        # 处理文件末尾仍未关闭的音符
        for key, (start, order) in list(active_notes.items()):
            notes.append({
                "t": start,
                "p": key[2],
                "d": max(0.05, current_sec - start),
                "_order": order,
            })

        notes.sort(key=lambda note: (note["t"], note["_order"]))
        for note in notes:
            note.pop("_order")

        return {
            "title": os.path.basename(file_path),
            "author": "Unknown",
            "bpm": self._estimate_bpm(mid),
            "duration": mid.length if mid.length else current_sec,
            "key": "Unknown",
            "notes": notes,
            "track_count": len(mid.tracks),
            "parsed_tracks": sorted(track_indices),
        }

    @staticmethod
    def _resolve_track_indices(mid: mido.MidiFile, tracks: str | list[int]) -> list[int]:
        if isinstance(tracks, list):
            return sorted({idx for idx in tracks if 0 <= idx < len(mid.tracks)})

        if tracks == "melody":
            return [MidiProcessor._detect_melody_track(mid)]

        # 默认 "all" —— 所有轨道
        return list(range(len(mid.tracks)))

    @staticmethod
    def _detect_melody_track(mid: mido.MidiFile) -> int:
        """自动检测主旋律轨道：选择 note_on 事件最多且非打击乐的轨道。"""
        best_idx = 0
        best_count = 0

        for idx, track in enumerate(mid.tracks):
            count = 0
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    ch = getattr(msg, "channel", 0)
                    if ch != 9:
                        count += 1
            if count > best_count:
                best_count = count
                best_idx = idx

        return best_idx

    @staticmethod
    def _estimate_bpm(mid: mido.MidiFile) -> int:
        """从 meta 消息中尝试提取 BPM，默认返回 120。"""
        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    return int(mido.tempo2bpm(msg.tempo))
        return 120
