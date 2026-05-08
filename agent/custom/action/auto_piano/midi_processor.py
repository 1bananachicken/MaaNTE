from __future__ import annotations

import os

import mido


class MidiProcessor:
    def parse(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".mid", ".midi"]:
            return self._parse_midi_with_mido(file_path)

        raise ValueError(
            f"Unsupported file format: {ext}. Only .mid and .midi are supported."
        )

    def _parse_midi_with_mido(self, file_path):
        mid = mido.MidiFile(file_path, clip=True)
        notes = []
        current_time_sec = 0.0

        for msg in mid:
            current_time_sec += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                if msg.channel == 9:
                    continue

                notes.append(
                    {
                        "t": current_time_sec,
                        "p": msg.note,
                    }
                )

        notes.sort(key=lambda item: item["t"])

        return {
            "title": os.path.basename(file_path),
            "author": "Unknown",
            "bpm": 120,
            "duration": mid.length,
            "key": "Unknown",
            "notes": notes,
        }
