from __future__ import annotations

import os

import mido
from music21 import chord, converter, note


class MidiProcessor:
    def parse(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".mid", ".midi"]:
            return self._parse_midi_with_mido(file_path)
        if ext in [".xml", ".mxl"]:
            return self._parse_xml_with_music21(file_path)

        raise ValueError(f"Unsupported file format: {ext}")

    def _parse_midi_with_mido(self, file_path):
        mid = mido.MidiFile(file_path)
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

    def _parse_xml_with_music21(self, file_path):
        score = converter.parse(file_path)
        bpm = self._get_bpm(score)
        bpm_factor = 60.0 / bpm
        notes = []

        for element in score.flat.notes:
            t_sec = float(element.offset) * bpm_factor

            if isinstance(element, note.Note):
                notes.append({"t": t_sec, "p": element.pitch.midi})
            elif isinstance(element, chord.Chord):
                for chord_note in element.notes:
                    notes.append({"t": t_sec, "p": chord_note.pitch.midi})

        notes.sort(key=lambda item: item["t"])
        total_duration_sec = float(score.duration.quarterLength) * bpm_factor

        return {
            "title": os.path.basename(file_path),
            "author": "Unknown",
            "bpm": bpm,
            "duration": total_duration_sec,
            "key": str(score.analyze("key")),
            "notes": notes,
        }

    @staticmethod
    def _get_bpm(score):
        marks = score.metronomeMarkBoundaries()
        return marks[0][2].number if marks else 120
