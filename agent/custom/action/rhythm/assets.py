from __future__ import annotations

from pathlib import Path


def _project_image_root() -> Path:
    root = Path(__file__).resolve().parents[4]
    if (root / "assets").is_dir():
        return root / "assets" / "resource" / "base" / "image" / "auto_rhythm"
    return root / "resource" / "base" / "image" / "auto_rhythm"


def image_root() -> Path:
    return _project_image_root()


def list_scene_templates(kind: str) -> list[tuple[str, Path]]:
    tpl_dir = _project_image_root() / "scene_templates" / kind
    if not tpl_dir.is_dir():
        return []
    results: list[tuple[str, Path]] = []
    for p in sorted(tpl_dir.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            results.append((p.stem, p))
    return results


def list_song_templates() -> list[tuple[str, Path]]:
    tpl_dir = _project_image_root() / "song_templates"
    if not tpl_dir.is_dir():
        return []
    results: list[tuple[str, Path]] = []
    for p in sorted(tpl_dir.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            results.append((p.stem, p))
    return results
