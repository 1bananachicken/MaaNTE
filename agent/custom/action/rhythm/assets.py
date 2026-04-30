from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CANDIDATE_ROOTS: list[Path] | None = None


def _find_image_root() -> Path:
    global _CANDIDATE_ROOTS
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for i in range(len(here.parents)):
        root = here.parents[i]
        p1 = root / "resource" / "base" / "image" / "auto_rhythm"
        if p1.is_dir():
            candidates.append(p1)
        p2 = root / "assets" / "resource" / "base" / "image" / "auto_rhythm"
        if p2.is_dir():
            candidates.append(p2)

    _CANDIDATE_ROOTS = candidates

    if candidates:
        logger.debug("图像资源根目录: %s (共 %d 个候选)", candidates[0], len(candidates))
        return candidates[0]

    fallback = here.parents[4] / "resource" / "base" / "image" / "auto_rhythm"
    logger.warning("未找到图像资源目录，回退到: %s", fallback)
    return fallback


_image_root_cache: Path | None = None


def _project_image_root() -> Path:
    global _image_root_cache
    if _image_root_cache is None:
        _image_root_cache = _find_image_root()
    return _image_root_cache


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


def drum_templates_dir() -> Path:
    return _project_image_root() / "drum_templates"


def list_drum_templates() -> list[tuple[str, Path]]:
    tpl_dir = drum_templates_dir()
    if not tpl_dir.is_dir():
        logger.warning("鼓面模板目录不存在: %s", tpl_dir)
        return []
    results: list[tuple[str, Path]] = []
    for p in sorted(tpl_dir.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            results.append((p.stem, p))
    return results
