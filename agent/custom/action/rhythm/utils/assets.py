from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_image_root_cache: Path | None = None


def _get_image_root() -> Path:
    global _image_root_cache
    if _image_root_cache is not None:
        return _image_root_cache

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

    if candidates:
        logger.debug("图像资源根目录: %s (共 %d 个候选)", candidates[0], len(candidates))
        _image_root_cache = candidates[0]
        return candidates[0]

    fallback = here.parents[4] / "resource" / "base" / "image" / "auto_rhythm"
    logger.warning("未找到图像资源目录，回退到: %s", fallback)
    _image_root_cache = fallback
    return fallback


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _list_templates(subdir: str) -> list[tuple[str, Path]]:
    tpl_dir = _get_image_root() / subdir
    if not tpl_dir.is_dir():
        return []
    results: list[tuple[str, Path]] = []
    for p in sorted(tpl_dir.iterdir()):
        if p.suffix.lower() in _IMAGE_EXTS:
            results.append((p.stem, p))
    return results


def list_scene_templates(kind: str) -> list[tuple[str, Path]]:
    return _list_templates(f"scene_templates/{kind}")


def list_song_templates() -> list[tuple[str, Path]]:
    return _list_templates("song_templates")


def list_drum_templates() -> list[tuple[str, Path]]:
    tpl_dir = _get_image_root() / "drum_templates"
    if not tpl_dir.is_dir():
        logger.warning("鼓面模板目录不存在: %s", tpl_dir)
        return []
    results: list[tuple[str, Path]] = []
    for p in sorted(tpl_dir.iterdir()):
        if p.suffix.lower() in _IMAGE_EXTS:
            results.append((p.stem, p))
    return results


def read_image(p: Path) -> NDArray[np.uint8] | None:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        img_bytes = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    return img
