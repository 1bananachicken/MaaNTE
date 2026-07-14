#!/usr/bin/env python3

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "agent" / "cpp-navi" / "MaaUtils" / "tools"
MAADEPS = ROOT / "agent" / "cpp-navi" / "MaaUtils" / "MaaDeps"

if not (TOOLS / "maadeps_download.py").is_file():
    raise SystemExit(
        "MaaUtils submodule is missing. Run "
        "`git submodule update --init agent/cpp-navi/MaaUtils`."
    )

sys.path.insert(0, str(TOOLS))

from maadeps_download import detect_host_triplet, main as download_main


def has_development_files(triplet: str) -> bool:
    installed = MAADEPS / "vcpkg" / "installed" / f"maa-{triplet}"
    try:
        return installed.exists() and any(path.is_file() for path in installed.rglob("*"))
    except OSError:
        return False


def extract_cached_archives(triplet: str) -> bool:
    tarball = MAADEPS / "tarball"
    archives = []
    for component in ("devel", "runtime"):
        matches = sorted(tarball.glob(f"MaaDeps-{triplet}-{component}.tar.*"))
        if not matches:
            return False
        archives.append(matches[-1])

    print(f"Using cached MaaDeps archives for {triplet}")
    for archive in archives:
        print(f"Extracting {archive.name}")
        shutil.unpack_archive(archive, MAADEPS)
    return has_development_files(triplet)


if __name__ == "__main__":
    triplet = sys.argv[1] if len(sys.argv) > 1 else detect_host_triplet()
    if not has_development_files(triplet) and not extract_cached_archives(triplet):
        download_main(triplet, "MaaXYZ/MaaDeps", "v2.12.2")
