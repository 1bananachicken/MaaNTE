from pathlib import Path


def resource_base_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        assets_base = parent / "assets" / "resource" / "base"
        if assets_base.exists():
            return assets_base

        resource_base = parent / "resource" / "base"
        if resource_base.exists():
            return resource_base

    raise FileNotFoundError("Unable to locate resource/base directory")
