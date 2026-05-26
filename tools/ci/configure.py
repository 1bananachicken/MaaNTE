from pathlib import Path

import shutil

assets_dir = Path(__file__).parent.parent.parent / "assets"


def configure_ocr_model():
    shutil.copytree(
        assets_dir / "MaaCommonAssets" / "OCR" / "ppocr_v4" / "zh_cn",
        assets_dir / "resource" / "base" / "model" / "ocr",
        dirs_exist_ok=True,
    )


if __name__ == "__main__":
    configure_ocr_model()
