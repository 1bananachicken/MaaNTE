from pathlib import Path

import shutil

assets_dir = Path(__file__).parent.parent.parent / "assets"


def configure_ocr_model():
    shutil.copytree(
        assets_dir / "MaaCommonAssets" / "OCR" / "ppocr_v5" / "zh_cn",
        assets_dir / "resource" / "base" / "model" / "ocr",
        dirs_exist_ok=True,
    )


def configure_classify_model():
    shutil.copytree(
        assets_dir / "MaaNTEModels" / "classify",
        assets_dir / "resource" / "base" / "model" / "classify",
        dirs_exist_ok=True,
    )


def configure_navi_model():
    shutil.copytree(
        assets_dir / "MaaNTEModels" / "navi",
        assets_dir / "resource" / "base" / "model" / "navi",
        dirs_exist_ok=True,
    )


def configure_all_models():
    configure_ocr_model()
    configure_classify_model()
    configure_navi_model()

    
if __name__ == "__main__":
    configure_ocr_model()
    configure_classify_model()
    configure_navi_model()
