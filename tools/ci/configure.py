from pathlib import Path

import shutil

assets_dir = Path(__file__).parent.parent.parent / "assets"


def configure_ocr_model():
    src = assets_dir / "MaaCommonAssets" / "OCR" / "ppocr_v4" / "zh_cn"
    dst = assets_dir / "resource" / "base" / "model" / "ocr"

    # 优先从 MaaCommonAssets 子模块同步 OCR 模型。
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return

    # 未拉取子模块时，允许直接使用仓库中已存在的模型目录继续构建。
    if dst.exists():
        print(f"[configure] OCR source missing, skip copy and use existing model: {dst}")
        return

    print(
        "[configure] OCR model not found, skip configure. "
        "You can init submodule with `git submodule update --init --recursive` "
        f"or place model files under `{dst}`."
    )


if __name__ == "__main__":
    configure_ocr_model()
