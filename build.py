#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaaNTE 本地自动化构建脚本
一键完成：下载依赖 → 配置资源 → 安装 → 打包

用法:
    python build.py                        # 默认构建
    python build.py --skip-mfa             # 跳过 MFAAvalonia GUI
    python build.py --skip-download        # 跳过下载，仅本地组装
    python build.py --tag v1.0.0           # 指定版本号
    python build.py --output-dir ./output  # 指定输出目录
"""

import os
import sys
import json
import shutil
import zipfile
import tarfile
import platform
import argparse
import subprocess
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# 可配置常量
# ---------------------------------------------------------------------------

MAA_FRAMEWORK_VERSION = "v5.10.2"
MFAA_VERSION = "v2.12.0"
PYTHON_VERSION_TARGET = "3.12.10"
PYTHON_BUILD_STANDALONE_RELEASE_TAG = "20250409"

ROOT = Path(__file__).resolve().parent
INSTALL_DIR = ROOT / "install"
DEPS_DIR = ROOT / "deps"
MFA_DIR = ROOT / "MFA"
PYTHON_DIR = INSTALL_DIR / "python"
PYTHON_DEPS_DIR = INSTALL_DIR / "deps"
ASSETS_DIR = ROOT / "assets"
TOOLS_DIR = ROOT / "tools" / "ci"

# platform tag mapping (pip-style)
PLATFORM_TAG_MAP = {
    ("Windows", "AMD64"): "win-x64",
    ("Windows", "x86_64"): "win-x64",
    ("Windows", "ARM64"): "win-arm64",
    ("Darwin", "x86_64"): "osx-x64",
    ("Darwin", "arm64"): "osx-arm64",
    ("Linux", "x86_64"): "linux-x64",
    ("Linux", "aarch64"): "linux-arm64",
}

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


# ========== 辅助函数 ==========

def run(cmd, cwd=None, check=True):
    """运行命令并实时输出"""
    print(f"  RUN: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=False, text=True)
    if check and result.returncode != 0:
        sys.exit(result.returncode)


def run_capture(cmd, cwd=None):
    """运行命令并捕获输出"""
    result = subprocess.run(
        cmd, cwd=cwd or ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
    return result


def download(url, dest):
    """下载文件"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    print(f"  -> {dest}")
    try:
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        print("  Download OK.")
    except Exception as e:
        print(f"  Download FAILED: {e}")
        raise


def get_platform():
    """获取当前平台信息"""
    os_type = platform.system()
    os_arch = platform.machine()

    # Windows ARM64 detection via PROCESSOR_IDENTIFIER
    if os_type == "Windows":
        proc_id = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if "ARMv8" in proc_id or "ARM64" in proc_id:
            os_arch = "ARM64"
        arch_map = {"AMD64": "AMD64", "x86_64": "AMD64", "ARM64": "ARM64", "aarch64": "ARM64"}
    elif os_type == "Darwin":
        arch_map = {"x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}
    elif os_type == "Linux":
        arch_map = {"x86_64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
    else:
        raise RuntimeError(f"不支持的操作系统: {os_type}")

    os_arch = arch_map.get(os_arch, os_arch)
    platform_tag = PLATFORM_TAG_MAP.get((os_type, os_arch))
    if not platform_tag:
        raise RuntimeError(f"无法确定平台标签: {os_type}/{os_arch}")

    return os_type, os_arch, platform_tag


# ========== 各步骤 ==========

def step_setup_python(os_type, os_arch):
    """步骤1: 安装嵌入式 Python + pip"""
    print("\n" + "=" * 60)
    print("[1/7] 安装嵌入式 Python")
    print("=" * 60)

    python_exe = PYTHON_DIR / ("python.exe" if os_type == "Windows" else "bin/python3")
    if python_exe.exists() and os_type == "Windows":
        print(f"  Python 已存在: {python_exe}，跳过安装。")
        return python_exe

    if PYTHON_DIR.exists():
        shutil.rmtree(PYTHON_DIR)
    PYTHON_DIR.mkdir(parents=True, exist_ok=True)

    if os_type == "Windows":
        # Windows: download embeddable Python from python.org
        win_arch = "amd64" if os_arch in ("AMD64", "x86_64") else "arm64"
        url = (
            f"https://www.python.org/ftp/python/{PYTHON_VERSION_TARGET}/"
            f"python-{PYTHON_VERSION_TARGET}-embed-{win_arch}.zip"
        )
        zip_path = PYTHON_DIR / f"python-embed-{win_arch}.zip"
        download(url, zip_path)
        shutil.unpack_archive(zip_path, PYTHON_DIR)
        zip_path.unlink()

        # 修改 ._pth 文件
        version_nodots = PYTHON_VERSION_TARGET.replace(".", "")[:3]
        pth_files = list(PYTHON_DIR.glob(f"python{version_nodots}._pth"))
        if not pth_files:
            pth_files = list(PYTHON_DIR.glob("python*._pth"))
        if not pth_files:
            raise FileNotFoundError(f"未在 {PYTHON_DIR} 中找到 ._pth 文件")

        pth_path = pth_files[0]
        print(f"  修改 ._pth: {pth_path}")
        content = pth_path.read_text(encoding="utf-8")
        content = content.replace("#import site", "import site")
        content = content.replace("# import site", "import site")
        for p in [".", "Lib", "Lib\\site-packages", "DLLs"]:
            if p not in content.splitlines():
                content += f"\n{p}"
        pth_path.write_text(content, encoding="utf-8")

    elif os_type in ("Darwin", "Linux"):
        # macOS/Linux: python-build-standalone
        pbs_arch = {"x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(
            os_arch, os_arch
        )
        os_slug = "apple-darwin" if os_type == "Darwin" else "unknown-linux-gnu"
        if os_type == "Linux":
            os_slug = (
                "unknown-linux-gnu"
                if pbs_arch == "x86_64"
                else f"aarch64-unknown-linux-gnu"
            )

        fname = (
            f"cpython-{PYTHON_VERSION_TARGET}+{PYTHON_BUILD_STANDALONE_RELEASE_TAG}-"
            f"{pbs_arch}-{os_slug}-install_only.tar.gz"
        )
        url = (
            f"https://github.com/indygreg/python-build-standalone/releases/download/"
            f"{PYTHON_BUILD_STANDALONE_RELEASE_TAG}/{fname}"
        )
        tar_path = PYTHON_DIR / fname
        download(url, tar_path)

        temp_dir = PYTHON_DIR / "_extract"
        temp_dir.mkdir(exist_ok=True)
        shutil.unpack_archive(tar_path, temp_dir)
        tar_path.unlink()

        # 移动 python/ 子目录内容
        inner_python = temp_dir / "python"
        if inner_python.is_dir():
            for item in inner_python.iterdir():
                shutil.move(str(item), str(PYTHON_DIR / item.name))
        shutil.rmtree(temp_dir)

        # 可执行权限
        bin_dir = PYTHON_DIR / "bin"
        if bin_dir.is_dir():
            for f in bin_dir.iterdir():
                if f.is_file():
                    f.chmod(f.stat().st_mode | 0o111)

    else:
        raise RuntimeError(f"不支持的操作系统: {os_type}")

    python_exe = PYTHON_DIR / ("python.exe" if os_type == "Windows" else "bin/python3")
    if not python_exe.exists():
        raise FileNotFoundError(f"Python 可执行文件未找到: {python_exe}")

    # 安装 pip
    print("  安装 pip...")
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    get_pip_path = PYTHON_DIR / "get-pip.py"
    download(get_pip_url, get_pip_path)
    run([str(python_exe), str(get_pip_path)])
    get_pip_path.unlink()

    print(f"  Python 就绪: {python_exe}")
    return python_exe


def step_download_python_deps(python_exe):
    """步骤2: 下载 Python 依赖 wheel"""
    print("\n" + "=" * 60)
    print("[2/7] 下载 Python 依赖")
    print("=" * 60)

    download_script = TOOLS_DIR / "download_deps.py"
    run([str(python_exe), str(download_script), "--deps-dir", str(PYTHON_DEPS_DIR)])


def step_download_maa_framework(os_arch):
    """步骤3: 下载 MaaFramework 原生库"""
    print("\n" + "=" * 60)
    print("[3/7] 下载 MaaFramework")
    print("=" * 60)

    if (DEPS_DIR / "bin").exists():
        print(f"  {DEPS_DIR}/bin 已存在，跳过下载（如需重新下载请删除 {DEPS_DIR}）")
        return

    arch_str = "x86_64" if os_arch in ("AMD64", "x86_64") else "arm64"
    url = (
        f"https://github.com/MaaXYZ/MaaFramework/releases/download/"
        f"{MAA_FRAMEWORK_VERSION}/MAA-win-{arch_str}-{MAA_FRAMEWORK_VERSION}.zip"
    )
    zip_path = ROOT / f"maa-framework-{arch_str}.zip"
    download(url, zip_path)
    DEPS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(zip_path, DEPS_DIR)
    zip_path.unlink()
    print(f"  MaaFramework 解压完成 -> {DEPS_DIR}")


def step_download_mfa(os_arch, platform_tag):
    """步骤4: 下载 MFAAvalonia GUI"""
    print("\n" + "=" * 60)
    print("[4/7] 下载 MFAAvalonia GUI")
    print("=" * 60)

    if MFA_DIR.exists():
        print(f"  {MFA_DIR} 已存在，跳过下载（如需重新下载请删除该目录）")
        return

    mfa_tag = "x64" if os_arch in ("AMD64", "x86_64") else "arm64"
    url = (
        f"https://github.com/MaaXYZ/MFAAvalonia/releases/download/"
        f"{MFAA_VERSION}/MFAAvalonia-{MFAA_VERSION}-win-{mfa_tag}.zip"
    )
    zip_path = ROOT / f"mfa-{mfa_tag}.zip"
    download(url, zip_path)
    MFA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(zip_path, MFA_DIR)
    zip_path.unlink()
    print(f"  MFAAvalonia 解压完成 -> {MFA_DIR}")


def step_convert_icon():
    """步骤5: 转换图标 (仅 x64 Windows)"""
    print("\n" + "=" * 60)
    print("[5/7] 转换图标")
    print("=" * 60)

    ico_path = ROOT / "maante.ico"
    if ico_path.exists():
        print(f"  {ico_path} 已存在，跳过转换。")
        return

    logo_path = ASSETS_DIR / "logo.png"
    if not logo_path.exists():
        print(f"  未找到 logo.png ({logo_path})，跳过图标转换。")
        return

    # 尝试 ImageMagick
    magick_cmd = "magick" if platform.system() == "Windows" else "convert"
    try:
        result = run_capture([magick_cmd, "convert", str(logo_path),
                              "-define", "icon:auto-resize=256,128,64,48,32,24,16",
                              str(ico_path)])
        if result.returncode == 0:
            print(f"  图标转换成功: {ico_path}")
        else:
            print("  ImageMagick 不可用，跳过 ICO 转换。请手动安装 ImageMagick。")
    except FileNotFoundError:
        print("  ImageMagick 未安装，跳过 ICO 转换。安装命令: choco install imagemagick")


def step_install(python_exe, tag, platform_tag):
    """步骤6: 运行 install.py 组装安装目录"""
    print("\n" + "=" * 60)
    print("[6/7] 组装安装目录")
    print("=" * 60)

    install_script = TOOLS_DIR / "install.py"
    run([str(python_exe), str(install_script), tag, platform_tag])


def step_copy_mfa():
    """步骤7: 复制 MFAAvalonia 文件到安装目录 + 处理图标"""
    print("\n" + "=" * 60)
    print("[7/7] 整合 MFAAvalonia GUI + 图标")
    print("=" * 60)

    if not MFA_DIR.exists():
        print("  MFA 目录不存在，跳过。")
        return

    # 复制 MFA 文件
    print(f"  复制 {MFA_DIR} -> {INSTALL_DIR}")
    for item in MFA_DIR.iterdir():
        src = str(item)
        dst = str(INSTALL_DIR / item.name)
        if item.is_dir():
            if (INSTALL_DIR / item.name).exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # 重命名 exe
    exe_path = INSTALL_DIR / "MFAAvalonia.exe"
    target_path = INSTALL_DIR / "MaaNTE.exe"
    if exe_path.exists():
        exe_path.rename(target_path)
        print(f"  重命名: {exe_path.name} -> {target_path.name}")
    elif (INSTALL_DIR / "MFAAvalonia").exists():
        (INSTALL_DIR / "MFAAvalonia").rename(INSTALL_DIR / "MaaNTE")
        print("  重命名: MFAAvalonia -> MaaNTE")

    # Windows: 注入管理员权限检查启动器
    if platform.system() == "Windows":
        app_path = INSTALL_DIR / "MaaNTE-app.exe"
        if target_path.exists():
            target_path.rename(app_path)
            print(f"  重命名: {target_path.name} -> {app_path.name}")
        for fname in ("MaaNTE_launcher.pyw", "MaaNTE.vbs", "MaaNTE.bat"):
            src = TOOLS_DIR / fname
            if src.exists():
                shutil.copy2(src, INSTALL_DIR / fname)
                print(f"  复制: {fname} -> install/{fname}")

    # 删除 MFA 自带 Assets
    assets_path = INSTALL_DIR / "Assets"
    if assets_path.exists():
        shutil.rmtree(assets_path)
        print("  删除 MFA 自带 Assets 目录")

    # 复制图标
    ico_path = ROOT / "maante.ico"
    if ico_path.exists():
        shutil.copy2(ico_path, INSTALL_DIR / "logo.ico")
        print("  复制 logo.ico -> install/logo.ico")


def step_package(platform_tag, tag, output_dir):
    """打包为 zip / tar.gz"""
    print("\n" + "=" * 60)
    print("打包")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os_type = platform.system()
    pkg_name = f"MaaNTE-{platform_tag}-{tag}"

    if os_type == "Windows":
        pkg_path = output_dir / f"{pkg_name}.zip"
        print(f"  创建 ZIP: {pkg_path}")
        with zipfile.ZipFile(pkg_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(INSTALL_DIR):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(INSTALL_DIR)
                    zf.write(file_path, arcname)
    else:
        pkg_path = output_dir / f"{pkg_name}.tar.gz"
        print(f"  创建 tar.gz: {pkg_path}")
        with tarfile.open(pkg_path, "w:gz") as tar:
            tar.add(INSTALL_DIR, arcname=".")

    print(f"  打包完成: {pkg_path}")
    return pkg_path


# ========== 主入口 ==========

def main():
    global MAA_FRAMEWORK_VERSION, MFAA_VERSION

    parser = argparse.ArgumentParser(description="MaaNTE 本地自动化构建脚本")
    parser.add_argument("--tag", default="v0.0.1", help="版本号 (默认: v0.0.1)")
    parser.add_argument("--skip-mfa", action="store_true", help="跳过 MFAAvalonia GUI 下载")
    parser.add_argument("--skip-download", action="store_true", help="跳过所有下载步骤")
    parser.add_argument("--skip-icon", action="store_true", help="跳过图标转换")
    parser.add_argument("--skip-package", action="store_true", help="跳过最终打包")
    parser.add_argument("--output-dir", default=str(ROOT / "output"), help="输出目录 (默认: ./output)")
    parser.add_argument("--maa-version", default=MAA_FRAMEWORK_VERSION,
                        help=f"MaaFramework 版本 (默认: {MAA_FRAMEWORK_VERSION})")
    parser.add_argument("--mfa-version", default=MFAA_VERSION,
                        help=f"MFAAvalonia 版本 (默认: {MFAA_VERSION})")

    args = parser.parse_args()
    MAA_FRAMEWORK_VERSION = args.maa_version
    MFAA_VERSION = args.mfa_version

    # 初始化 git 子模块
    submodules_ok = (ASSETS_DIR / "MaaCommonAssets" / ".git").exists()
    if not submodules_ok:
        print("正在初始化 git 子模块...")
        r = subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=ROOT)
        if r.returncode != 0:
            print("警告: git submodule 初始化失败，OCR 模型可能无法正确配置")

    # 检测平台
    os_type, os_arch, platform_tag = get_platform()
    print(f"平台: {os_type} / {os_arch} / {platform_tag}")
    print(f"版本: {args.tag}")
    print(f"输出: {args.output_dir}")
    print(f"MaaFramework: {MAA_FRAMEWORK_VERSION}")
    print(f"MFAAvalonia: {MFAA_VERSION}")

    if not args.skip_download:
        # 1. 嵌入式 Python
        python_exe = step_setup_python(os_type, os_arch)

        # 2. Python 依赖
        step_download_python_deps(python_exe)

        # 3. MaaFramework
        step_download_maa_framework(os_arch)

        # 4. MFAAvalonia
        if not args.skip_mfa:
            step_download_mfa(os_arch, platform_tag)

        # 5. 图标
        if not args.skip_icon:
            step_convert_icon()
    else:
        python_exe = PYTHON_DIR / ("python.exe" if os_type == "Windows" else "bin/python3")
        if not python_exe.exists():
            print(f"Python 未安装: {python_exe}，请先运行 build.py (不带 --skip-download)")
            sys.exit(1)

    # 6. 安装
    step_install(python_exe, args.tag, platform_tag)

    # 7. 整合 MFA
    if not args.skip_mfa:
        step_copy_mfa()

    # 打包
    if not args.skip_package:
        step_package(platform_tag, args.tag, args.output_dir)

    print("\n" + "=" * 60)
    print("构建完成!")
    print(f"安装目录: {INSTALL_DIR}")
    if not args.skip_package:
        print(f"打包文件: {Path(args.output_dir) / f'MaaNTE-{platform_tag}-{args.tag}'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
