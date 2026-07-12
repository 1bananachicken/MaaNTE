#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载Python依赖到deps目录的脚本
自动检测当前平台并下载对应架构的wheel文件
"""

import email
import os
import sys
import subprocess
import argparse
import platform
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# packaging 随 pip 一起分发（pip._vendor），保证运行本脚本的任何 Python 环境都可用
try:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
except ImportError:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

try:
    from pip._vendor.packaging.markers import default_environment as _default_marker_environment
except ImportError:
    try:
        from packaging.markers import default_environment as _default_marker_environment
    except ImportError:
        _default_marker_environment = None


def get_platform_tag():
    """自动检测当前平台并返回对应的平台标签"""
    os_type = platform.system()
    os_arch = platform.machine()

    print(f"检测到操作系统: {os_type}, 架构: {os_arch}")

    if os_type == "Windows":
        # 在Windows ARM64环境中，platform.machine()可能错误返回AMD64
        # 我们需要检查处理器标识符来确定真实架构
        processor_identifier = os.environ.get("PROCESSOR_IDENTIFIER", "")

        # 检查是否为ARM64处理器
        if "ARMv8" in processor_identifier or "ARM64" in processor_identifier:
            print(f"检测到ARM64处理器: {processor_identifier}")
            os_arch = "ARM64"

        # 映射platform.machine()到pip的平台标签
        arch_mapping = {
            "AMD64": "win_amd64",
            "x86_64": "win_amd64",
            "ARM64": "win_arm64",
            "aarch64": "win_arm64",
        }
        platform_tag = arch_mapping.get(os_arch, f"win_{os_arch.lower()}")

    elif os_type == "Darwin":  # macOS
        # 映射platform.machine()到pip的平台标签
        arch_mapping = {
            "x86_64": "macosx_10_9_x86_64",
            "arm64": "macosx_11_0_arm64",
            "aarch64": "macosx_11_0_arm64",
        }
        platform_tag = arch_mapping.get(os_arch, f"macosx_10_9_{os_arch}")

    elif os_type == "Linux":
        # 映射platform.machine()到pip的平台标签
        arch_mapping = {
            "x86_64": "linux_x86_64",
            "aarch64": "linux_aarch64",
            "arm64": "linux_aarch64",
        }
        platform_tag = arch_mapping.get(os_arch, f"linux_{os_arch}")

    else:
        raise ValueError(f"不支持的操作系统: {os_type}")

    print(f"使用平台标签: {platform_tag}")
    return platform_tag


def _target_marker_environment(platform_tag, python_version):
    """构造目标平台的 PEP 508 marker 评估环境。

    pip download --platform 只影响 wheel 标签选择，Requires-Dist 中的环境
    marker（如 loguru 的 colorama; sys_platform=='win32'）仍按宿主环境评估。
    交叉下载时这会静默漏掉目标平台的条件依赖，导致离线安装失败，
    必须用目标环境重新评估依赖闭包。
    """
    if python_version:
        parts = python_version.split(".")
        py_ver = ".".join(parts[:2])
        py_full = python_version if len(parts) >= 3 else f"{py_ver}.0"
    else:
        py_ver = ".".join(str(v) for v in sys.version_info[:2])
        py_full = platform.python_version()

    if platform_tag.startswith("win"):
        env = {
            "sys_platform": "win32",
            "platform_system": "Windows",
            "os_name": "nt",
            "platform_machine": "ARM64" if "arm64" in platform_tag else "AMD64",
        }
    elif platform_tag.startswith("macosx"):
        env = {
            "sys_platform": "darwin",
            "platform_system": "Darwin",
            "os_name": "posix",
            "platform_machine": "arm64" if "arm64" in platform_tag else "x86_64",
        }
    else:
        env = {
            "sys_platform": "linux",
            "platform_system": "Linux",
            "os_name": "posix",
            "platform_machine": "aarch64" if "aarch64" in platform_tag else "x86_64",
        }

    # 以 default_environment() 补齐 PEP 508 全部标准键（如 implementation_version），
    # 避免 marker 使用未覆盖的变量时评估出错；目标平台相关字段随后全部覆盖
    base_env = dict(_default_marker_environment()) if _default_marker_environment else {}
    base_env.update(env)
    base_env.update(
        {
            "python_version": py_ver,
            "python_full_version": py_full,
            "implementation_name": "cpython",
            "implementation_version": py_full,
            "platform_python_implementation": "CPython",
            "platform_release": "",
            "platform_version": "",
            # extra 置空：不评估 extras 引入的可选依赖
            "extra": "",
        }
    )
    return base_env


def _find_missing_requirements(deps_path, env):
    """扫描已下载 wheel 的 Requires-Dist，按目标环境评估 marker，
    返回缺失的依赖 {canonical_name: 不含 marker 的 requirement 字符串}。

    只做包名级校验：版本一致性由 pip download 的初始解析保证，
    缺失项均为 marker 评估差异导致的整包遗漏。
    """
    have = set()
    all_requires = []
    for whl in sorted(Path(deps_path).glob("*.whl")):
        have.add(canonicalize_name(whl.name.split("-")[0]))
        try:
            with zipfile.ZipFile(whl) as zf:
                meta_name = next(
                    n for n in zf.namelist() if n.endswith(".dist-info/METADATA")
                )
                meta = zf.read(meta_name).decode("utf-8", errors="replace")
        except (StopIteration, zipfile.BadZipFile, OSError) as e:
            print(f"警告: 无法读取 wheel 元数据 {whl.name}: {e}")
            continue
        # METADATA 为 RFC 822 风格头部，Requires-Dist 允许折叠续行（PEP 566），
        # 用 email 解析器正确合并续行，避免按行解析丢失折叠部分
        msg = email.message_from_string(meta)
        for req_line in msg.get_all("Requires-Dist") or []:
            all_requires.append(" ".join(str(req_line).split()))

    missing = {}
    for req_str in all_requires:
        try:
            req = Requirement(req_str)
        except Exception:
            continue
        if req.marker is not None and not req.marker.evaluate(environment=env):
            continue
        name = canonicalize_name(req.name)
        if name not in have and name not in missing:
            # 保留 extras（如 pkg[extra]）：补齐下载时让 pip 一并解析
            # extras 引入的附加依赖，否则闭包仍然不完整
            extras = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
            missing[name] = f"{req.name}{extras}{req.specifier}"
    return missing


def _complete_dependency_closure(deps_path, platform_tag, python_version, max_rounds=5):
    """迭代补齐目标平台的依赖闭包。新下载的 wheel 可能引入新的条件依赖，
    循环校验直至闭包完整；超过 max_rounds 视为失败。
    """
    env = _target_marker_environment(platform_tag, python_version)
    for round_index in range(max_rounds):
        missing = _find_missing_requirements(deps_path, env)
        if not missing:
            if round_index > 0:
                print("目标平台依赖闭包已补齐")
            return True

        print(f"闭包校验第 {round_index + 1} 轮，发现宿主 marker 评估漏掉的依赖: "
              f"{', '.join(sorted(missing))}")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            *sorted(missing.values()),
            "-d",
            str(deps_path),
            "--platform",
            platform_tag,
            "--only-binary=:all:",
        ]
        if python_version:
            cmd += ["--python-version", python_version]
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"补齐依赖闭包失败:\n{result.stdout}\n{result.stderr}")
            return False

    print(f"错误: 依赖闭包在 {max_rounds} 轮内未收敛")
    return False


def download_dependencies(deps_dir, platform_tag, python_version=None, allow_fallback=True):
    """下载依赖到指定目录

    python_version: 目标 Python 版本（如 "3.12"）。交叉构建时宿主 Python 版本
    与打包目标不同，必须显式传入以获取正确 ABI 的 wheel。
    allow_fallback: 平台特定下载失败时是否回退到不限平台的通用下载。
    交叉构建时必须禁用——回退会静默混入宿主平台的二进制 wheel。
    """
    # 创建deps目录
    deps_path = Path(deps_dir)
    deps_path.mkdir(parents=True, exist_ok=True)

    print(f"开始下载平台 {platform_tag} 的依赖到 {deps_dir}")

    # 从requirements.txt读取依赖
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("错误: requirements.txt 文件不存在")
        return False

    # 部分纯Python包在PyPI上只有sdist，需要先用pip wheel本地构建成wheel
    # 例如: proxy_tools (pywebview的依赖) 只有 .tar.gz
    # 构建前先确保 setuptools 可用（嵌入式Python默认不含 setuptools）
    SDIST_ONLY_PACKAGES = ["proxy_tools"]
    print("预构建纯Python sdist包为wheel...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "setuptools", "--quiet"],
        check=False,
        capture_output=True,
    )

    # 首先尝试下载平台特定的wheel文件
    try:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "-r",
            str(requirements_file),
            "-d",
            str(deps_path),
            "--platform",
            platform_tag,
            "--only-binary=:all:",
        ]
        if python_version:
            cmd += ["--python-version", python_version]

        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        if result.stderr:
            print("警告信息:")
            print(result.stderr)

        # 交叉下载模式：宿主评估 marker 会漏掉目标平台条件依赖，补齐闭包
        if not allow_fallback:
            if not _complete_dependency_closure(deps_path, platform_tag, python_version):
                return False

        # 列出下载的文件
        whl_files = list(deps_path.glob("*.whl"))
        print(f"\n下载的wheel文件 ({len(whl_files)} 个):")
        for whl_file in whl_files:
            print(f"  {whl_file.name}")

        print(f"依赖下载完成到: {deps_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"平台特定下载失败: {e}")
        if not allow_fallback:
            print("交叉构建模式：禁用通用回退下载（会混入宿主平台 wheel），直接失败")
            if e.stdout:
                print("stdout:", e.stdout)
            if e.stderr:
                print("stderr:", e.stderr)
            return False
        if e.stderr and (
            "Could not find a version" in e.stderr
            or "No matching distribution" in e.stderr
        ):
            print("某些包可能不支持当前平台，尝试通用下载策略...")

            # 回退到通用下载策略（不指定平台）
            try:
                cmd_fallback = [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "-r",
                    str(requirements_file),
                    "-d",
                    str(deps_path),
                    "--only-binary=:all:",
                    "--find-links",
                    str(deps_path),
                ]

                print(f"执行回退命令: {' '.join(cmd_fallback)}")
                result = subprocess.run(
                    cmd_fallback, check=True, capture_output=True, text=True
                )
                print(result.stdout)

                if result.stderr:
                    print("警告信息:")
                    print(result.stderr)

                # 列出下载的文件
                whl_files = list(deps_path.glob("*.whl"))
                print(f"\n下载的wheel文件 ({len(whl_files)} 个):")
                for whl_file in whl_files:
                    print(f"  {whl_file.name}")

                print(f"通用策略下载完成到: {deps_path}")
                return True

            except subprocess.CalledProcessError as e2:
                print(f"通用策略也失败: {e2}")
                if e2.stdout:
                    print("stdout:", e2.stdout)
                if e2.stderr:
                    print("stderr:", e2.stderr)
                return False
        else:
            if e.stdout:
                print("stdout:", e.stdout)
            if e.stderr:
                print("stderr:", e.stderr)
            return False


def main():
    parser = argparse.ArgumentParser(description="下载Python依赖到deps目录")
    parser.add_argument("--deps-dir", default="deps", help="依赖下载目录 (默认: deps)")
    parser.add_argument(
        "--platform-tag",
        default=None,
        help="pip 平台标签（如 win_amd64）。指定后跳过宿主平台自动探测，用于交叉构建",
    )
    parser.add_argument(
        "--python-version",
        default=None,
        help='目标 Python 版本（如 "3.12"）。交叉构建时必须与打包的 Python 一致',
    )

    args = parser.parse_args()

    try:
        if args.platform_tag:
            platform_tag = args.platform_tag
            print(f"使用指定平台标签: {platform_tag}")
        else:
            # 自动检测平台
            platform_tag = get_platform_tag()

        # 下载依赖
        success = download_dependencies(
            args.deps_dir,
            platform_tag,
            python_version=args.python_version,
            allow_fallback=args.platform_tag is None,
        )

        if success:
            print("✅ 依赖下载成功")
            sys.exit(0)
        else:
            print("❌ 依赖下载失败")
            sys.exit(1)

    except Exception as e:
        print(f"❌ 脚本执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
