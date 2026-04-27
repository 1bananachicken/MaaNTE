#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载Python依赖到deps目录的脚本
自动检测当前平台并下载对应架构的wheel文件
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


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


def download_dependencies(deps_dir, platform_tag):
    """下载依赖到指定目录"""
    # 创建deps目录
    deps_path = Path(deps_dir)
    deps_path.mkdir(parents=True, exist_ok=True)

    print(f"开始下载平台 {platform_tag} 的依赖到 {deps_dir}")

    # 从requirements.txt读取依赖
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("错误: requirements.txt 文件不存在")
        return False

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

        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        if result.stderr:
            print("警告信息:")
            print(result.stderr)

        # 列出下载的文件
        whl_files = list(deps_path.glob("*.whl"))
        print(f"\n下载的wheel文件 ({len(whl_files)} 个):")
        for whl_file in whl_files:
            print(f"  {whl_file.name}")

        print(f"依赖下载完成到: {deps_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"平台特定下载失败: {e}")
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

    args = parser.parse_args()

    try:
        # 自动检测平台
        platform_tag = get_platform_tag()

        # 下载依赖
        success = download_dependencies(args.deps_dir, platform_tag)

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