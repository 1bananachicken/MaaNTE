import os
import sys
import platform
import shutil
import subprocess
import urllib.request
import zipfile
import tarfile
import stat  # 用于在 macOS/Linux 上设置文件权限

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
print(os.getcwd())
# --- 配置 ---
# 可以根据需要修改这些值
PYTHON_VERSION_TARGET = "3.12.10"  # 目标 Python 版本
# python-build-standalone 的发布标签，需要与 PYTHON_VERSION_TARGET 兼容
# 前往 https://github.com/indygreg/python-build-standalone/releases 查看最新标签和可用版本
PYTHON_BUILD_STANDALONE_RELEASE_TAG = "20250409"

DEST_DIR = os.path.join("install", "python")  # Python 安装的目标目录

# --- 辅助函数 ---


def download_file(url, dest_path):
    """下载文件到指定路径"""
    print(f"正在下载: {url}")
    print(f"到: {dest_path}")
    # 确保目标目录存在
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        print("下载完成。")
    except urllib.error.HTTPError as e:
        print(f"HTTP 错误 {e.code}: {e.reason} (URL: {url})")
        raise
    except urllib.error.URLError as e:
        print(f"URL 错误: {e.reason} (URL: {url})")
        raise
    except Exception as e:
        print(f"下载过程中发生意外错误: {e}")
        raise


def extract_zip(zip_path, dest_dir):
    """解压 ZIP 文件"""
    print(f"正在解压 ZIP: {zip_path} 到 {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    print("ZIP 解压完成。")


def extract_tar(tar_path, dest_dir):
    """解压 TAR (tar.gz, tar.xz, tar.bz2) 文件"""
    print(f"正在解压 TAR: {tar_path} 到 {dest_dir}")
    try:
        # 'r:*' 会自动检测压缩格式
        with tarfile.open(tar_path, "r:*") as tar_ref:
            tar_ref.extractall(path=dest_dir)
        print("TAR 解压完成。")
    except tarfile.ReadError as e:
        print(f"Tarfile 读取错误: {e}。文件可能已损坏或不是有效的 TAR 归档。")
        raise
    except Exception as e:
        print(f"TAR 解压过程中发生意外错误: {e}")
        raise


def get_python_executable_path(base_dir, os_type):
    """获取已安装 Python 环境中的可执行文件路径"""
    if os_type == "Windows":
        return os.path.join(base_dir, "python.exe")
    elif os_type == "Darwin":  # macOS
        # python-build-standalone 通常包含 python 和 python3
        # 我们优先使用 python3 (通常 python 是指向 python3 的符号链接)
        py3_path = os.path.join(base_dir, "bin", "python3")
        py_path = os.path.join(base_dir, "bin", "python")
        if os.path.exists(py3_path):
            return py3_path
        elif os.path.exists(py_path):  # 作为备选
            return py_path
        else:
            return None  # 未找到
    return None


def ensure_pip(python_executable, python_install_dir):
    """安装 pip"""
    if not python_executable or not os.path.exists(python_executable):
        print("错误: Python 可执行文件未找到，无法安装 pip。")
        return False

    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    # 将 get-pip.py 下载到 Python 安装目录下，执行后再删除
    get_pip_script_path = os.path.join(python_install_dir, "get-pip.py")

    print(f"正在下载 get-pip.py 从 {get_pip_url}")
    try:
        download_file(get_pip_url, get_pip_script_path)
    except Exception as e:
        print(f"下载 get-pip.py 失败: {e}")
        return False

    print("正在使用 get-pip.py 安装 pip...")
    try:
        # 在 Python 安装目录下执行 get-pip.py
        subprocess.run([python_executable, get_pip_script_path], check=True)
        print("pip 安装成功。")
        return True
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"pip 安装失败: {e}")
        return False
    finally:
        if os.path.exists(get_pip_script_path):
            os.remove(get_pip_script_path)  # 清理下载的脚本


# --- 主逻辑 ---
def main():
    os_type = platform.system()
    os_arch = platform.machine()

    print(f"操作系统: {os_type}, 架构: {os_arch}")
    print(f"目标 Python 版本: {PYTHON_VERSION_TARGET}")
    print(f"目标安装目录: {DEST_DIR}")

    # 检查 Python 是否已经存在
    python_exe_check = get_python_executable_path(DEST_DIR, os_type)
    if python_exe_check and os.path.exists(python_exe_check):
        print(f"Python 似乎已存在于 {DEST_DIR} (找到: {python_exe_check})。")
        if ensure_pip(python_exe_check, DEST_DIR):
            print("Python 和 pip 已配置。跳过安装。")
        else:
            print("Python 存在但 pip 配置失败。请检查。")
        return

    if os.path.exists(DEST_DIR):
        print(f"目标目录 {DEST_DIR} 已存在但 Python 未完全配置，将尝试清理并重新安装。")
        try:
            shutil.rmtree(DEST_DIR)
        except OSError as e:
            print(f"清理目录 {DEST_DIR} 失败: {e}。请手动删除后重试。")
            return

    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"已创建目录: {DEST_DIR}")

    python_executable_final_path = None

    if os_type == "Windows":
        # 在Windows ARM64环境中，platform.machine()可能错误返回AMD64
        # 我们需要检查处理器标识符来确定真实架构
        processor_identifier = os.environ.get("PROCESSOR_IDENTIFIER", "")

        # 检查是否为ARM64处理器
        if "ARMv8" in processor_identifier or "ARM64" in processor_identifier:
            print(f"检测到ARM64处理器: {processor_identifier}")
            os_arch = "ARM64"

        # 映射platform.machine()到Python官网的命名
        arch_mapping = {
            "AMD64": "amd64",
            "x86_64": "amd64",
            "ARM64": "arm64",
            "aarch64": "arm64",
        }
        win_arch_suffix = arch_mapping.get(os_arch, os_arch.lower())

        if win_arch_suffix not in ["amd64", "arm64"]:
            print(f"错误: 不支持的Windows架构: {os_arch} -> {win_arch_suffix}")
            return

        print(f"使用Windows架构: {os_arch} -> {win_arch_suffix}")

        download_url = f"https://www.python.org/ftp/python/{PYTHON_VERSION_TARGET}/python-{PYTHON_VERSION_TARGET}-embed-{win_arch_suffix}.zip"
        zip_filename = f"python-{PYTHON_VERSION_TARGET}-embed-{win_arch_suffix}.zip"
        zip_filepath = os.path.join(DEST_DIR, zip_filename)  # 下载到目标目录内再解压

        try:
            download_file(download_url, zip_filepath)
            extract_zip(zip_filepath, DEST_DIR)
        except Exception as e:
            print(f"Windows Python 下载或解压失败: {e}")
            return
        finally:
            if os.path.exists(zip_filepath):
                os.remove(zip_filepath)

        # 修改 ._pth 文件
        # pth 文件名格式如: python312._pth for Python 3.12.x
        version_nodots = PYTHON_VERSION_TARGET.replace(".", "")[:3]
        pth_filename_pattern = f"python{version_nodots}._pth"

        pth_file_path = os.path.join(DEST_DIR, pth_filename_pattern)
        if not os.path.exists(pth_file_path):
            # 有时 embeddable zip 中 pth 文件的命名可能不带 minor version，如 python3._pth
            # 尝试查找所有 python*._pth 文件
            found_pth_files = [
                f
                for f in os.listdir(DEST_DIR)
                if f.startswith("python") and f.endswith("._pth")
            ]
            if found_pth_files:
                pth_file_path = os.path.join(DEST_DIR, found_pth_files[0])
            else:
                print(f"错误: 未在 {DEST_DIR} 中找到 ._pth 文件。")
                return

        print(f"正在修改 ._pth 文件: {pth_file_path}")
        try:
            with open(pth_file_path, "r+", encoding="utf-8") as f:
                content = f.read()
                # 取消注释 import site
                content = content.replace("#import site", "import site")
                content = content.replace(
                    "# import site", "import site"
                )  # 处理可能的空格

                # 添加必要的相对路径 (相对于 DEST_DIR)
                required_paths = [".", "Lib", "Lib\\site-packages", "DLLs"]
                for p_path in required_paths:
                    if p_path not in content.splitlines():  # 避免重复添加
                        content += f"\n{p_path}"
                f.seek(0)
                f.write(content)
                f.truncate()
            print("._pth 文件修改完成。")
        except Exception as e:
            print(f"修改 ._pth 文件失败: {e}")
            return
        python_executable_final_path = get_python_executable_path(DEST_DIR, os_type)

    elif os_type == "Darwin":  # macOS
        # 映射platform.machine()到python-build-standalone的架构名称
        arch_mapping = {"x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}
        pbs_arch = arch_mapping.get(os_arch, os_arch)

        if pbs_arch not in ["x86_64", "aarch64"]:
            print(f"错误: 不支持的 macOS 架构: {os_arch} -> {pbs_arch}")
            return

        # 文件名格式: cpython-{PYTHON_VERSION}+{RELEASE_TAG_DATE}-{ARCH}-apple-darwin-install_only.tar.gz
        pbs_filename = f"cpython-{PYTHON_VERSION_TARGET}+{PYTHON_BUILD_STANDALONE_RELEASE_TAG}-{pbs_arch}-apple-darwin-install_only.tar.gz"
        download_url = f"https://github.com/indygreg/python-build-standalone/releases/download/{PYTHON_BUILD_STANDALONE_RELEASE_TAG}/{pbs_filename}"
        tar_filename = pbs_filename  # 使用原始文件名
        tar_filepath = os.path.join(DEST_DIR, tar_filename)  # 下载到目标目录内

        try:
            download_file(download_url, tar_filepath)
            # python-build-standalone 的包解压后通常包含一个名为 'python' 的顶层目录
            # 我们需要将这个 'python' 目录的内容移动到 DEST_DIR
            temp_extract_dir = os.path.join(DEST_DIR, "_temp_extract")
            os.makedirs(temp_extract_dir, exist_ok=True)
            extract_tar(tar_filepath, temp_extract_dir)

            extracted_python_root = os.path.join(temp_extract_dir, "python")
            if os.path.isdir(extracted_python_root):
                print(f"正在移动 {extracted_python_root} 的内容到 {DEST_DIR}")
                for item_name in os.listdir(extracted_python_root):
                    s = os.path.join(extracted_python_root, item_name)
                    d = os.path.join(DEST_DIR, item_name)
                    shutil.move(s, d)
                shutil.rmtree(temp_extract_dir)  # 清理临时解压目录
            else:
                print(f"错误: 解压后未找到预期的 'python' 子目录于 {temp_extract_dir}")
                shutil.rmtree(temp_extract_dir)
                return
        except Exception as e:
            print(f"macOS Python 下载或解压失败: {e}")
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)
            return
        finally:
            if os.path.exists(tar_filepath):
                os.remove(tar_filepath)

        # 为 bin 目录下的可执行文件设置执行权限
        bin_dir = os.path.join(DEST_DIR, "bin")
        if os.path.isdir(bin_dir):
            print(f"正在为 {bin_dir} 中的文件设置执行权限...")
            for item_name in os.listdir(bin_dir):
                item_path = os.path.join(bin_dir, item_name)
                if os.path.isfile(item_path) and not os.access(item_path, os.X_OK):
                    try:
                        current_mode = os.stat(item_path).st_mode
                        os.chmod(
                            item_path,
                            current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
                        )
                        print(f"  已为 {item_name} 设置执行权限。")
                    except Exception as e:
                        print(f"  为 {item_name} 设置执行权限失败: {e}")
        python_executable_final_path = get_python_executable_path(DEST_DIR, os_type)
    else:
        print(f"错误: 不支持的操作系统: {os_type}")
        return

    if not python_executable_final_path or not os.path.exists(
        python_executable_final_path
    ):
        print("错误: Python 可执行文件在安装后未找到。")
        return

    print(f"Python 环境已初步设置在: {DEST_DIR}")
    print(f"Python 可执行文件: {python_executable_final_path}")

    # 安装 pip
    if ensure_pip(python_executable_final_path, DEST_DIR):
        print("嵌入式 Python 环境安装和 pip 配置完成。")
    else:
        print("嵌入式 Python 环境安装完成，但 pip 配置失败。")


if __name__ == "__main__":
    main()
