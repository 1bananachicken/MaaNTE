"""
config/resource/ 透明覆盖 assets/resource/base/

规则:
  config/resource/image/Fish/slider.png  覆盖  assets/resource/base/image/Fish/slider.png
  ...所有子目录结构完全一致...

用法:
  from utils.resource_overlay import res
  res.resolve("image", "Fish", "slider.png")  # 读: 用户覆盖优先
  res.user_path("image", "BondGift", "x.png") # 写: 统一落 config/resource/
"""

from pathlib import Path
from utils.singleton import Singleton


class ResourceOverlay(Singleton):
    def __init__(self):
        # main.py 启动时已 os.chdir(project_root_dir), CWD = 项目根
        root = Path(".")
        self._builtin = root / "assets" / "resource" / "base"
        self._user = root / "config" / "resource"

    def resolve(self, *segments: str) -> Path:
        """查找文件: 用户覆盖优先, 没有则用内置"""
        rel = Path(*segments)
        user = self._user / rel
        return user if user.exists() else self._builtin / rel

    def user_path(self, *segments: str) -> Path:
        """获取用户目录下路径(用于写入), 自动创建父目录"""
        p = self._user / Path(*segments)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


res = ResourceOverlay()
