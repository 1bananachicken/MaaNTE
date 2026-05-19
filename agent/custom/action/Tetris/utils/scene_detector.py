from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maa.context import Context


class TetrisSceneDetector:
    """Uses MaaFramework pipeline nodes for Tetris scene recognition.

    Scene nodes are defined in TetrisScene.json. This class calls them
    via context.run_recognition() for use in custom actions that need
    scene information (e.g. play_round game loop).
    """

    _NODE_DROP = "TetrisSceneDrop"
    _NODE_MATCHEND = "TetrisSceneMatchend"

    def check_drop(self, context: Context, img) -> dict | None:
        return self._check(context, self._NODE_DROP, img)

    def check_matchend(self, context: Context, img) -> dict | None:
        return self._check(context, self._NODE_MATCHEND, img)

    @staticmethod
    def _check(context: Context, node_name: str, img) -> dict | None:
        try:
            detail = context.run_recognition(node_name, img)
            if detail is None or not detail.hit or detail.box is None:
                return None
            x, y, w, h = detail.box
            return {"x": x, "y": y, "w": w, "h": h, "cx": x + w // 2, "cy": y + h // 2}
        except Exception:
            return None
