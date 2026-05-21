import argparse
import json
from pathlib import Path

import cv2


DEFAULT_LABELS = [
    "food_1_ready_2",
    "food_1_ready_3",
    "food_2_ready_2",
    "food_2_ready_3",
    "food_3_ready_2",
    "food_3_ready_3",
    "timer",
    "exit",
    "enter",
    "level_chose",
]


class RoiMarker:
    def __init__(self, image_path: Path, labels: list[str], scale: float):
        self.image_path = image_path
        self.labels = labels
        self.scale = scale
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        self.image = image

        self.display_base = cv2.resize(
            self.image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        self.window_name = "ROI Marker"
        self.rois: list[dict[str, object]] = []
        self.current_label_index = 0
        self.dragging = False
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:
            cv2.imshow(self.window_name, self._render())
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                self._save()
            elif key in (8, 127, ord("u")):
                self._undo()
            elif key == ord("c"):
                self.rois.clear()
                print("已清空全部 ROI")
            elif key == ord("n"):
                self._next_label()
            elif ord("1") <= key <= ord("9"):
                self.current_label_index = min(key - ord("1"), len(self.labels) - 1)
                print(f"当前标签: {self.current_label}")

        cv2.destroyAllWindows()

    @property
    def current_label(self) -> str:
        return self.labels[self.current_label_index]

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.drag_end = (x, y)
            roi = self._scaled_rect_to_original(self.drag_start, self.drag_end)
            if roi[2] <= 0 or roi[3] <= 0:
                print("忽略空 ROI")
                return
            item = {"label": self.current_label, "roi": roi}
            self.rois.append(item)
            print(f"{self.current_label}: {roi}")
            self._next_label()

    def _scaled_rect_to_original(
        self, start: tuple[int, int] | None, end: tuple[int, int] | None
    ) -> list[int]:
        if start is None or end is None:
            return [0, 0, 0, 0]

        x1, y1 = start
        x2, y2 = end
        left = int(round(min(x1, x2) / self.scale))
        top = int(round(min(y1, y2) / self.scale))
        right = int(round(max(x1, x2) / self.scale))
        bottom = int(round(max(y1, y2) / self.scale))

        height, width = self.image.shape[:2]
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(0, min(right, width))
        bottom = max(0, min(bottom, height))
        return [left, top, right - left, bottom - top]

    def _render(self):
        canvas = self.display_base.copy()

        for item in self.rois:
            label = str(item["label"])
            x, y, w, h = self._original_rect_to_scaled(item["roi"])
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                canvas,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if self.dragging and self.drag_start and self.drag_end:
            cv2.rectangle(canvas, self.drag_start, self.drag_end, (0, 200, 255), 2)

        help_lines = [
            f"Current: {self.current_label}",
            "Drag: add ROI | 1-9: choose label | n: next label",
            "u/backspace: undo | c: clear | s: save | q/esc: quit",
        ]
        for i, line in enumerate(help_lines):
            y = 24 + i * 24
            cv2.putText(
                canvas,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def _original_rect_to_scaled(self, roi) -> tuple[int, int, int, int]:
        x, y, w, h = [int(value) for value in roi]
        return (
            int(round(x * self.scale)),
            int(round(y * self.scale)),
            int(round(w * self.scale)),
            int(round(h * self.scale)),
        )

    def _next_label(self):
        self.current_label_index = (self.current_label_index + 1) % len(self.labels)
        print(f"当前标签: {self.current_label}")

    def _undo(self):
        if not self.rois:
            print("没有可撤销的 ROI")
            return
        removed = self.rois.pop()
        print(f"撤销: {removed['label']} {removed['roi']}")

    def _save(self):
        output_json = self.image_path.with_name(f"{self.image_path.stem}_roi.json")
        output_image = self.image_path.with_name(f"{self.image_path.stem}_roi.png")

        data = [
            {"label": str(item["label"]), "roi": item["roi"]}
            for item in self.rois
        ]
        output_json.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n")
        cv2.imwrite(str(output_image), self._render())

        print(f"已保存 JSON: {output_json}")
        print(f"已保存标注图: {output_image}")
        print(json.dumps(data, ensure_ascii=False, indent=4))


def parse_args():
    parser = argparse.ArgumentParser(
        description="在截图上拖拽标注 MaaFW ROI。请传入要标注的截图路径，例如：samples/main.jpg。"
    )
    parser.add_argument(
        "image",
        help="要标注的截图路径（必填），例如：samples/main.jpg。",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="显示缩放比例；只影响窗口显示，不影响输出 ROI 坐标。",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help="按顺序标注的标签名。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"错误：图片文件不存在或不可读取：{image_path}")

    marker = RoiMarker(image_path, args.labels, args.scale)
    print("操作说明：拖拽鼠标左键画框，松开后输出 [x, y, w, h]。")
    print("按 s 保存 JSON/标注图，按 q 或 Esc 退出。")
    print(f"当前标签: {marker.current_label}")
    marker.run()


if __name__ == "__main__":
    main()
