"""
坐标标定工具 — 用于确定游戏界面中需要点击的位置。
依赖：pillow（requirements.txt 已包含），tkinter（Python 标准库）。

用法：
    python tools/calibrate_clicks.py              # 自动截取"异环"窗口
    python tools/calibrate_clicks.py <图片路径>    # 使用已有截图

操作：
    左键单击   — 记录坐标（画红点+编号）
    右键拖拽   — 框选矩形区域，按提示保存为模板 PNG
    C 键       — 清除所有标记
    ESC / 关窗 — 退出并打印所有坐标
"""

import sys
import ctypes
import ctypes.wintypes as wintypes
import tkinter as tk
from tkinter import simpledialog
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageGrab

WINDOW_TITLE_KEYWORD = "异环"
OUTPUT_DIR = Path(__file__).parent.parent / "assets/resource/base/image/auto_make_coffee"

clicks = []
drag_start = None
drag_end = None
base_image = None       # PIL Image, original
overlay_image = None    # PIL Image, with annotations
tk_image = None         # ImageTk.PhotoImage kept alive


def find_game_window_bbox():
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, ctypes.c_long)
    def enum_cb(hwnd, _):
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if WINDOW_TITLE_KEYWORD in title and ctypes.windll.user32.IsWindowVisible(hwnd):
            found.append((hwnd, title))
        return True

    ctypes.windll.user32.EnumWindows(enum_cb, 0)

    if not found:
        return None

    if len(found) > 1:
        print(f'找到多个包含"{WINDOW_TITLE_KEYWORD}"的窗口：')
        for h, t in found:
            print(f"  hwnd={h}  title={repr(t)}")
        print(f"使用第一个：{repr(found[0][1])}")

    hwnd, title = found[0]
    print(f"找到窗口：{repr(title)}")
    pt = wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    cr = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(cr))
    return (pt.x, pt.y, pt.x + cr.right, pt.y + cr.bottom)


def capture_game():
    bbox = find_game_window_bbox()
    if bbox is None:
        return None
    return ImageGrab.grab(bbox=bbox)


def redraw(canvas):
    global overlay_image, tk_image
    overlay_image = base_image.copy()
    draw = ImageDraw.Draw(overlay_image)

    for i, (x, y) in enumerate(clicks):
        r = 6
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
        draw.text((x + 9, y - 9), str(i + 1), fill=(0, 220, 0))

    if drag_start and drag_end:
        x1, y1 = drag_start
        x2, y2 = drag_end
        draw.rectangle((min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)),
                        outline=(255, 200, 0), width=2)

    tk_image = ImageTk.PhotoImage(overlay_image)
    canvas.itemconfig("img", image=tk_image)


def on_left_click(event, canvas):
    clicks.append((event.x, event.y))
    print(f"  click {len(clicks):2d}: ({event.x:4d}, {event.y:4d})")
    redraw(canvas)


def on_right_press(event):
    global drag_start, drag_end
    drag_start = (event.x, event.y)
    drag_end = None


def on_right_drag(event, canvas):
    global drag_end
    drag_end = (event.x, event.y)
    redraw(canvas)


def on_right_release(event, canvas, root):
    global drag_start, drag_end
    if drag_start is None:
        return
    x1, y1 = drag_start
    x2, y2 = event.x, event.y
    drag_start = None
    drag_end = None

    lx, rx = min(x1, x2), max(x1, x2)
    ty, by = min(y1, y2), max(y1, y2)
    if rx <= lx or by <= ty:
        redraw(canvas)
        return

    region_str = f"[{lx}, {ty}, {rx - lx}, {by - ty}]"
    print(f"\n框选区域: {region_str}")

    name = simpledialog.askstring(
        "保存模板",
        f"区域 {region_str}\n文件名（不含 .png）：",
        parent=root,
    )
    if name and name.strip():
        crop = base_image.crop((lx, ty, rx, by))
        out_path = OUTPUT_DIR / f"{name.strip()}.png"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        crop.save(str(out_path))
        print(f"  已保存 → {out_path}")
        print(f"  搜索区域: {region_str}")

    redraw(canvas)


def on_key(event, canvas, root):
    if event.keysym == "Escape":
        root.destroy()
    elif event.keysym.lower() == "c":
        clicks.clear()
        print("  已清除所有标记")
        redraw(canvas)


def print_results():
    print("\n" + "=" * 50)
    print("=== 记录的点击坐标 ===")
    print("CLICK_POSITIONS = [")
    for i, (x, y) in enumerate(clicks):
        print(f"    [{x}, {y}],  # click {i + 1}")
    print("]")
    print("=" * 50)


def main():
    global base_image

    if len(sys.argv) > 1:
        base_image = Image.open(sys.argv[1]).convert("RGB")
        print(f"已加载截图：{sys.argv[1]}")
    else:
        print(f'正在查找"{WINDOW_TITLE_KEYWORD}"窗口...')
        base_image = capture_game()
        if base_image is None:
            print(f"未找到包含\"{WINDOW_TITLE_KEYWORD}\"的窗口。")
            print("请先启动游戏，或将截图路径作为参数传入：")
            print("  python tools/calibrate_clicks.py <截图.png>")
            sys.exit(1)
        print(f"截图成功，分辨率：{base_image.width}×{base_image.height}")

    print("\n操作说明：")
    print("  左键单击   → 记录坐标")
    print("  右键拖拽   → 框选区域并保存为模板 PNG")
    print("  C 键       → 清除所有标记")
    print("  ESC / 关窗 → 退出并输出坐标\n")

    root = tk.Tk()
    root.title("Calibrate — ESC退出  C清除  左键记录  右键拖选模板")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=base_image.width, height=base_image.height,
                       cursor="crosshair", highlightthickness=0)
    canvas.pack()

    # 初始渲染
    global tk_image
    tk_image = ImageTk.PhotoImage(base_image)
    canvas.create_image(0, 0, anchor="nw", image=tk_image, tags="img")

    canvas.bind("<Button-1>",        lambda e: on_left_click(e, canvas))
    canvas.bind("<Button-3>",        on_right_press)
    canvas.bind("<B3-Motion>",       lambda e: on_right_drag(e, canvas))
    canvas.bind("<ButtonRelease-3>", lambda e: on_right_release(e, canvas, root))
    root.bind("<Key>",               lambda e: on_key(e, canvas, root))
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    root.mainloop()
    print_results()


if __name__ == "__main__":
    main()
