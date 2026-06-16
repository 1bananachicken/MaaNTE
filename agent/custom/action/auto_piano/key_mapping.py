"""异环钢琴键位映射

游戏钢琴物理布局（3 个八度 × 12 音阶 = 36 键）：
  高音：Q W E R T Y U  + Shift/Ctrl 半音
  中音：A S D F G H J  + Shift/Ctrl 半音
  低音：Z X C V B N M  + Shift/Ctrl 半音

半音规则：
  Shift + 白键 = 升半音 (#)
  Ctrl  + 白键 = 降半音 (b)
"""

# ==========================================
# 36键完整映射（含半音）
# ==========================================
NOTE_KEY_MAPPING = {
    # ---- 低音 C4 (MIDI 60) ~ B4 (71) ----
    60: "z",          # C4
    61: "shift+z",    # C#4
    62: "x",          # D4
    63: "ctrl+c",     # D#4 / Eb4
    64: "c",          # E4
    65: "v",          # F4
    66: "shift+v",    # F#4
    67: "b",          # G4
    68: "shift+b",    # G#4
    69: "n",          # A4
    70: "ctrl+m",     # A#4 / Bb4
    71: "m",          # B4

    # ---- 中音 C5 (72) ~ B5 (83) ----
    72: "a",          # C5
    73: "shift+a",    # C#5
    74: "s",          # D5
    75: "ctrl+d",     # D#5 / Eb5
    76: "d",          # E5
    77: "f",          # F5
    78: "shift+f",    # F#5
    79: "g",          # G5
    80: "shift+g",    # G#5
    81: "h",          # A5
    82: "ctrl+j",     # A#5 / Bb5
    83: "j",          # B5

    # ---- 高音 C6 (84) ~ B6 (95) ----
    84: "q",          # C6
    85: "shift+q",    # C#6
    86: "w",          # D6
    87: "ctrl+e",     # D#6 / Eb6
    88: "e",          # E6
    89: "r",          # F6
    90: "shift+r",    # F#6
    91: "t",          # G6
    92: "shift+t",    # G#6
    93: "y",          # A6
    94: "ctrl+u",     # A#6 / Bb6
    95: "u",          # B6
}


# ==========================================
# 21键白键映射（不含半音，简单模式）
# ==========================================
NOTE_KEY_MAPPING_WHITE = {
    # 低音 C4~B4
    60: "z", 62: "x", 64: "c", 65: "v", 67: "b", 69: "n", 71: "m",
    # 中音 C5~B5
    72: "a", 74: "s", 76: "d", 77: "f", 79: "g", 81: "h", 83: "j",
    # 高音 C6~B6
    84: "q", 86: "w", 88: "e", 89: "r", 91: "t", 93: "y", 95: "u",
}


# ==========================================
# 工具函数
# ==========================================

# 12音阶中的白键偏移量（C=0, D=2, E=4, F=5, G=7, A=9, B=11）
_WHITE_KEY_OFFSETS = frozenset({0, 2, 4, 5, 7, 9, 11})


def is_white_key(midi_pitch: int) -> bool:
    """判断 MIDI 音高是否为白键（自然音）"""
    return (midi_pitch % 12) in _WHITE_KEY_OFFSETS


def snap_to_white_key(midi_pitch: int) -> int:
    """
    将任意 MIDI 音高映射到最近的白键。
    若距离两侧白键相等，优先向低音方向取（向下取整）。
    """
    offset = midi_pitch % 12

    if offset in _WHITE_KEY_OFFSETS:
        return midi_pitch

    # 黑键：寻找最近的白键，优先向下（低音方向）
    for delta in range(1, 12):
        lower = (offset - delta) % 12
        if lower in _WHITE_KEY_OFFSETS:
            return midi_pitch - delta
        upper = (offset + delta) % 12
        if upper in _WHITE_KEY_OFFSETS:
            return midi_pitch + delta

    # 理论上不可能到达此处
    return midi_pitch


def get_mapping(key_mode: str = "36") -> dict[int, str]:
    """
    根据键位模式返回对应的映射表。

    Args:
        key_mode:
            - "36" : 完整 36 键（含半音，默认）
            - "21" : 仅白键 21 键（半音会被映射到最近白键）
    """
    if key_mode == "21":
        return NOTE_KEY_MAPPING_WHITE
    return NOTE_KEY_MAPPING
