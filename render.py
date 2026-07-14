# render.py
import numpy as np

_CELL_FULL = "██"
_CELL_EMPTY = "  "


def render_board(board: np.ndarray, title: str = "") -> str:
    """Render a (H, W) 0/1 board array as a bordered ASCII grid."""
    width = board.shape[1]
    lines = []
    if title:
        lines.append(title.center(width * 2 + 2))
    lines.append("+" + "-" * (width * 2) + "+")
    for row in board:
        cells = "".join(_CELL_FULL if v > 0 else _CELL_EMPTY for v in row)
        lines.append("|" + cells + "|")
    lines.append("+" + "-" * (width * 2) + "+")
    return "\n".join(lines)


def render_side_by_side(board_a: np.ndarray, board_b: np.ndarray,
                         label_a: str = "REAL", label_b: str = "DREAM") -> str:
    """Render two same-shaped boards side by side, row-aligned, for visual comparison."""
    left_lines = render_board(board_a).split("\n")
    right_lines = render_board(board_b).split("\n")
    # Stamp each side's label into its top border so the label is visible
    # without adding an extra line (keeps row count == render_board's).
    left_lines[0] = _stamp_label(left_lines[0], label_a)
    right_lines[0] = _stamp_label(right_lines[0], label_b)
    left_width = max(len(l) for l in left_lines)
    rows = [l.ljust(left_width) + "    " + r for l, r in zip(left_lines, right_lines)]
    return "\n".join(rows)


def _stamp_label(border_line: str, label: str) -> str:
    """Center `label` inside a border line like '+----+', preserving its length."""
    if not label:
        return border_line
    inner_width = len(border_line) - 2
    stamped = label.center(inner_width)[:inner_width]
    return "+" + stamped + "+"
