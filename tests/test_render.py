# tests/test_render.py
import numpy as np
from render import render_board, render_side_by_side


def test_render_board_has_one_row_per_board_row_plus_border():
    board = np.zeros((20, 10))
    text = render_board(board)
    lines = text.split("\n")
    # top border + 20 board rows + bottom border
    assert len(lines) == 22


def test_render_board_marks_occupied_cells_differently_from_empty():
    board = np.zeros((20, 10))
    board[19, 0] = 1  # bottom-left cell occupied
    text = render_board(board)
    lines = text.split("\n")
    occupied_row = lines[1 + 19]  # skip top border
    empty_row = lines[1 + 0]
    assert occupied_row != empty_row


def test_render_side_by_side_has_both_labels():
    board = np.zeros((20, 10))
    text = render_side_by_side(board, board, label_a="REAL", label_b="DREAM")
    assert "REAL" in text
    assert "DREAM" in text
    # every real line should be paired with a dream line on the same row
    assert len(text.split("\n")) == len(render_board(board).split("\n"))
