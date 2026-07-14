import numpy as np
from env.tetris_env import SimplifiedTetrisEnv, ACTIONS, OBS_DIM


def test_action_space_excludes_hold():
    assert len(ACTIONS) == 7
    assert "hold" not in ACTIONS and "swap" not in ACTIONS


def test_reset_returns_correct_obs_shape():
    env = SimplifiedTetrisEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_obs_has_no_queue_or_holder_leakage():
    # The observation vector must be exactly board (200) + active mask (200) + piece one-hot (7).
    # If a future edit accidentally concatenates the native "queue" or "holder" arrays in,
    # this shape check catches it immediately.
    assert OBS_DIM == 200 + 200 + 7


def test_step_runs_and_reports_lines_cleared():
    env = SimplifiedTetrisEnv(seed=0)
    env.reset()
    obs, reward, done, info = env.step(ACTIONS.index("hard_drop"))
    assert obs.shape == (OBS_DIM,)
    assert "lines_cleared" in info
    assert "piece_type" in info
    assert 0 <= info["piece_type"] <= 6


def test_top_out_gives_negative_reward():
    # Hard-drop repeatedly until the board tops out; the terminal reward must be
    # negative (design doc: "penalty on topping out" — the library's own default
    # is 0, so this also guards the RewardsMapping override).
    env = SimplifiedTetrisEnv(seed=0)
    env.reset()
    done = False
    last_reward = None
    for _ in range(500):
        _, last_reward, done, _ = env.step(ACTIONS.index("hard_drop"))
        if done:
            break
    assert done, "expected the board to top out within 500 hard drops"
    assert last_reward < 0


def test_board_from_obs_shape():
    env = SimplifiedTetrisEnv(seed=0)
    obs = env.reset()
    board = SimplifiedTetrisEnv.board_from_obs(obs)
    assert board.shape == (20, 10)
