import numpy as np
from env.tetris_env import SimplifiedTetrisEnv, ACTIONS, OBS_DIM, PER_TICK_SURVIVAL_REWARD


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


def test_terminal_reward_is_not_diluted_by_survival_bonus():
    # The per-tick survival bonus must only apply on non-terminal steps --
    # otherwise the game_over penalty gets partially cancelled out, making
    # topping out look less bad than it should.
    env = SimplifiedTetrisEnv(seed=0)
    env.reset()
    done = False
    last_reward = None
    for _ in range(500):
        _, last_reward, done, _ = env.step(ACTIONS.index("hard_drop"))
        if done:
            break
    assert done
    assert last_reward == -10.0


def test_non_terminal_no_op_gets_per_tick_survival_reward():
    # Tetris-Gymnasium's own reward is 0 for any non-committing action
    # (verified against the installed library source) -- the wrapper must
    # add PER_TICK_SURVIVAL_REWARD on top, on every non-terminal step,
    # regardless of action. This is the fix for the reward-timing bug where
    # only piece-commits (hard_drop) ever earned reward, making "commit as
    # fast as possible" strictly optimal under discounting.
    env = SimplifiedTetrisEnv(seed=0)
    env.reset()
    obs, reward, done, info = env.step(ACTIONS.index("no_op"))
    assert not done
    assert reward == PER_TICK_SURVIVAL_REWARD


def test_board_from_obs_shape():
    env = SimplifiedTetrisEnv(seed=0)
    obs = env.reset()
    board = SimplifiedTetrisEnv.board_from_obs(obs)
    assert board.shape == (20, 10)
