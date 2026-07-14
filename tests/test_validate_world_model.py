import numpy as np
from env.tetris_env import SimplifiedTetrisEnv, ACTIONS
from models.rssm import RSSMEnsemble
from validate_world_model import rollout_real_and_dream, board_divergence, validate


def test_rollout_real_and_dream_shapes():
    env = SimplifiedTetrisEnv(seed=0)
    ensemble = RSSMEnsemble(n_models=3)
    actions = [ACTIONS.index("no_op")] * 5
    result = rollout_real_and_dream(env, ensemble, member_idx=0, action_sequence=actions)
    assert result["real_boards"].shape == (5, 20, 10)
    assert result["dream_boards"].shape == (5, 20, 10)


def test_board_divergence_zero_for_identical_boards():
    boards = np.random.randint(0, 2, size=(5, 20, 10)).astype(np.float32)
    div = board_divergence(boards, boards)
    assert np.allclose(div, 0.0)


def test_board_divergence_positive_for_different_boards():
    a = np.zeros((3, 20, 10), dtype=np.float32)
    b = np.ones((3, 20, 10), dtype=np.float32)
    div = board_divergence(a, b)
    assert (div > 0).all()


def test_validate_runs_end_to_end_on_untrained_model():
    # An untrained ensemble is expected to FAIL the gate — this test checks
    # the gate mechanism runs and produces a well-formed verdict, not that
    # an untrained model passes (it shouldn't).
    env = SimplifiedTetrisEnv(seed=0)
    ensemble = RSSMEnsemble(n_models=3)
    result = validate(env, ensemble, horizon=5, n_held_out=2, plot_path="plots/test_dream_vs_reality.png")
    assert "passed" in result and isinstance(result["passed"], bool)
    assert "mean_final_divergence" in result
    assert result["mean_final_divergence"] >= 0
