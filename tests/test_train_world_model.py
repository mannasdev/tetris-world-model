# tests/test_train_world_model.py
import numpy as np
from replay_buffer import ReplayBuffer
from models.rssm import RSSMEnsemble
from train_world_model import build_training_batch, train_world_model

OBS_DIM = 407


def _fill_buffer():
    buf = ReplayBuffer(OBS_DIM, num_actions=7)
    for _ in range(5):
        length = 20
        obs_seq = [np.random.rand(OBS_DIM).astype(np.float32) for _ in range(length + 1)]
        # zero out piece one-hot slice, then set a single valid class, so
        # next_piece derivation has a well-defined argmax
        for o in obs_seq:
            o[400:407] = 0.0
            o[400 + np.random.randint(7)] = 1.0
        action_seq = list(np.random.randint(0, 7, size=length))
        reward_seq = list(np.random.rand(length))
        done_seq = [False] * (length - 1) + [True]
        buf.add_episode(obs_seq, action_seq, reward_seq, done_seq)
    return buf


def test_build_training_batch_shapes():
    buf = _fill_buffer()
    raw = buf.sample_sequences(batch_size=4, seq_len=8)
    batch = build_training_batch(raw)
    assert batch["next_board"].shape == (4, 8, 200)
    assert batch["next_piece"].shape == (4, 8)


def test_train_world_model_runs_and_loss_is_finite():
    buf = _fill_buffer()
    ensemble = RSSMEnsemble(n_models=3)
    losses = train_world_model(ensemble, buf, steps=5, batch_size=4, seq_len=8)
    assert len(losses) == 5
    assert all(l == l for l in losses)  # no NaNs


def test_train_world_model_appends_to_log_path(tmp_path):
    import json
    log_path = tmp_path / "world_model_losses.json"
    buf = _fill_buffer()
    ensemble = RSSMEnsemble(n_models=3)
    train_world_model(ensemble, buf, steps=3, batch_size=4, seq_len=8, log_path=str(log_path))
    train_world_model(ensemble, buf, steps=2, batch_size=4, seq_len=8, log_path=str(log_path))
    with open(log_path) as f:
        logged = json.load(f)
    assert len(logged) == 5  # appended across two calls, not overwritten
