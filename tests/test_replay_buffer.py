import numpy as np
import torch
from replay_buffer import ReplayBuffer

OBS_DIM = 407
NUM_ACTIONS = 7


def _fake_episode(length):
    obs_seq = [np.random.rand(OBS_DIM).astype(np.float32) for _ in range(length + 1)]
    action_seq = [np.random.randint(0, NUM_ACTIONS) for _ in range(length)]
    reward_seq = [float(np.random.rand()) for _ in range(length)]
    done_seq = [False] * (length - 1) + [True]
    return obs_seq, action_seq, reward_seq, done_seq


def test_len_counts_transitions():
    buf = ReplayBuffer(OBS_DIM, NUM_ACTIONS)
    obs_seq, action_seq, reward_seq, done_seq = _fake_episode(10)
    buf.add_episode(obs_seq, action_seq, reward_seq, done_seq)
    assert len(buf) == 10


def test_sample_sequences_shapes():
    buf = ReplayBuffer(OBS_DIM, NUM_ACTIONS)
    for _ in range(3):
        obs_seq, action_seq, reward_seq, done_seq = _fake_episode(20)
        buf.add_episode(obs_seq, action_seq, reward_seq, done_seq)

    batch = buf.sample_sequences(batch_size=4, seq_len=8)
    assert batch["obs"].shape == (4, 8, OBS_DIM)
    assert batch["next_obs"].shape == (4, 8, OBS_DIM)
    assert batch["action"].shape == (4, 8, NUM_ACTIONS)
    assert batch["reward"].shape == (4, 8)
    assert batch["done"].shape == (4, 8)
    assert isinstance(batch["obs"], torch.Tensor)


def test_sample_raises_if_no_episode_long_enough():
    buf = ReplayBuffer(OBS_DIM, NUM_ACTIONS)
    obs_seq, action_seq, reward_seq, done_seq = _fake_episode(3)
    buf.add_episode(obs_seq, action_seq, reward_seq, done_seq)
    try:
        buf.sample_sequences(batch_size=2, seq_len=8)
        assert False, "expected a ValueError"
    except ValueError:
        pass


def test_add_episode_rejects_out_of_range_actions():
    buf = ReplayBuffer(OBS_DIM, NUM_ACTIONS)

    # Test with action index equal to NUM_ACTIONS (out of range)
    obs_seq, action_seq, reward_seq, done_seq = _fake_episode(5)
    action_seq[2] = NUM_ACTIONS
    try:
        buf.add_episode(obs_seq, action_seq, reward_seq, done_seq)
        assert False, "expected an AssertionError for action >= num_actions"
    except AssertionError as e:
        assert "action indices must be in" in str(e).lower()

    # Test with negative action index
    obs_seq_neg, action_seq_neg, reward_seq_neg, done_seq_neg = _fake_episode(5)
    action_seq_neg[1] = -1
    try:
        buf.add_episode(obs_seq_neg, action_seq_neg, reward_seq_neg, done_seq_neg)
        assert False, "expected an AssertionError for negative action"
    except AssertionError as e:
        assert "action indices must be in" in str(e).lower()
