import numpy as np
import torch


class ReplayBuffer:
    """Stores complete episodes, samples fixed-length sequence chunks for
    RSSM training (the GRU needs temporal context, so we can't sample
    single transitions)."""

    def __init__(self, obs_dim: int, num_actions: int):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self._episodes = []  # list of dicts: obs, action, reward, done (all np arrays)

    def add_episode(self, obs_seq, action_seq, reward_seq, done_seq):
        # obs_seq has length T+1 (includes the final next_obs); the rest have length T.
        length = len(action_seq)
        assert len(obs_seq) == length + 1
        assert len(reward_seq) == length and len(done_seq) == length
        action_array = np.array(action_seq, dtype=np.int64)
        assert np.all((action_array >= 0) & (action_array < self.num_actions)), \
            f"All action indices must be in [0, {self.num_actions}), got range [{action_array.min()}, {action_array.max()}]"
        self._episodes.append({
            "obs": np.stack(obs_seq).astype(np.float32),
            "action": action_array,
            "reward": np.array(reward_seq, dtype=np.float32),
            "done": np.array(done_seq, dtype=bool),
            "length": length,
        })

    def __len__(self):
        return sum(ep["length"] for ep in self._episodes)

    def sample_sequences(self, batch_size: int, seq_len: int):
        eligible = [ep for ep in self._episodes if ep["length"] >= seq_len]
        if not eligible:
            raise ValueError(
                f"no episode has at least seq_len={seq_len} transitions "
                f"(longest available: {max((ep['length'] for ep in self._episodes), default=0)})"
            )

        obs_batch, next_obs_batch, action_batch, reward_batch, done_batch = [], [], [], [], []
        for _ in range(batch_size):
            ep = eligible[np.random.randint(len(eligible))]
            start = np.random.randint(0, ep["length"] - seq_len + 1)
            end = start + seq_len

            obs_batch.append(ep["obs"][start:end])
            next_obs_batch.append(ep["obs"][start + 1:end + 1])
            action_onehot = np.zeros((seq_len, self.num_actions), dtype=np.float32)
            action_onehot[np.arange(seq_len), ep["action"][start:end]] = 1.0
            action_batch.append(action_onehot)
            reward_batch.append(ep["reward"][start:end])
            done_batch.append(ep["done"][start:end].astype(np.float32))

        return {
            "obs": torch.from_numpy(np.stack(obs_batch)),
            "next_obs": torch.from_numpy(np.stack(next_obs_batch)),
            "action": torch.from_numpy(np.stack(action_batch)),
            "reward": torch.from_numpy(np.stack(reward_batch)),
            "done": torch.from_numpy(np.stack(done_batch)),
        }
