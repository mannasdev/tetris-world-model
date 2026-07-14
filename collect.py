# collect.py
import numpy as np
from env.tetris_env import SimplifiedTetrisEnv, ACTIONS
from replay_buffer import ReplayBuffer


def collect_episodes(env: SimplifiedTetrisEnv, buffer: ReplayBuffer, policy_fn, n_episodes: int) -> dict:
    total_steps = 0
    total_lines_cleared = 0

    for _ in range(n_episodes):
        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        obs = env.reset()
        obs_seq.append(obs)
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, done, info = env.step(action)
            obs_seq.append(obs)
            action_seq.append(action)
            reward_seq.append(reward)
            done_seq.append(done)
            total_steps += 1
            total_lines_cleared += info["lines_cleared"]

        buffer.add_episode(obs_seq, action_seq, reward_seq, done_seq)

    return {
        "episodes": n_episodes,
        "total_steps": total_steps,
        "total_lines_cleared": total_lines_cleared,
    }


def random_policy(obs: np.ndarray) -> int:
    return int(np.random.randint(len(ACTIONS)))


if __name__ == "__main__":
    env = SimplifiedTetrisEnv(seed=0)
    buffer = ReplayBuffer(obs_dim=407, num_actions=len(ACTIONS))
    stats = collect_episodes(env, buffer, random_policy, n_episodes=20)
    print(stats)
