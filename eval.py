# eval.py
import numpy as np
import torch
from env.tetris_env import SimplifiedTetrisEnv, ACTIONS
from models.rssm import RSSMEnsemble
from models.actor_critic import ActorCritic
from collect import random_policy


class DreamTrainedPolicy:
    """Stateful policy for real-environment evaluation: maintains the RSSM's
    (h, z) belief state across real steps via the posterior (since real
    observations are available at eval time, unlike during imagination)."""

    def __init__(self, ensemble: RSSMEnsemble, actor_critic: ActorCritic):
        self.core = ensemble.members[0]
        self.actor_critic = actor_critic
        self.num_actions = len(ACTIONS)
        self.h = None
        self.z = None

    def reset(self):
        self.h, self.z = self.core.initial_state(batch_size=1, device="cpu")

    def __call__(self, obs: np.ndarray) -> int:
        zero_action = torch.zeros(1, self.num_actions)
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            self.h, self.z, _prior, _post = self.core.step_posterior(self.h, self.z, zero_action, obs_t)
            action, _log_prob = self.actor_critic.act(self.h, self.z)
        return int(action.item())


def evaluate_policy(env: SimplifiedTetrisEnv, policy_fn, n_games=20) -> dict:
    total_lines = 0
    for _ in range(n_games):
        if hasattr(policy_fn, "reset"):
            policy_fn.reset()
        obs = env.reset()
        done = False
        while not done:
            action = policy_fn(obs)
            obs, _reward, done, info = env.step(action)
            total_lines += info["lines_cleared"]
    return {"mean_lines_cleared": total_lines / n_games, "games": n_games}


if __name__ == "__main__":
    env = SimplifiedTetrisEnv(seed=1)
    random_result = evaluate_policy(env, random_policy, n_games=20)
    print(f"random policy: {random_result}")

    ensemble = RSSMEnsemble(n_models=3)
    ensemble.load_state_dict(torch.load("world_model_ensemble.pt"))
    actor_critic = ActorCritic()
    actor_critic.load_state_dict(torch.load("actor_critic.pt"))
    dream_policy = DreamTrainedPolicy(ensemble, actor_critic)
    dream_result = evaluate_policy(env, dream_policy, n_games=20)
    print(f"dream-trained policy: {dream_result}")
