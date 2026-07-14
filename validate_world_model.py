import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.tetris_env import SimplifiedTetrisEnv, ACTIONS
from models.rssm import RSSMEnsemble


def rollout_real_and_dream(env: SimplifiedTetrisEnv, ensemble: RSSMEnsemble, member_idx: int, action_sequence: list) -> dict:
    start_obs = env.reset()
    real_obs = start_obs
    real_boards = []
    for action in action_sequence:
        real_obs, _reward, done, _info = env.step(action)
        real_boards.append(SimplifiedTetrisEnv.board_from_obs(real_obs))
        if done:
            break

    member = ensemble.members[member_idx]
    with torch.no_grad():
        # Anchor imagination to the real starting observation via one posterior
        # step (same convention Task 9's imagine_rollout uses) before switching
        # to pure prior-only rollout — an unanchored dream starting from a blank
        # zero state isn't a fair "dream vs reality from the same start" test,
        # and it wouldn't match how imagination is actually initiated during
        # actor-critic training.
        h0, z0 = member.initial_state(batch_size=1, device="cpu")
        zero_action = torch.zeros(1, len(ACTIONS))
        start_obs_t = torch.from_numpy(start_obs).unsqueeze(0)
        h, z, _prior_logits, _post_logits = member.step_posterior(h0, z0, zero_action, start_obs_t)

        dream_boards = []
        for action in action_sequence[:len(real_boards)]:
            action_onehot = torch.eye(len(ACTIONS))[action].unsqueeze(0)
            h, z, _prior_logits = member.step_prior(h, z, action_onehot)
            board_logits, _piece_logits, _reward, _cont_logits = member.heads(h, z)
            board_prob = torch.sigmoid(board_logits).reshape(20, 10).numpy()
            dream_boards.append((board_prob > 0.5).astype(np.float32))

    return {
        "real_boards": np.stack(real_boards[:len(dream_boards)]),
        "dream_boards": np.stack(dream_boards),
    }


def board_divergence(real_boards, dream_boards) -> np.ndarray:
    return np.abs(real_boards - dream_boards).mean(axis=(1, 2))


def validate(env: SimplifiedTetrisEnv, ensemble: RSSMEnsemble, horizon=15, n_held_out=10,
             threshold=0.15, plot_path="plots/dream_vs_reality.png") -> dict:
    all_divergences = []
    for _ in range(n_held_out):
        actions = [int(np.random.randint(len(ACTIONS))) for _ in range(horizon)]
        result = rollout_real_and_dream(env, ensemble, member_idx=0, action_sequence=actions)
        div = board_divergence(result["real_boards"], result["dream_boards"])
        if len(div) < horizon:
            div = np.pad(div, (0, horizon - len(div)), constant_values=1.0)
        all_divergences.append(div)

    all_divergences = np.stack(all_divergences)  # (n_held_out, horizon)
    mean_curve = all_divergences.mean(axis=0)
    mean_final_divergence = float(mean_curve[-1])

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure()
    plt.plot(mean_curve)
    plt.axhline(threshold, linestyle="--", label="pass threshold")
    plt.xlabel("imagination step")
    plt.ylabel("mean board divergence")
    plt.title("Dream vs. reality divergence")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    return {
        "passed": mean_final_divergence < threshold,
        "mean_final_divergence": mean_final_divergence,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    env = SimplifiedTetrisEnv(seed=0)
    ensemble = RSSMEnsemble(n_models=3)
    ensemble.load_state_dict(torch.load("world_model_ensemble.pt"))
    result = validate(env, ensemble, horizon=15, n_held_out=10)
    print(result)
    if not result["passed"]:
        print("GATE FAILED — per design doc, stop here and ship the world-model-only result.")
