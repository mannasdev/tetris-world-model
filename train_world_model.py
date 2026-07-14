# train_world_model.py
import torch
from models.rssm import RSSMEnsemble, rssm_loss
from replay_buffer import ReplayBuffer


def build_training_batch(raw_batch: dict) -> dict:
    next_obs = raw_batch["next_obs"]
    next_board = next_obs[:, :, :200]
    next_piece = next_obs[:, :, 400:407].argmax(dim=-1)
    return {**raw_batch, "next_board": next_board, "next_piece": next_piece}


def train_world_model(ensemble: RSSMEnsemble, buffer: ReplayBuffer, steps: int,
                       batch_size=32, seq_len=15, lr=1e-3, log_path=None) -> list:
    optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in ensemble.members]
    losses = []

    for _ in range(steps):
        step_losses = []
        for member, opt in zip(ensemble.members, optimizers):
            # Independently-sampled batch per member (not shared) — members
            # must be able to diverge on scarce/novel data for the ensemble
            # disagreement signal (Task 4) to mean anything.
            raw = buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len)
            batch = build_training_batch(raw)
            opt.zero_grad()
            loss = rssm_loss(member, batch)["total"]
            loss.backward()
            opt.step()
            step_losses.append(loss.item())
        losses.append(sum(step_losses) / len(step_losses))

    if log_path is not None:
        _append_losses(log_path, losses)

    return losses


def _append_losses(log_path: str, losses: list):
    import json
    import os
    existing = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            existing = json.load(f)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(existing + losses, f)


if __name__ == "__main__":
    import numpy as np
    from env.tetris_env import SimplifiedTetrisEnv, ACTIONS
    from collect import collect_episodes, random_policy

    env = SimplifiedTetrisEnv(seed=0)
    buffer = ReplayBuffer(obs_dim=407, num_actions=len(ACTIONS))
    collect_episodes(env, buffer, random_policy, n_episodes=50)

    ensemble = RSSMEnsemble(n_models=3)
    losses = train_world_model(ensemble, buffer, steps=500)
    print(f"final loss: {losses[-1]:.4f} (started at {losses[0]:.4f})")
    torch.save(ensemble.state_dict(), "world_model_ensemble.pt")
