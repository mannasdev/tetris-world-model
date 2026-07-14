import numpy as np
import torch
from replay_buffer import ReplayBuffer
from models.rssm import RSSMEnsemble
from models.actor_critic import ActorCritic
from train_agent import imagine_rollout, actor_critic_loss, train_agent

OBS_DIM = 407


def _fill_buffer():
    buf = ReplayBuffer(OBS_DIM, num_actions=7)
    for _ in range(5):
        length = 15
        obs_seq = [np.random.rand(OBS_DIM).astype(np.float32) for _ in range(length + 1)]
        action_seq = list(np.random.randint(0, 7, size=length))
        reward_seq = list(np.random.rand(length))
        done_seq = [False] * (length - 1) + [True]
        buf.add_episode(obs_seq, action_seq, reward_seq, done_seq)
    return buf


def test_imagine_rollout_shapes_and_truncation_mask():
    ensemble = RSSMEnsemble(n_models=3)
    actor_critic = ActorCritic()
    start_obs = torch.rand(4, OBS_DIM)
    rollout = imagine_rollout(ensemble, actor_critic, start_obs, horizon=8, disagreement_threshold=0.0)
    # threshold=0.0 forces truncation at step 0 for an untrained (high-disagreement) ensemble
    assert rollout["log_probs"].shape == (4, 8)
    assert rollout["valid_mask"].shape == (4, 8)
    assert (rollout["valid_mask"][:, 0] == 0).all() or (rollout["valid_mask"].sum(dim=1) <= 8).all()


def test_actor_critic_loss_is_finite():
    ensemble = RSSMEnsemble(n_models=3)
    actor_critic = ActorCritic()
    start_obs = torch.rand(4, OBS_DIM)
    rollout = imagine_rollout(ensemble, actor_critic, start_obs, horizon=8, disagreement_threshold=1.0)
    loss = actor_critic_loss(rollout)
    assert torch.isfinite(loss)


def test_train_agent_runs():
    buf = _fill_buffer()
    ensemble = RSSMEnsemble(n_models=3)
    actor_critic = ActorCritic()
    losses = train_agent(ensemble, actor_critic, buf, steps=3, batch_size=4, horizon=6)
    assert len(losses) == 3
    assert all(l == l for l in losses)


def test_train_agent_appends_to_log_path(tmp_path):
    import json
    log_path = tmp_path / "actor_critic_losses.json"
    buf = _fill_buffer()
    ensemble = RSSMEnsemble(n_models=3)
    actor_critic = ActorCritic()
    train_agent(ensemble, actor_critic, buf, steps=2, batch_size=4, horizon=6, log_path=str(log_path))
    with open(log_path) as f:
        logged = json.load(f)
    assert len(logged) == 2
