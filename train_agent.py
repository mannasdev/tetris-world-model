# train_agent.py
import torch
from models.rssm import RSSMEnsemble
from models.actor_critic import ActorCritic
from replay_buffer import ReplayBuffer


def imagine_rollout(ensemble: RSSMEnsemble, actor_critic: ActorCritic, start_obs_batch: torch.Tensor,
                     # disagreement_threshold compares against ensemble INTER-MEMBER
                     # disagreement (per-cell RMS across members, see ensemble.imagine_step) --
                     # NOT the same metric as validate_world_model.validate's board_divergence
                     # threshold, despite sharing the same 0.15 default and O(0-1) scale.
                     horizon=15, disagreement_threshold=0.15) -> dict:
    B = start_obs_batch.shape[0]
    # GRU input was cat([z, action]); latent_dim is known from members[0], so
    # action_dim is derived from the GRU's input width rather than duplicated
    # as a hardcoded constant that could silently drift out of sync.
    latent_dim = ensemble.members[0].latent_categories * ensemble.members[0].latent_classes
    num_actions = ensemble.members[0].gru.weight_ih.shape[1] - latent_dim

    states = ensemble.initial_state(B, device=start_obs_batch.device)
    # Bootstrap h,z for every member from one posterior step on the real start observation,
    # using a zero action (no-op) — this anchors imagination to a real state per the design
    # doc's "dreams are only trustworthy near experience" principle (see the CartPole repo's
    # Act 4 finding, which motivated this project's validation gate in the first place).
    zero_action = torch.zeros(B, num_actions, device=start_obs_batch.device)
    states = [m.step_posterior(h, z, zero_action, start_obs_batch)[:2] for m, (h, z) in zip(ensemble.members, states)]

    log_probs, values, rewards, valid_mask = [], [], [], []
    still_valid = torch.ones(B, dtype=torch.bool, device=start_obs_batch.device)

    for _t in range(horizon):
        h0, z0 = states[0]
        action, log_prob = actor_critic.act(h0, z0)
        value = actor_critic.value(h0, z0)
        action_onehot = torch.nn.functional.one_hot(action, num_classes=num_actions).float()

        states, disagreement, per_member_heads = ensemble.imagine_step(states, action_onehot)
        reward = per_member_heads[0]["reward"]

        still_valid = still_valid & (disagreement < disagreement_threshold)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        valid_mask.append(still_valid.float())

    return {
        "log_probs": torch.stack(log_probs, dim=1),
        "values": torch.stack(values, dim=1),
        "rewards": torch.stack(rewards, dim=1),
        "valid_mask": torch.stack(valid_mask, dim=1),
    }


def actor_critic_loss(rollout: dict, gamma=0.99, gae_lambda=0.95) -> torch.Tensor:
    rewards, values, mask = rollout["rewards"], rollout["values"], rollout["valid_mask"]
    B, T = rewards.shape

    # The GAE backward recursion carries `next_value`/`gae` from step t into step t-1's
    # advantage. Without masking, a truncated (post-disagreement-threshold) step's reward
    # and value estimate would silently bootstrap-contaminate the advantage/return of the
    # last *trustworthy* step before it, even though that truncated step's own advantage
    # gets zeroed out by `mask` below — defeating the point of truncation. Masking `delta`
    # and `gae` at each step (not just the final advantages/loss) stops the recursion carry
    # dead at the trust boundary: once mask[:, t] == 0, gae/next_value freeze at zero and no
    # contribution from step t or beyond can reach step t-1.
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device)
    next_value = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        m = mask[:, t]
        delta = (rewards[:, t] + gamma * next_value - values[:, t]) * m
        gae = delta + gamma * gae_lambda * gae * m
        advantages[:, t] = gae
        next_value = values[:, t] * m

    returns = advantages + values
    policy_loss = -(rollout["log_probs"] * advantages.detach() * mask).sum() / mask.sum().clamp(min=1)
    value_loss = (((values - returns.detach()) ** 2) * mask).sum() / mask.sum().clamp(min=1)
    return policy_loss + 0.5 * value_loss


def train_agent(ensemble: RSSMEnsemble, actor_critic: ActorCritic, buffer: ReplayBuffer, steps: int,
                 batch_size=32, horizon=15, disagreement_threshold=0.15, lr=1e-3, log_path=None) -> list:
    opt = torch.optim.Adam(actor_critic.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        raw = buffer.sample_sequences(batch_size=batch_size, seq_len=1)
        start_obs = raw["obs"][:, 0]

        rollout = imagine_rollout(ensemble, actor_critic, start_obs, horizon=horizon,
                                   disagreement_threshold=disagreement_threshold)
        opt.zero_grad()
        loss = actor_critic_loss(rollout)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    if log_path is not None:
        from train_world_model import _append_losses
        _append_losses(log_path, losses)

    return losses


if __name__ == "__main__":
    import torch as _torch
    from env.tetris_env import SimplifiedTetrisEnv, ACTIONS
    from collect import collect_episodes, random_policy

    env = SimplifiedTetrisEnv(seed=0)
    buffer = ReplayBuffer(obs_dim=407, num_actions=len(ACTIONS))
    collect_episodes(env, buffer, random_policy, n_episodes=50)

    ensemble = RSSMEnsemble(n_models=3)
    ensemble.load_state_dict(_torch.load("world_model_ensemble.pt"))
    actor_critic = ActorCritic()
    losses = train_agent(ensemble, actor_critic, buffer, steps=200)
    print(f"final actor-critic loss: {losses[-1]:.4f}")
    _torch.save(actor_critic.state_dict(), "actor_critic.pt")
