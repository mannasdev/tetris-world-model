import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSMCore(nn.Module):
    """Single RSSM-lite: GRU deterministic state + categorical stochastic
    latent. Sized per the design doc's compute-budget finding (well below
    DreamerV3's 512-dim/32x32 defaults, appropriate for a ~400-dim
    board-state observation rather than pixels)."""

    def __init__(self, obs_dim=407, action_dim=7, board_cells=200, piece_types=7,
                 hidden_dim=192, latent_categories=12, latent_classes=12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_categories = latent_categories
        self.latent_classes = latent_classes
        latent_dim = latent_categories * latent_classes

        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        feat_dim = hidden_dim + latent_dim
        self.board_head = nn.Linear(feat_dim, board_cells)
        self.piece_head = nn.Linear(feat_dim, piece_types)
        self.reward_head = nn.Linear(feat_dim, 1)
        self.continue_head = nn.Linear(feat_dim, 1)

    def _sample(self, logits_flat):
        logits = logits_flat.view(-1, self.latent_categories, self.latent_classes)
        sample = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        return sample.reshape(logits_flat.shape[0], -1), logits

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_categories * self.latent_classes, device=device)
        return h, z

    def step_prior(self, h, z, action):
        h_next = self.gru(torch.cat([z, action], dim=-1), h)
        z_prior, prior_logits = self._sample(self.prior_net(h_next))
        return h_next, z_prior, prior_logits

    def step_posterior(self, h, z, action, obs):
        h_next, _z_prior, prior_logits = self.step_prior(h, z, action)
        post_logits_flat = self.posterior_net(torch.cat([h_next, obs], dim=-1))
        z_post, post_logits = self._sample(post_logits_flat)
        return h_next, z_post, prior_logits, post_logits

    def heads(self, h, z):
        feat = torch.cat([h, z], dim=-1)
        return (
            self.board_head(feat),
            self.piece_head(feat),
            self.reward_head(feat).squeeze(-1),
            self.continue_head(feat).squeeze(-1),
        )


def rssm_loss(core: RSSMCore, batch: dict, kl_weight: float = 1.0) -> dict:
    """Teacher-forced (posterior-driven) training loss over a sequence chunk.
    `batch` must have obs (B,T,obs_dim), action (B,T,action_dim), reward (B,T),
    done (B,T), next_board (B,T,board_cells), next_piece (B,T) int64."""
    B, T = batch["action"].shape[0], batch["action"].shape[1]
    device = batch["obs"].device
    h, z = core.initial_state(B, device)

    recon_total = torch.zeros((), device=device)
    kl_total = torch.zeros((), device=device)

    for t in range(T):
        h, z, prior_logits, post_logits = core.step_posterior(
            h, z, batch["action"][:, t], batch["obs"][:, t]
        )
        board_logits, piece_logits, reward_pred, cont_logits = core.heads(h, z)

        recon_total = recon_total + F.binary_cross_entropy_with_logits(
            board_logits, batch["next_board"][:, t]
        )
        recon_total = recon_total + F.cross_entropy(piece_logits, batch["next_piece"][:, t])
        recon_total = recon_total + F.mse_loss(reward_pred, batch["reward"][:, t])
        recon_total = recon_total + F.binary_cross_entropy_with_logits(
            cont_logits, 1.0 - batch["done"][:, t]
        )

        # KL balancing term (design doc — not optional): gives the prior training
        # pressure to match the posterior, so prior-only imagination tracks
        # what the model learned from real (posterior-derived) data.
        post_dist = torch.distributions.Categorical(logits=post_logits)
        prior_dist = torch.distributions.Categorical(logits=prior_logits)
        kl_total = kl_total + torch.distributions.kl_divergence(post_dist, prior_dist).sum(-1).mean()

    recon_total = recon_total / T
    kl_total = kl_total / T
    return {"recon": recon_total, "kl": kl_total, "total": recon_total + kl_weight * kl_total}


class RSSMEnsemble(nn.Module):
    """N independently-initialized RSSMCore members. Disagreement across
    members' board predictions is the dream-trust signal (design doc):
    imagination should be truncated/downweighted where members disagree,
    since that marks states outside what any member confidently learned."""

    def __init__(self, n_models=3, **rssm_core_kwargs):
        super().__init__()
        self.members = nn.ModuleList([RSSMCore(**rssm_core_kwargs) for _ in range(n_models)])

    def initial_state(self, batch_size, device):
        return [m.initial_state(batch_size, device) for m in self.members]

    def imagine_step(self, states, action):
        new_states = []
        disagreement_probs = []
        per_member_heads = []
        for member, (h, z) in zip(self.members, states):
            h_next, z_next, prior_logits = member.step_prior(h, z, action)
            # Predictions used downstream (reward the actor-critic trains on, etc.)
            # come from the hard-sampled z_next, matching proper RSSM rollout
            # semantics (a genuine discrete state must propagate forward).
            board_logits, piece_logits, reward, cont_logits = member.heads(h_next, z_next)
            new_states.append((h_next, z_next))
            per_member_heads.append({
                "board_logits": board_logits, "piece_logits": piece_logits,
                "reward": reward, "continue_logits": cont_logits,
            })

            # Disagreement is computed from the DETERMINISTIC expected latent
            # (softmax over prior_logits, not the hard gumbel sample above).
            # Using the hard sample here would conflate genuine epistemic
            # (parameter) disagreement between members with per-step sampling
            # noise — identical-weight members would still show nonzero
            # "disagreement" from independently-drawn Gumbel noise alone,
            # which defeats the point of the signal (see tests/test_rssm_ensemble.py).
            soft_z = torch.softmax(prior_logits, dim=-1).reshape(h_next.shape[0], -1)
            disagree_feat = torch.cat([h_next, soft_z], dim=-1)
            disagree_board_logits = member.board_head(disagree_feat)
            disagreement_probs.append(torch.sigmoid(disagree_board_logits))

        stacked = torch.stack(disagreement_probs, dim=0)  # (n_models, B, board_cells)
        n = stacked.shape[0]
        pairwise_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                # mean(-1), not sum(-1): this must stay per-cell RMS, on the same
                # O(0-1) scale as validate_world_model.board_divergence's per-cell
                # mean-abs-diff, since both are compared against the same
                # disagreement_threshold default (0.15) across Tasks 4/9/13. A
                # sum(-1) here reintroduces a ~sqrt(board_cells) scale blowup that
                # silently truncates every imagined rollout at step 0 (see Task 11's
                # scale-fix report — the real pipeline run once measured exactly
                # this, and actor-critic loss was 0.0 for all 800 training steps).
                pairwise_dists.append((stacked[i] - stacked[j]).pow(2).mean(-1).sqrt())
        disagreement = torch.stack(pairwise_dists, dim=0).mean(0) if pairwise_dists else torch.zeros(
            stacked.shape[1], device=stacked.device, dtype=stacked.dtype
        )

        return new_states, disagreement, per_member_heads
