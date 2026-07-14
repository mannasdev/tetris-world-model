import torch

from models.rssm import RSSMCore, rssm_loss

OBS_DIM = 407
ACTION_DIM = 7
BOARD_CELLS = 200
PIECE_TYPES = 7


def _fake_batch(batch_size=4, seq_len=6):
    return {
        "obs": torch.rand(batch_size, seq_len, OBS_DIM),
        "action": torch.eye(ACTION_DIM)[torch.randint(0, ACTION_DIM, (batch_size, seq_len))],
        "reward": torch.rand(batch_size, seq_len),
        "done": torch.zeros(batch_size, seq_len),
        "next_board": torch.randint(0, 2, (batch_size, seq_len, BOARD_CELLS)).float(),
        "next_piece": torch.randint(0, PIECE_TYPES, (batch_size, seq_len)),
    }


def test_initial_state_shapes():
    core = RSSMCore()
    h, z = core.initial_state(batch_size=4, device="cpu")
    assert h.shape == (4, 192)
    assert z.shape == (4, 12 * 12)


def test_step_posterior_shapes():
    core = RSSMCore()
    h, z = core.initial_state(batch_size=4, device="cpu")
    action = torch.eye(ACTION_DIM)[torch.randint(0, ACTION_DIM, (4,))]
    obs = torch.rand(4, OBS_DIM)
    h2, z2, prior_logits, post_logits = core.step_posterior(h, z, action, obs)
    assert h2.shape == (4, 192)
    assert z2.shape == (4, 12 * 12)
    assert prior_logits.shape == (4, 12, 12)
    assert post_logits.shape == (4, 12, 12)


def test_heads_shapes():
    core = RSSMCore()
    h, z = core.initial_state(batch_size=4, device="cpu")
    board_logits, piece_logits, reward, cont_logits = core.heads(h, z)
    assert board_logits.shape == (4, BOARD_CELLS)
    assert piece_logits.shape == (4, PIECE_TYPES)
    assert reward.shape == (4,)
    assert cont_logits.shape == (4,)


def test_loss_is_finite_and_decreases_on_overfit():
    torch.manual_seed(0)
    core = RSSMCore()
    batch = _fake_batch()
    opt = torch.optim.Adam(core.parameters(), lr=1e-3)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = rssm_loss(core, batch)["total"]
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # A tiny fixed batch should be learnable — this is the sanity check that
    # catches "loss doesn't go down" bugs (wrong shapes silently broadcasting,
    # detached gradients, etc.) before any real training run.
    assert losses[-1] < losses[0]
