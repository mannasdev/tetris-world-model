import torch
from models.actor_critic import ActorCritic


def test_action_logits_and_value_shapes():
    ac = ActorCritic()
    h = torch.rand(4, 192)
    z = torch.rand(4, 144)
    logits = ac.action_logits(h, z)
    value = ac.value(h, z)
    assert logits.shape == (4, 7)
    assert value.shape == (4,)


def test_act_returns_valid_action_and_log_prob():
    ac = ActorCritic()
    h = torch.rand(4, 192)
    z = torch.rand(4, 144)
    action, log_prob = ac.act(h, z)
    assert action.shape == (4,)
    assert log_prob.shape == (4,)
    assert (action >= 0).all() and (action < 7).all()
