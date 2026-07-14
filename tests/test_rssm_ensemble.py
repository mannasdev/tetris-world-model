import torch
from models.rssm import RSSMEnsemble

ACTION_DIM = 7


def test_ensemble_has_n_members():
    ens = RSSMEnsemble(n_models=3)
    assert len(ens.members) == 3


def test_imagine_step_shapes():
    ens = RSSMEnsemble(n_models=3)
    states = ens.initial_state(batch_size=4, device="cpu")
    action = torch.eye(ACTION_DIM)[torch.randint(0, ACTION_DIM, (4,))]
    new_states, disagreement, heads = ens.imagine_step(states, action)
    assert len(new_states) == 3
    assert disagreement.shape == (4,)
    assert (disagreement >= 0).all()


def test_disagreement_is_zero_for_identical_members():
    # If all members share weights, they must produce identical predictions
    # and therefore zero disagreement — the sanity check the eng review
    # flagged as missing (does disagreement behave correctly at all,
    # before trusting it to gate imagination).
    ens = RSSMEnsemble(n_models=3)
    shared_state_dict = ens.members[0].state_dict()
    for member in ens.members[1:]:
        member.load_state_dict(shared_state_dict)

    states = ens.initial_state(batch_size=4, device="cpu")
    action = torch.eye(ACTION_DIM)[torch.randint(0, ACTION_DIM, (4,))]
    torch.manual_seed(0)
    _new_states, disagreement, _heads = ens.imagine_step(states, action)
    assert torch.allclose(disagreement, torch.zeros_like(disagreement), atol=1e-5)
