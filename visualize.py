# visualize.py
"""Terminal visualization for the Tetris world model project.

  watch     -- watch a policy (random or dream-trained) play live, in the
               real environment
  dream     -- watch the agent play entirely inside its own imagination:
               no real environment involved after one anchoring step
  compare   -- watch the real board and the world model's imagined board
               side by side, frame by frame, under the same action sequence
  progress  -- plot world-model and actor-critic training loss curves
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from device import get_device
from env.tetris_env import SimplifiedTetrisEnv, ACTIONS
from collect import random_policy
from models.rssm import RSSMEnsemble
from models.actor_critic import ActorCritic
from eval import DreamTrainedPolicy
from validate_world_model import rollout_real_and_dream, board_divergence
from render import render_board, render_side_by_side

_CLEAR = "\033c"


def _load_ensemble_and_actor_critic():
    device = get_device()
    ensemble = RSSMEnsemble(n_models=3).to(device)
    ensemble.load_state_dict(torch.load("world_model_ensemble.pt", map_location=device))
    actor_critic = ActorCritic().to(device)
    actor_critic.load_state_dict(torch.load("actor_critic.pt", map_location=device))
    return ensemble, actor_critic


def cmd_watch(args):
    env = SimplifiedTetrisEnv(seed=args.seed)
    if args.policy == "random":
        policy = random_policy
    else:
        ensemble, actor_critic = _load_ensemble_and_actor_critic()
        policy = DreamTrainedPolicy(ensemble, actor_critic)

    if hasattr(policy, "reset"):
        policy.reset()
    obs = env.reset()
    done = False
    step = 0
    total_lines = 0
    while not done and step < args.max_steps:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        total_lines += info["lines_cleared"]
        step += 1
        print(_CLEAR, end="")
        print(render_board(SimplifiedTetrisEnv.board_from_obs(obs), title=f"{args.policy} policy"))
        print(f"step {step}  action={ACTIONS[action]}  reward={reward:.2f}  lines_cleared={total_lines}")
        time.sleep(args.delay)
    print(f"\ngame over after {step} steps, {total_lines} lines cleared")


def cmd_compare(args):
    env = SimplifiedTetrisEnv(seed=args.seed)
    ensemble, _actor_critic = _load_ensemble_and_actor_critic()
    actions = [int(np.random.randint(len(ACTIONS))) for _ in range(args.horizon)]
    result = rollout_real_and_dream(env, ensemble, member_idx=0, action_sequence=actions)
    divergence = board_divergence(result["real_boards"], result["dream_boards"])

    for t, (real_b, dream_b, div) in enumerate(zip(result["real_boards"], result["dream_boards"], divergence)):
        print(_CLEAR, end="")
        print(render_side_by_side(real_b, dream_b))
        print(f"step {t + 1}/{len(divergence)}  divergence={div:.3f}")
        time.sleep(args.delay)


def cmd_dream(args):
    """Watch the agent play entirely inside its own imagination. One real
    observation anchors the starting belief state (same convention as
    validate_world_model.rollout_real_and_dream and train_agent.imagine_rollout);
    after that, nothing rendered here comes from the real environment again."""
    env = SimplifiedTetrisEnv(seed=args.seed)
    ensemble, actor_critic = _load_ensemble_and_actor_critic()
    num_actions = len(ACTIONS)
    device = next(ensemble.parameters()).device

    start_obs = env.reset()
    with torch.no_grad():
        states = ensemble.initial_state(batch_size=1, device=device)
        zero_action = torch.zeros(1, num_actions, device=device)
        start_obs_t = torch.from_numpy(start_obs).unsqueeze(0).to(device)
        anchored = []
        for member, (h, z) in zip(ensemble.members, states):
            h2, z2, _prior, _post = member.step_posterior(h, z, zero_action, start_obs_t)
            anchored.append((h2, z2))
        states = anchored

        step = 0
        total_reward = 0.0
        while step < args.horizon:
            h, z = states[args.member]
            action, _log_prob = actor_critic.act(h, z)
            action_onehot = torch.nn.functional.one_hot(action, num_classes=num_actions).float()

            states, disagreement, per_member_heads = ensemble.imagine_step(states, action_onehot)
            heads = per_member_heads[args.member]
            # .cpu() before .numpy(): MPS/CUDA tensors can't convert to numpy directly.
            board_prob = torch.sigmoid(heads["board_logits"]).reshape(20, 10).cpu().numpy()
            board = (board_prob > 0.5).astype(np.float32)
            reward = heads["reward"].item()
            p_continue = torch.sigmoid(heads["continue_logits"]).item()
            total_reward += reward
            step += 1

            print(_CLEAR, end="")
            print(render_board(board, title="DREAM -- imagined, not real"))
            print(f"step {step}/{args.horizon}  action={ACTIONS[action.item()]}  "
                  f"imagined_reward={reward:.2f}  ensemble_disagreement={disagreement.item():.3f}  "
                  f"p(continue)={p_continue:.2f}")
            if disagreement.item() > args.disagreement_threshold:
                print("  ^ disagreement above threshold -- real training would truncate the dream here")
            time.sleep(args.delay)

            if p_continue < 0.5:
                print("\nmodel predicts game over (imagined)")
                break

    print(f"\ndream ended after {step} steps, total imagined reward {total_reward:.2f}")


def cmd_progress(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = [
        (axes[0], "logs/world_model_losses.json", "world model loss"),
        (axes[1], "logs/actor_critic_losses.json", "actor-critic loss"),
    ]
    for ax, path, title in panels:
        if not os.path.exists(path):
            ax.set_title(f"{title} (no log yet — run.py hasn't trained this component)")
            continue
        with open(path) as f:
            losses = json.load(f)
        ax.plot(losses)
        ax.set_title(title)
        ax.set_xlabel("training step")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out)
    print(f"saved {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    watch = sub.add_parser("watch", help="watch a policy play live in the terminal")
    watch.add_argument("--policy", choices=["random", "dream"], default="dream")
    watch.add_argument("--seed", type=int, default=None)
    watch.add_argument("--delay", type=float, default=0.15)
    watch.add_argument("--max-steps", type=int, default=2000)
    watch.set_defaults(func=cmd_watch)

    dream = sub.add_parser("dream", help="watch the agent play entirely inside its own imagination")
    dream.add_argument("--seed", type=int, default=None)
    dream.add_argument("--horizon", type=int, default=15)
    dream.add_argument("--delay", type=float, default=0.4)
    dream.add_argument("--member", type=int, default=0, help="which ensemble member to render")
    dream.add_argument("--disagreement-threshold", type=float, default=0.15,
                        help="ensemble inter-member disagreement (per-cell RMS) threshold above "
                             "which the dream would be truncated in real training -- not the same "
                             "metric as validate_world_model.validate's board_divergence threshold, "
                             "despite sharing the same default")
    dream.set_defaults(func=cmd_dream)

    compare = sub.add_parser("compare", help="watch real vs. dream boards side by side")
    compare.add_argument("--seed", type=int, default=None)
    compare.add_argument("--horizon", type=int, default=15)
    compare.add_argument("--delay", type=float, default=0.4)
    compare.set_defaults(func=cmd_compare)

    progress = sub.add_parser("progress", help="plot training loss curves")
    progress.add_argument("--out", default="plots/training_progress.png")
    progress.set_defaults(func=cmd_progress)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
