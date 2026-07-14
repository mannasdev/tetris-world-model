# run.py
"""Orchestrates the 2-4 round collect -> train world model -> validate ->
train agent -> collect more loop, with the design doc's time-boxed
checkpoints. This script is meant to be run interactively and watched —
it prints explicit STOP instructions rather than silently continuing past
a failed gate, per the design doc's kill-criteria finding."""
import torch
from env.tetris_env import SimplifiedTetrisEnv, ACTIONS
from replay_buffer import ReplayBuffer
from models.rssm import RSSMEnsemble
from models.actor_critic import ActorCritic
from collect import collect_episodes, random_policy
from train_world_model import train_world_model
from validate_world_model import validate
from train_agent import train_agent
from eval import evaluate_policy, DreamTrainedPolicy

N_ROUNDS = 6
EPISODES_PER_ROUND = 150
WORLD_MODEL_STEPS_PER_ROUND = 2000
AGENT_STEPS_PER_ROUND = 1000


def main():
    env = SimplifiedTetrisEnv(seed=0)
    buffer = ReplayBuffer(obs_dim=407, num_actions=len(ACTIONS))
    ensemble = RSSMEnsemble(n_models=3)
    actor_critic = ActorCritic()

    policy = random_policy
    for round_num in range(1, N_ROUNDS + 1):
        # NOTE: round 1's alife=1.0 shaping vs. round 2+'s alife=0.0 means the replay
        # buffer mixes transitions with contradictory reward labels; see "Known
        # Limitations" in the design spec (docs/superpowers/specs/2026-07-14-tetris-world-model-design.md).
        env.set_round(round_num)
        print(f"=== round {round_num}: collecting {EPISODES_PER_ROUND} episodes ===")
        stats = collect_episodes(env, buffer, policy, n_episodes=EPISODES_PER_ROUND)
        print(stats)
        # NOTE: no round-1 "zero line clears" kill-criterion here (removed after
        # real use exposed it as broken): a random policy over 50 short episodes
        # essentially never clears a line by chance, so total_lines_cleared==0 was
        # always true on round 1 regardless of whether alife shaping was providing
        # signal -- it tested the wrong thing. The world-model validation gate
        # below is the real quality checkpoint; it actually measures whether the
        # model learned something, and correctly stops the pipeline if not.

        print(f"=== round {round_num}: training world model ===")
        losses = train_world_model(ensemble, buffer, steps=WORLD_MODEL_STEPS_PER_ROUND,
                                    log_path="logs/world_model_losses.json")
        print(f"world model loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

        print(f"=== round {round_num}: validation gate ===")
        result = validate(env, ensemble, horizon=15, n_held_out=10,
                           plot_path=f"plots/dream_vs_reality_round{round_num}.png")
        print(result)
        if not result["passed"]:
            print("GATE FAILED. Per the design doc's time-boxed checkpoint: stop tuning "
                  "the RSSM and ship the world-model-only result (the divergence plot) "
                  "rather than proceeding to actor-critic training on an unvalidated model.")
            torch.save(ensemble.state_dict(), "world_model_ensemble.pt")
            return

        print(f"=== round {round_num}: training actor-critic in imagination ===")
        agent_losses = train_agent(ensemble, actor_critic, buffer, steps=AGENT_STEPS_PER_ROUND,
                                    log_path="logs/actor_critic_losses.json")
        print(f"actor-critic loss: {agent_losses[0]:.4f} -> {agent_losses[-1]:.4f}")

        dream_policy = DreamTrainedPolicy(ensemble, actor_critic)
        policy = dream_policy  # next round collects with the improved policy

    torch.save(ensemble.state_dict(), "world_model_ensemble.pt")
    torch.save(actor_critic.state_dict(), "actor_critic.pt")

    print("=== final evaluation ===")
    env.set_round(N_ROUNDS + 1)  # alife=0, matches how the agent was actually trained/evaluated
    # (see "Known Limitations" in the design spec re: mixed-round reward labels in the buffer)
    random_result = evaluate_policy(env, random_policy, n_games=20)
    dream_result = evaluate_policy(env, DreamTrainedPolicy(ensemble, actor_critic), n_games=20)
    print(f"random policy:       {random_result}")
    print(f"dream-trained policy: {dream_result}")


if __name__ == "__main__":
    main()
