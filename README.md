# Tetris World Model

An agent that learns to play a simplified Tetris **by dreaming**, not by practicing on the real game.

It first learns a mental model of how Tetris behaves, then improves its play entirely inside that imagined world. The real game is used only to gather experience and to grade the final result. This is the same idea behind DeepMind's DreamerV3 and the original "World Models" paper, shrunk to a single-machine, one-day build you can read end to end.

> New here? Read [The one idea](#the-one-idea-learning-by-dreaming), run the [Quickstart](#quickstart), then follow the [Reading order](#reading-order-how-to-understand-the-code).

---

## For the impatient

```bash
pip install -r requirements.txt   # torch, gymnasium, tetris-gymnasium, numpy, matplotlib, pytest
python run.py                     # runs the whole collect -> train -> validate -> dream loop
python visualize.py watch --policy random   # watch a game play out in your terminal
pytest                            # run the test suite
```

`python run.py` trains from scratch. On CPU that takes a while; it auto-uses an Apple Silicon (MPS) or NVIDIA (CUDA) GPU if you have one. To just *see* things move without training, use `visualize.py watch --policy random`.

---

## The one idea: learning by dreaming

Most game-playing agents learn by trial and error against the real game: try a move, see what happens, adjust. That is slow and needs a lot of real games.

This project splits the problem in two:

1. **A world model** learns to *predict* Tetris. Given the current board and an action, it predicts the next board, the next piece, the reward, and whether the game ended. It is just a predictor. It has no idea what "good play" means. Think of it as the agent building a Tetris simulator inside its own head by watching games.

2. **An actor-critic** (the actual player) then practices *only inside that imagined simulator*. It never touches the real game while learning. It plays thousands of imagined games against the world model, and gets better at them.

The bet is that if the imagined Tetris is faithful enough, getting good at the dream means getting good at the real thing.

A useful analogy: a chess player who studies by replaying positions in their head. The "board in their head" is the world model. The "getting better by thinking through lines" is the actor-critic training. They only sit down at a real board to check whether the practice paid off.

**Why bother?** Real games are expensive; imagined ones are cheap. Once you have a good world model, you can generate unlimited practice. The catch, and the interesting engineering, is keeping the dream honest (see [Keeping the dream honest](#keeping-the-dream-honest)).

---

## The training loop

Everything runs in rounds. Each round the agent gets a little better, so the next round collects data from smarter play. `run.py` orchestrates this.

```
  ┌─ every round ─────────────────────────────────────────────────────┐
  │                                                                    │
  │   1. COLLECT  ──▶  2. TRAIN  ──▶  3. VALIDATION ──▶  4. TRAIN      │
  │      real           WORLD           GATE               THE AGENT    │
  │      games          MODEL           dream vs           in pure      │
  │      with the       learns to       reality:           imagination  │
  │      current        predict the     does the model     (never the   │
  │      policy         next board,     stay accurate      real game)    │
  │         ▲           reward, done    for 15 steps?          │         │
  │         │                           if not, STOP           │         │
  │         └──────────  the improved policy collects ◀────────┘         │
  │                       better data next round                        │
  └────────────────────────────────────────────────────────────────────┘

  after the last round ──▶  eval.py: random policy vs dream-trained policy
```

1. **Collect** real games with the current policy (a random player on round 1) and store them.
2. **Train the world model** on everything collected so far, so it predicts Tetris more accurately.
3. **Validation gate.** Roll the world model forward "blind" for 15 steps and compare its imagined board against what really happened. If the dream drifts too far from reality, the pipeline **stops here** on purpose rather than training an agent on a broken simulator.
4. **Train the agent** purely inside the world model's imagination.

Then the improved agent collects the next round of real data, and the loop repeats.

---

## Keeping the dream honest

An agent practicing in its own imagination has an obvious failure mode: it can learn to exploit the *bugs* in its imagined world instead of learning real Tetris. Two guards address this.

- **An ensemble of world models.** The project trains **three** independent world models, not one. When they *disagree* about what happens next, that is a sign the agent has wandered into a situation none of them understand well, so the imagined game is truncated there. Trust the dream only where the models agree.
- **The validation gate.** Before any agent training in a round, the world model must prove it can predict 15 steps of real Tetris without drifting off. A single-step check is not enough; small errors compound over a rollout, and that is exactly the failure this gate catches.

These are the load-bearing ideas, and both come straight from the research this build is based on.

---

## Quickstart

**1. Install dependencies** (a virtual environment is recommended):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Run the full pipeline:**

```bash
python run.py
```

You will see per-round output: episodes collected, world-model loss, the validation-gate result, actor-critic loss, and a final random-vs-dream comparison. It writes:
- `world_model_ensemble.pt` and `actor_critic.pt` (trained weights),
- `plots/dream_vs_reality_round*.png` (gate plots),
- `logs/*.json` (loss curves).

**3. Watch it play** (any time after a run has produced the `.pt` files):

```bash
python visualize.py watch --policy dream    # the trained agent, live in the real game
python visualize.py dream                    # the agent playing inside its own imagination
python visualize.py compare                  # real board vs imagined board, side by side
python visualize.py progress                 # save training loss curves to a PNG
```

**4. Run the tests:**

```bash
pytest
```

---

## Reading order: how to understand the code

The code is small and every file has a smoke test at the bottom (`if __name__ == "__main__":`) you can run directly. Read in this order and each file builds on the last:

| # | File | What you learn |
|---|------|----------------|
| 1 | `env/tetris_env.py` | The game itself: the 7 actions, what the agent sees (a 407-number observation), and how reward works. Start here. |
| 2 | `models/rssm.py` | The **world model** (the dreamer): a GRU + a small stochastic latent that predicts the next board, piece, reward, and done. Also the 3-model ensemble. |
| 3 | `models/actor_critic.py` | The **player**: a tiny network that picks actions (actor) and estimates how good a situation is (critic). |
| 4 | `collect.py` + `replay_buffer.py` | How real games are played and stored as sequences for training. |
| 5 | `train_world_model.py` | Teaching the world model to predict, from stored real games. |
| 6 | `validate_world_model.py` | The reality check: roll the dream forward and measure how far it drifts. |
| 7 | `train_agent.py` | Learning to play from imagined games only (imagined rollouts + advantage-based updates). |
| 8 | `eval.py` | The scoreboard: random policy vs dream-trained policy, measured in lines cleared. |
| 9 | `run.py` | The conductor that runs 1 through 8 in rounds. Read this last to see how it all fits. |
| 10 | `visualize.py` | Optional but fun: watch any of the above happen in your terminal. |

If you only read two files, read `env/tetris_env.py` (what the agent sees) and `run.py` (how the whole thing is wired).

---

## Repository map

```
tetris-world-model/
├── run.py                    # entry point: the full multi-round training pipeline
├── env/
│   └── tetris_env.py         # SimplifiedTetrisEnv: wraps Tetris-Gymnasium, defines actions/obs/reward
├── models/
│   ├── rssm.py               # world model: RSSMCore, rssm_loss, RSSMEnsemble
│   └── actor_critic.py       # the policy + value network
├── collect.py                # play episodes (random or a trained policy) into the buffer
├── replay_buffer.py          # stores whole episodes, samples sequence chunks for the GRU
├── train_world_model.py      # supervised training of the world-model ensemble
├── validate_world_model.py   # the dream-vs-reality gate + divergence plot
├── train_agent.py            # actor-critic training inside imagined rollouts
├── eval.py                   # random vs dream-trained comparison (lines cleared)
├── visualize.py              # terminal viewer: watch / dream / compare / progress
├── render.py                 # ASCII board rendering used by visualize.py
├── device.py                 # picks CUDA > MPS > CPU automatically
├── tests/                    # pytest suite, one file per module
├── requirements.txt
└── pytest.ini
```

---

## How it actually works, in a bit more detail

**What the agent sees.** Each observation is 407 numbers: a 10x20 board (200 cells, occupied or empty), the active piece's shape as a 10x20 mask (another 200 cells), and a 7-way one-hot for the piece type. No next-piece preview: the upcoming piece is genuinely unknown until it appears, which is why the world model needs a *stochastic* (random-capable) part rather than a single confident guess.

**The seven actions:** move left, move right, rotate clockwise, rotate counter-clockwise, soft drop, hard drop, no-op. No hold piece, no wall kicks, on purpose. These simplifications are enforced by `SimplifiedTetrisEnv` and checked by `tests/test_env.py`.

**Reward.** The agent is mostly rewarded for *staying alive longer* (a small positive reward each step it survives), with a penalty for topping out. Lines cleared are tracked separately and are the metric `eval.py` reports. This split matters, see the honest note below.

**The world model (RSSM).** "RSSM" means Recurrent State-Space Model. It keeps a running memory `h` (a GRU) and a small random latent `z` that stands in for the hidden randomness (which piece comes next). From `(h, z)` it predicts the next board, next piece, reward, and whether the game continues. It is trained by plain supervised learning; reward is just one more number it predicts, not something it tries to maximize.

**The actor-critic.** Given the world model's belief state `(h, z)`, the actor outputs a distribution over the 7 actions and the critic estimates future reward. It trains by playing imagined games (roll the world model forward ~15 steps, collect imagined rewards, update with an advantage estimate). It never sees the real game during training.

---

## Honest status

This is a research build, not a Tetris champion. Read this so the results are not surprising:

- The pipeline runs end to end: collect, train, validate, dream-train, evaluate.
- The interesting, working part is the **world model and the guards around it** (the ensemble-disagreement truncation and the validation gate).
- At the scales tried so far, the agent learns mainly to **survive longer**, not to clear lines. No round has produced a real line clear yet, so "lines cleared per game" is not where the signal shows up; episode length is.
- Along the way, two real bugs were found and fixed, and they are good lessons in why world-model RL is tricky:
  - A **hard-drop-spam exploit**: because the underlying game only pays out reward when a piece locks, the agent learned to slam every piece straight down and top out almost immediately. Fixed by adding a per-step survival reward (`PER_TICK_SURVIVAL_REWARD` in `env/tetris_env.py`).
  - A **reward-label conflict**: an earlier design turned reward shaping off after round 1, which left the shared replay buffer full of contradictory labels for near-identical states, and the agent exploited that instead of improving. Fixed by keeping the shaping constant across all rounds (`SimplifiedTetrisEnv.set_round`).

If you extend this, larger-scale runs, a narrower board, or a richer reward are the natural next things to try.

---

## Requirements and environment notes

- **Python 3.10+** and the packages in `requirements.txt` (PyTorch, Gymnasium, Tetris-Gymnasium, NumPy, Matplotlib, pytest).
- **GPU is optional.** `device.py` auto-selects CUDA, then Apple MPS, then CPU. You do not need to configure anything.
- **Tetris-Gymnasium 0.2.1 quirks.** `env/tetris_env.py` monkeypatches two rough edges in that exact library version (a `RandomState` vs `Generator` bug in the piece randomizer, and how the queue/randomizer must be attached). If you upgrade the library, revisit those two workarounds; the reasons are documented inline.

---

## Credit

The architecture follows DreamerV3 (Hafner et al.) and the original World Models paper (Ha and Schmidhuber), scaled down for a board-state (not pixel) Tetris and a single-day build.
