import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.rewards import RewardsMapping
from tetris_gymnasium.components.tetromino_randomizer import TrueRandomizer
from tetris_gymnasium.components.tetromino_queue import TetrominoQueue

ACTIONS = [
    "move_left",
    "move_right",
    "rotate_cw",
    "rotate_ccw",
    "soft_drop",
    "hard_drop",
    "no_op",
]

# Native Tetris-Gymnasium action indices for each of our 7 actions, in order.
# Verified against the installed tetris_gymnasium/mappings/actions.py (ActionsMapping) —
# swap (hold, index 6) is intentionally excluded.
_NATIVE_ACTION = {
    "move_left": 0,
    "move_right": 1,
    "rotate_cw": 3,
    "rotate_ccw": 4,
    "soft_drop": 2,   # native "move_down"
    "hard_drop": 5,
    "no_op": 7,
}

BOARD_H, BOARD_W = 20, 10
PIECE_TYPES = 7
OBS_DIM = BOARD_H * BOARD_W + BOARD_H * BOARD_W + PIECE_TYPES


class _FixedTrueRandomizer(TrueRandomizer):
    """The installed tetris_gymnasium==0.2.1 TrueRandomizer.get_next_tetromino()
    calls `self.rng.randint(...)`, but Randomizer.reset() builds `self.rng` as a
    numpy.random.Generator (via np.random.default_rng()/PCG64), and Generator has
    no `.randint()` method (that's the legacy RandomState API — Generator uses
    `.integers()`). This is a genuine bug in the installed package: calling
    reset() then get_next_tetromino() on a stock TrueRandomizer always raises
    AttributeError. Fixed downstream in tetris_gymnasium 0.3.1, but upgrading
    would force numpy>=2.0 and a new jax/chex dependency that conflicts with
    this project's pinned environment, so we patch just the one method here."""

    def get_next_tetromino(self) -> int:
        return int(self.rng.integers(0, self.size))


class SimplifiedTetrisEnv:
    """Wraps Tetris-Gymnasium's Tetris env to enforce this project's
    simplifications: no hold, no next-piece preview. Wall kicks need no
    enforcement — the underlying env has no kick logic at all."""

    def __init__(self, width=10, height=20, seed=None):
        self._round = 1  # controls alife shaping; see set_round()
        self._seed = seed
        self._env = self._make_env(width, height, alife=1.0)

    def _make_env(self, width, height, alife):
        # NOTE: the installed tetris_gymnasium==0.2.1 Tetris.__init__ only assigns
        # self.queue / self.randomizer inside "if queue is None: ..." / "if
        # randomizer is None: ..." branches, so passing queue= / randomizer=
        # kwargs (as the task brief assumed) leaves those attributes unset and
        # reset()/step() crash with AttributeError. Work around it by constructing
        # with defaults, then assigning the custom queue/randomizer directly
        # before reset() is ever called — reset() only calls .reset()/.get_*() on
        # whatever self.queue/self.randomizer already reference, it never
        # recreates them, so this is honored correctly.
        randomizer = _FixedTrueRandomizer(PIECE_TYPES)
        env = Tetris(
            width=width,
            height=height,
            rewards_mapping=RewardsMapping(alife=alife, game_over=-10.0, invalid_action=-0.1),
        )
        env.randomizer = randomizer
        env.queue = TetrominoQueue(randomizer, size=1)
        return env

    def set_round(self, round_num: int):
        """Alife shaping stays on across every round (changed from the original
        design: alife=1.0 in round 1 only, alife=0.0 from round 2 on). Dropping
        it after round 1 meant the shared replay buffer mixed transitions with
        contradictory reward labels for near-identical states (surviving = +1
        in round-1 data, +0 in round-2+ data) -- a real cause of the
        actor-critic never improving past round 1 in practice, not just a
        theoretical risk. See "Known Limitations" in the design spec. Keeping
        alife constant across all rounds removes the label conflict entirely."""
        self._round = round_num
        width, height = self._env.width, self._env.height
        self._env = self._make_env(width, height, alife=1.0)

    def reset(self) -> np.ndarray:
        raw_obs, _info = self._env.reset(seed=self._seed)
        return self._encode(raw_obs)

    def step(self, action: int):
        native_action = _NATIVE_ACTION[ACTIONS[action]]
        raw_obs, reward, done, _truncated, info = self._env.step(native_action)
        obs = self._encode(raw_obs)
        piece_id = self._env.active_tetromino.id if self._env.active_tetromino is not None else 0
        piece_type = max(0, piece_id - len(self._env.base_pixels))
        return obs, float(reward), bool(done), {
            "lines_cleared": int(info["lines_cleared"]),
            "piece_type": piece_type,
        }

    def _encode(self, raw_obs) -> np.ndarray:
        board = raw_obs["board"]
        mask = raw_obs["active_tetromino_mask"]
        pad = self._env.padding
        board_crop = board[:BOARD_H, pad:pad + BOARD_W]
        mask_crop = mask[:BOARD_H, pad:pad + BOARD_W]
        board_bin = (board_crop > 0).astype(np.float32).reshape(-1)
        mask_bin = (mask_crop > 0).astype(np.float32).reshape(-1)

        piece_id = self._env.active_tetromino.id if self._env.active_tetromino is not None else 0
        piece_type = max(0, piece_id - len(self._env.base_pixels))
        piece_onehot = np.zeros(PIECE_TYPES, dtype=np.float32)
        piece_onehot[piece_type] = 1.0

        return np.concatenate([board_bin, mask_bin, piece_onehot]).astype(np.float32)

    @staticmethod
    def board_from_obs(obs: np.ndarray) -> np.ndarray:
        return obs[:BOARD_H * BOARD_W].reshape(BOARD_H, BOARD_W)
