import os
import shutil
from collections import deque
from typing import Optional, List, Callable

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

from sb3_connect4_env_selfplay import Connect4

# ------------------ Config ------------------

NUM_ENVS = 8
SEED = 42
TOTAL_TIMESTEPS = 500_000

# How often to freeze the current agent as an opponent (in env steps, not gradient steps)
UPDATE_INTERVAL = 50_000

# Keep a small pool of past opponents to diversify training
OPPONENT_POOL_SIZE = 5

SAVE_DIR = "./selfplay_ckpts"
BEST_DIR = "./best_model"
LOG_DIR = "./logs"

# ------------------------------------------------

def mask_fn(env):
    return env.action_mask()

class OpponentPool:
    """
    Holds frozen MaskablePPO policies. Provides a callable that envs use to pick an action
    for the opponent="policy" player.

    We keep the interface simple: for each opponent move request,
    we use the *current* head of the pool (or sample uniformly).
    """
    def __init__(self, sample_latest_prob: float = 0.7):
        self.paths: deque[str] = deque(maxlen=OPPONENT_POOL_SIZE)
        self.models: deque[MaskablePPO] = deque(maxlen=OPPONENT_POOL_SIZE)
        self.sample_latest_prob = sample_latest_prob

    def __len__(self):
        return len(self.models)

    def add_snapshot(self, model: MaskablePPO, tag: str) -> None:
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR, f"opponent_{tag}")
        model.save(path)
        # Load a fresh copy so we don't keep references to the live model
        frozen = MaskablePPO.load(path, device="cpu")  # opponent inference can be on CPU
        self.paths.append(path)
        self.models.append(frozen)
        print(f"[OpponentPool] Added snapshot: {path} (pool size={len(self.models)})")

    def pick_model(self) -> Optional[MaskablePPO]:
        if not self.models:
            return None
        if np.random.random() < self.sample_latest_prob:
            return self.models[-1]
        idx = np.random.randint(0, len(self.models))
        return self.models[idx]

    def opponent_move_fn(self):
        """
        Returns a function(env, obs, mask) -> action int
        Uses the picked frozen model to predict. If the model proposes an illegal action,
        we just pick a random legal one (the env will re-check).
        """
        def _fn(env: Connect4, obs: np.ndarray, mask: np.ndarray) -> int:
            model = self.pick_model()
            if model is None:
                # fallback: random if pool is empty
                legal_cols = np.nonzero(mask)[0]
                return int(np.random.choice(legal_cols))
            # SB3 expects batch obs; also note we're not inside ActionMasker here,
            # so we can't pass masks directly. We'll check legality after.
            action, _ = model.predict(obs, deterministic=True)
            a = int(action)
            if 0 <= a < env.COLS and mask[a]:
                return a
            # If illegal, pick a random legal move
            legal_cols = np.nonzero(mask)[0]
            return int(np.random.choice(legal_cols))
        return _fn


def make_training_env(idx: int, pool: OpponentPool):
    """
    Each worker env uses opponent='policy' and calls into the pool's opponent_move_fn().
    Initially the pool might be empty; that's fine—the opponent_move_fn falls back to random.
    """
    def _init():
        env = Connect4(
            seed=SEED + idx,
            opponent="policy",
            opponent_starts_prob=0.0,
            opponent_move_fn=pool.opponent_move_fn(),
        )
        env = ActionMasker(env, mask_fn)  # mask for the learning agent
        env = Monitor(env)
        return env
    return _init


def make_eval_env(opponent: str = "heuristic", pool: Optional[OpponentPool] = None):
    """
    Eval against a fixed heuristic (recommended) or the latest pool opponent.
    """
    def _init():
        if opponent == "policy" and pool is not None and len(pool) > 0:
            env = Connect4(
                seed=SEED + 999,
                opponent="policy",
                opponent_starts_prob=0.0,
                opponent_move_fn=pool.opponent_move_fn(),
            )
        else:
            env = Connect4(
                seed=SEED + 999,
                opponent="heuristic",
                opponent_starts_prob=0.0,
            )
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    set_random_seed(SEED)

    # Create opponent pool (empty at start)
    pool = OpponentPool(sample_latest_prob=0.7)

    # Vectorized training envs using the pool as the opponent
    vec_env = SubprocVecEnv([make_training_env(i, pool) for i in range(NUM_ENVS)])

    # Separate evaluation env (vs heuristic by default)
    eval_env = SubprocVecEnv([make_eval_env("heuristic", pool)])

    # Create learning agent
    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=max(64, 2048 // NUM_ENVS),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        seed=SEED,
        device="auto",
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=LOG_DIR,
        eval_freq=max(2_000 // NUM_ENVS, 1),
        deterministic=False,
        render=False,
    )

    # ---- Train in segments; after each segment, freeze a snapshot as the new opponent ----
    steps_done = 0
    while steps_done < TOTAL_TIMESTEPS:
        steps_to_do = min(UPDATE_INTERVAL, TOTAL_TIMESTEPS - steps_done)
        model.learn(total_timesteps=steps_to_do, reset_num_timesteps=False, callback=eval_cb, progress_bar=True)
        steps_done += steps_to_do

        # Freeze current policy into the opponent pool
        tag = f"{steps_done}"
        pool.add_snapshot(model, tag)

        # (Optional) Also drop a copy of the "latest agent" weights for your records
        latest_path = os.path.join(SAVE_DIR, "latest_agent")
        model.save(latest_path)
        print(f"[SelfPlay] Frozen current agent as opponent at {steps_done} steps.")

    # Save final learner
    model.save(os.path.join(SAVE_DIR, "selfplay_final"))
    print("Training finished. Final model saved to ./selfplay_ckpts/selfplay_final.zip")


if __name__ == "__main__":
    main()
