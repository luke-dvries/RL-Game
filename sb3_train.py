import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

# Maskable PPO + action masking wrapper from sb3-contrib
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from sb3_connect4_env import Connect4

NUM_ENVS = 8
SEED = 42
TOTAL_TIMESTEPS = 500_000

def mask_fn(env):
    # Must return a boolean array of shape (7,)
    return env.action_mask()

def make_env(idx: int, opponent: str = "random"):
    def _init():
        # You can switch opponent="heuristic" later for a tougher eval/train
        env = Connect4(seed=SEED + idx, opponent=opponent, opponent_starts_prob=0.0)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init

def main():
    os.makedirs("./best_model", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    set_random_seed(SEED)

    # Quick sanity check on a single raw env (no wrappers)
    check_env(Connect4(seed=SEED), warn=True)

    # Vectorized training envs
    vec_env = SubprocVecEnv([make_env(i, opponent="random") for i in range(NUM_ENVS)])

    # Separate evaluation env (slightly stronger opponent recommended once learning starts)
    eval_env = make_env(10, opponent="heuristic")()
    
    # Model (flat 42-dim obs => MLP policy)
    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=max(64, 2048 // NUM_ENVS),  # per-env steps; keep rollout size ~2k+
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,               # exploration reduced thanks to masking
        verbose=1,
        seed=SEED,
        device="auto",
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=max(2_000 // NUM_ENVS, 1),
        deterministic=False,  # keep False; opponent is stochastic/heuristic
        render=False,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)
    model.save("maskable_ppo_connect4_final")
    print("Training completed! Model saved as maskable_ppo_connect4_final")

if __name__ == "__main__":
    main()
