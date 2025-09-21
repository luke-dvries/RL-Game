import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from sb3_connect4_env import Connect4

def train():
    # Create and register the environment
    env = Connect4()
    
    # Validate the environment
    check_env(env)
    
    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1
    )
    
    # Create evaluation callback
    eval_env = Connect4()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the agent
    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("ppo_connect4_final")
    print("Training completed! Model saved as ppo_connect4_final")

if __name__ == "__main__":
    train()