import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from sb3_connect4_env import Connect4

def play_against_ai():
    # Load the trained model
    model = PPO.load("best_model/best_model")
    env = Connect4()
    
    # Play game
    obs, info = env.reset()
    done = False
    
    print("Welcome to Connect 4!")
    print("You are Player 1 (value 1), AI is Player 2 (value 2)")
    print("Enter a column number between 0-6 to make your move")
    
    while not done:
        env.render()
        
        if env.current_player == 1:
            # Human player's turn
            while True:
                try:
                    action = int(input("\nYour move (0-6): "))
                    if 0 <= action <= 6 and not np.all(env.state[:, action] != 0):
                        break
                    print("Invalid move! Column is either full or out of range")
                except ValueError:
                    print("Please enter a number between 0 and 6")
        else:
            # AI's turn
            action, _ = model.predict(obs, deterministic=True)
            print(f"\nAI plays column {action}")
        
        # Make move
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Check game end
        if done:
            env.render()
            if terminated:
                winner = "You win!" if env.current_player == 2 else "AI wins!"
                print(f"\nGame Over! {winner}")
            else:
                print("\nGame Over! It's a draw!")
            
            # Ask to play again
            play_again = input("\nPlay again? (y/n): ").lower()
            if play_again == 'y':
                obs, info = env.reset()
                done = False
                print("\nNew game starting...")

if __name__ == "__main__":
    play_against_ai()