import pickle
import gymnasium as gym
import numpy as np
from testEnv import Connect4
from test import Connect4Agent
from collections import defaultdict


def human_move(board):
    print(np.flip(board, 0))
    while True:
        try:
            col = int(input("Your move! Enter column (0-6): "))
            if 0 <= col < 7 and board[0, col] == 0:
                return col
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Please enter a valid integer between 0 and 6.")

if __name__ == "__main__":
    env = Connect4()
    agent = Connect4Agent(
        env=env,
        learning_rate=0.01,
        initial_epsilon=0.0,  # No exploration for agent
        epsilon_decay=0.0,
        final_epsilon=0.0,
    )
    
    with open("connect4_qtable.pkl", "rb") as f:
        agent.q_values = pickle.load(f)
    agent.q_values = defaultdict(lambda: np.zeros(agent.env.action_space.n), agent.q_values)
    print("Q-table loaded from connect4_qtable.pkl")


    obs, info = env.reset()
    done = False

    print("You are Player 1 (pieces = 1). Agent is Player 2 (pieces = 2).")
    while not done:
        if env.current_player == 1:
            action = human_move(obs)
        else:
            action = agent.get_action(obs)
            print(f"Agent chooses column {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()

        if done:
            if terminated:
                winner = 3 - env.current_player  # The player who made the last move
                if winner == 1:
                    print("You win!")
                else:
                    print("Agent wins!")
            else:
                print("It's a draw!")

