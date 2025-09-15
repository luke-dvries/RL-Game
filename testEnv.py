import gymnasium as gym
import numpy as np
from collections import defaultdict

class Connect4(gym.Env):
    def __init__(self):
        super(Connect4, self).__init__()
        # Action space: 7 possible columns to drop a piece
        self.action_space = gym.spaces.Discrete(7)
        # Observation space: 6 rows x 7 columns, values 0 (empty), 1 (player 1), 2 (player 2)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.int32)
        # Game state: board initialized to all zeros
        self.state = np.zeros((6, 7), dtype=np.int32)
        # Player 1 starts
        self.current_player = 1

    def reset(self, *, seed=None, options=None):
        # Reset the board and set current player to 1
        self.state = np.zeros((6, 7), dtype=np.int32)
        self.current_player = 1
        return self.state, {}

    def step(self, action):
        # Validate action: must be a valid column and not full
        if action < 0 or action >= 7 or np.all(self.state[:, action] != 0):
            raise ValueError("Invalid action")

        # Find the lowest empty row in the chosen column
        for row in range(5, -1, -1):
            if self.state[row, action] == 0:
                self.state[row, action] = self.current_player
                break

        # Check if the move ended the game (win or draw)
        terminated = self.check_winner(row, action)
        truncated = np.all(self.state != 0) and not terminated  # Draw if board is full and no winner

        done = terminated or truncated
        reward = 1 if terminated else 0
        self.current_player = 3 - self.current_player
        return self.state, reward, terminated, truncated, {}

    def check_winner(self, row, col):
        # Check if the last move at (row, col) resulted in a win for the current player
        player = self.state[row, col]
        # Directions: vertical, horizontal, diagonal (both ways)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # Check both directions from the placed piece
            for d in [1, -1]:
                r, c = row + d * dr, col + d * dc
                while 0 <= r < 6 and 0 <= c < 7 and self.state[r, c] == player:
                    count += 1
                    if count >= 4:
                        return True
                    r += d * dr
                    c += d * dc
        return False

    def render(self):
        # Print the board with the bottom row at the bottom
        print(np.flip(self.state, 0))