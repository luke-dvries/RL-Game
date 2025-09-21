import gymnasium as gym
import numpy as np

class Connect4(gym.Env):
    def __init__(self):
        super().__init__()
        # Flatten the observation space (6x7 = 42 values)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=2, 
            shape=(42,),  # Flattened 6x7 board
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(7)
        self.state = np.zeros((6, 7), dtype=np.float32)
        self.current_player = 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((6, 7), dtype=np.float32)
        self.current_player = 1
        return self.state.flatten(), {}  # Return flattened state

    def step(self, action):
        # Validate action
        if action < 0 or action >= 7 or np.all(self.state[:, action] != 0):
            return (
                self.state.flatten(),
                -10.0,  # Penalty for invalid move
                True,   # terminated
                False,  # truncated
                {"invalid_action": True}
            )

        # Find lowest empty row
        for row in range(5, -1, -1):
            if self.state[row, action] == 0:
                self.state[row, action] = self.current_player
                break

        # Check win condition
        terminated = bool(self.check_winner(row, action))  # Explicit boolean
        truncated = bool(np.all(self.state != 0) and not terminated)  # Explicit boolean

        # Reward scheme
        if terminated:
            reward = 1.0  # Win
        else:
            reward = 0.0  # Draw or ongoing

        # Switch player
        self.current_player = 3 - self.current_player

        return (
            self.state.flatten(),  # Flattened observation
            float(reward),         # Explicit float
            terminated,            # Boolean
            truncated,             # Boolean
            {}                     # Info dict
        )

    def check_winner(self, row, col):
        player = self.state[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
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
        print("\n")
        print(np.flip(self.state, 0))
        print("\n")