from collections import defaultdict
import gymnasium as gym
import numpy as np


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

#connect 4 agent
class Connect4Agent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []
    
    def _obs_to_key(self, obs): # helper function to convert observation to a hashable key
        # Convert numpy array to a hashable type (tuple)
        return tuple(obs.flatten())

    def get_action(self, obs: tuple) -> int:
        key = self._obs_to_key(obs)
        # Find valid actions (columns that are not full)
        valid_actions = [c for c in range(self.env.action_space.n) if obs[0, c] == 0]

        if not valid_actions:
            # No valid moves, just pick something (shouldn't happen)
            return self.env.action_space.sample()

        if np.random.random() < self.epsilon:
            # Explore: pick a random valid action
            return np.random.choice(valid_actions)
        else:
            # Exploit: pick the best valid action
            q_values = self.q_values[key]
            # Mask invalid actions by setting their Q-value very low
            masked_q = np.full_like(q_values, -np.inf)
            for a in valid_actions:
                masked_q[a] = q_values[a]
            return int(np.argmax(masked_q))

    def update(
        self,
        obs: tuple,
        action: int,
        reward: float,
        next_obs: tuple,
        done: bool,
    ):
        """Update Q-values based on observed transition.

        Args:
            obs: Current state
            action: Action taken
            reward: Reward received after taking action
            next_obs: Next state after taking action
            done: Whether the episode has ended
        """
        key = self._obs_to_key(obs)# Convert observation to a hashable key
        next_key = self._obs_to_key(next_obs)

        current_q = self.q_values[key][action]

        if done:
            target = reward  # No future rewards if episode is done
        else:
            next_max_q = np.max(self.q_values[next_key])
            target = reward + self.discount_factor * next_max_q

        temporal_difference = target - current_q
        self.q_values[key][action] += self.lr * temporal_difference

        # Track learning progress
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)