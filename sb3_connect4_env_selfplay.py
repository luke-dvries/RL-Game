import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, Callable

# Type for a function the env can call to get the opponent's move.
# Signature: (env, obs: np.ndarray, mask: np.ndarray[bool]) -> int
OpponentFn = Callable[["Connect4", np.ndarray, np.ndarray], int]


class Connect4(gym.Env):
    """
    Single-agent Connect-4 environment with multiple opponent modes:
      - "random": uniform over legal moves
      - "heuristic": try to win, then block, then prefer center
      - "policy": call a user-provided function (e.g., frozen PPO) to select the opponent move
      - "human": call a user-provided function to ask a human

    Board encoding (fixed perspective):
      +1 = agent (the learner, "X")
      -1 = opponent (the sparring partner, "O")
       0 = empty

    Rewards:
      +1.0  agent win
      -1.0  agent loss (opponent win)
       0.0  draw
     -0.01  step penalty (configurable)
     -0.05  illegal move penalty (no turn consumed)

    Observation: flat (42,) float32 in {-1, 0, +1}
    Action space: Discrete(7)
    Info includes "action_mask" (bool[7]), "winner", "opp_action".
    """
    metadata = {"render_modes": ["human"]}

    ROWS = 6
    COLS = 7

    def __init__(
        self,
        seed: Optional[int] = None,
        opponent: str = "random",              # "random", "heuristic", "policy", "human"
        step_penalty: float = -0.01,
        illegal_penalty: float = -0.05,
        opponent_starts_prob: float = 0.0,
        render_mode: Optional[str] = None,
        # Used when opponent="policy" or "human"
        opponent_move_fn: Optional[OpponentFn] = None,
    ):
        super().__init__()
        assert opponent in {"random", "heuristic", "policy", "human"}
        assert 0.0 <= opponent_starts_prob <= 1.0

        self.opponent_type = opponent
        self.step_penalty = float(step_penalty)
        self.illegal_penalty = float(illegal_penalty)
        self.opponent_starts_prob = float(opponent_starts_prob)
        self.render_mode = render_mode

        # Callback for "policy" or "human" opponents
        self.opponent_move_fn = opponent_move_fn

        # Board: int8 in {-1, 0, +1}; +1 = agent, -1 = opponent
        self.state = np.zeros((self.ROWS, self.COLS), dtype=np.int8)

        self.agent_id = 1
        self.opp_id = -1

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.ROWS * self.COLS,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.COLS)

        self.np_random = np.random.default_rng(seed)
        self._seed_value = seed

    # ---------- Gymnasium API ----------

    def seed(self, seed: Optional[int] = None) -> None:
        self.np_random = np.random.default_rng(seed)
        self._seed_value = seed

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        self.state.fill(0)

        opp_started = False
        if self.np_random.random() < self.opponent_starts_prob:
            # Opponent plays first
            a = self._select_opponent_action()
            if a is not None:
                self._drop_piece(a, self.opp_id)
                opp_started = True

        obs = self._obs()
        info = {"action_mask": self.action_mask(), "winner": None, "opp_started": opp_started}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Agent move
        if not self._is_legal(action):
            obs = self._obs()
            info = {"action_mask": self.action_mask(), "invalid_action": True, "winner": None}
            return obs, float(self.illegal_penalty), False, False, info

        row = self._drop_piece(action, self.agent_id)

        if self._is_win(row, action, self.agent_id):
            obs = self._obs()
            info = {"action_mask": self.action_mask(), "winner": self.agent_id}
            return obs, 1.0, True, False, info

        if self._is_draw():
            obs = self._obs()
            info = {"action_mask": self.action_mask(), "winner": 0}
            return obs, 0.0, True, False, info

        # Opponent move
        opp_action = self._select_opponent_action()
        opp_row = self._drop_piece(opp_action, self.opp_id)

        if self._is_win(opp_row, opp_action, self.opp_id):
            obs = self._obs()
            info = {"action_mask": self.action_mask(), "winner": self.opp_id, "opp_action": opp_action}
            return obs, -1.0, True, False, info

        if self._is_draw():
            obs = self._obs()
            info = {"action_mask": self.action_mask(), "winner": 0, "opp_action": opp_action}
            return obs, 0.0, True, False, info

        obs = self._obs()
        info = {"action_mask": self.action_mask(), "winner": None, "opp_action": opp_action}
        return obs, float(self.step_penalty), False, False, info

    def render(self) -> None:
        chars = {0: ".", 1: "X", -1: "O"}
        print("\n  0 1 2 3 4 5 6")
        for r in range(self.ROWS):
            print(" ", " ".join(chars[int(v)] for v in self.state[r]))
        print()

    # ---------- Public helpers ----------

    def action_mask(self) -> np.ndarray:
        return (self.state[0, :] == 0)

    # ---------- Internal helpers ----------

    def _obs(self) -> np.ndarray:
        return self.state.astype(np.float32, copy=False).reshape(-1)

    def _is_legal(self, action: int) -> bool:
        return 0 <= action < self.COLS and self.state[0, action] == 0

    def _legal_actions(self) -> np.ndarray:
        return np.nonzero(self.state[0, :] == 0)[0]

    def _drop_piece(self, col: int, player: int) -> int:
        for r in range(self.ROWS - 1, -1, -1):
            if self.state[r, col] == 0:
                self.state[r, col] = player
                return r
        raise RuntimeError("Tried to drop into a full column")

    def _is_draw(self) -> bool:
        return not np.any(self.state[0, :] == 0)

    def _is_win(self, row: int, col: int, player: int) -> bool:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.state[r, c] == player:
                count += 1; r += dr; c += dc
            r, c = row - dr, col - dc
            while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.state[r, c] == player:
                count += 1; r -= dr; c -= dc
            if count >= 4:
                return True
        return False

    def _select_opponent_action(self) -> Optional[int]:
        legal = self._legal_actions()
        if legal.size == 0:
            return None

        mask = self.action_mask()
        obs = self._obs()

        if self.opponent_type == "random":
            return int(self.np_random.choice(legal))

        if self.opponent_type == "heuristic":
            # Win now?
            for a in legal:
                if self._would_win(a, self.opp_id): return a
            # Block agent?
            for a in legal:
                if self._would_win(a, self.agent_id): return a
            for a in [3, 2, 4, 1, 5, 0, 6]:
                if a in legal: return a
            return int(self.np_random.choice(legal))

        if self.opponent_type in {"policy", "human"}:
            if self.opponent_move_fn is None:
                raise RuntimeError(f"Opponent type '{self.opponent_type}' requires opponent_move_fn.")
            # The callback must return an int action; we validate & repair if needed.
            a = int(self.opponent_move_fn(self, obs, mask))
            if self._is_legal(a):
                return a
            # If the policy proposes illegal (e.g., not masked), fall back to any legal move
            return int(self.np_random.choice(legal))

        raise RuntimeError("Unknown opponent type")

    def _would_win(self, col: int, player: int) -> bool:
        r = None
        for rr in range(self.ROWS - 1, -1, -1):
            if self.state[rr, col] == 0:
                r = rr; break
        if r is None:
            return False
        self.state[r, col] = player
        won = self._is_win(r, col, player)
        self.state[r, col] = 0
        return won
