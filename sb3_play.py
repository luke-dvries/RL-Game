import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from sb3_connect4_env import Connect4

MODEL_PATH = "./selfplay_ckpts/selfplay_final.zip" #"best_model/best_model"  # or "maskable_ppo_connect4_final.zip" if you used that
SEED = 123


def mask_fn(env):
    return env.action_mask()


def ask_human_move(env: Connect4) -> int:
    """
    Called by the environment when opponent='human'.
    Must return an integer column [0..6]. The env will validate legality.
    """
    env.render()
    while True:
        try:
            s = input("Your move (0-6): ").strip()
            a = int(s)
            if 0 <= a <= 6:
                return a
            print("Please enter a number between 0 and 6.")
        except ValueError:
            print("Please enter a number between 0 and 6.")


def main():
    if not (os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH + ".zip")):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Update MODEL_PATH to your saved MaskablePPO model."
        )

    # Load trained MaskablePPO policy
    model = MaskablePPO.load(MODEL_PATH, device="auto")

    print("Welcome to Connect 4!")
    print("You play as 'O' (opponent), the AI plays as 'X' (agent).")
    print("Columns are 0..6. Good luck!\n")

    # Human plays as the OPPONENT; AI is the agent (same as during training)
    # We set opponent='human' and pass a callback for human moves.
    env = Connect4(
        seed=SEED,
        opponent="human",
        opponent_starts_prob=0.0,  # agent (AI) starts; set 0.5 to randomize who starts
        human_move_fn=ask_human_move,
    )
    # Wrap with action mask so the model cannot pick illegal columns
    env = ActionMasker(env, mask_fn)

    obs, info = env.reset()
    done = False

    while True:
        # AI (agent) turn — the env will immediately ask you for your move inside step()
        # because after the agent acts, it invokes the opponent ('human') policy.
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        if done:
            env.render()
            winner = info.get("winner", None)
            if winner == 1:
                print("\nGame Over! AI (X) wins!")
            elif winner == -1:
                print("\nGame Over! You (O) win! 🎉")
            else:
                print("\nGame Over! It's a draw.")
            # play again?
            again = input("Play again? (y/n): ").strip().lower()
            if again == "y":
                obs, info = env.reset()
                done = False
                continue
            break


if __name__ == "__main__":
    main()
