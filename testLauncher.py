from tqdm import tqdm  # Progress bar
import gymnasium as gym  # OpenAI Gym for environments
from test import BlackjackAgent
import numpy as np
from collections import defaultdict
from testEnv import Connect4
from test import Connect4Agent
import pickle 

gym.envs.registration.register(
    id="Connect4-v0",
    entry_point="testEnv:Connect4",
)

# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 100_000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration

# Create environment and agent
# env = gym.make("Blackjack-v1", sab=False)
# env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# agent = BlackjackAgent(
#     env=env,
#     learning_rate=learning_rate,
#     initial_epsilon=start_epsilon,
#     epsilon_decay=epsilon_decay,
#     final_epsilon=final_epsilon,
# )

env = gym.make("Connect4-v0")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# Create agent
agent = Connect4Agent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# Training loop
for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    done = False

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from this experience
        #agent.update(obs, action, reward, terminated, next_obs)

        agent.update(obs, action, reward, next_obs, done)

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()

# ---- Plotly version of your plots (interactive, WSL-friendly) ----
# pip install plotly kaleido   # kaleido only needed if you also want PNG export


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(list(arr)).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500

# Compute the same series you had before
reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
training_error_moving_average = get_moving_avgs(agent.training_error, rolling_length, "same")

# X-axes for each series
x_rewards = list(range(len(reward_moving_average)))
x_lengths = list(range(len(length_moving_average)))
x_errors  = list(range(len(training_error_moving_average)))

# Create a 1x3 subplot layout to mirror your matplotlib figure
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Episode rewards", "Episode lengths", "Training Error")
)

# Episode rewards (win/loss performance)
fig.add_trace(
    go.Scatter(x=x_rewards, y=reward_moving_average, mode="lines", name="Average Reward"),
    row=1, col=1
)
fig.update_xaxes(title_text="Episode", row=1, col=1)
fig.update_yaxes(title_text="Average Reward", row=1, col=1)

# Episode lengths (how many actions per hand)
fig.add_trace(
    go.Scatter(x=x_lengths, y=length_moving_average, mode="lines", name="Average Episode Length"),
    row=1, col=2
)
fig.update_xaxes(title_text="Episode", row=1, col=2)
fig.update_yaxes(title_text="Average Episode Length", row=1, col=2)

# Training error (how much we're still learning)
fig.add_trace(
    go.Scatter(x=x_errors, y=training_error_moving_average, mode="lines", name="TD Error"),
    row=1, col=3
)
fig.update_xaxes(title_text="Step", row=1, col=3)
fig.update_yaxes(title_text="Temporal Difference Error", row=1, col=3)

fig.update_layout(
    width=1200, height=500,
    showlegend=False,
    margin=dict(l=50, r=30, t=40, b=40)
)

# best fit line function

def add_best_fit_line(fig, x, y, row, col, name="Best Fit"):
    # Fit a line (degree 1 polynomial)
    coeffs = np.polyfit(x, y, 2)
    best_fit = np.polyval(coeffs, x)
    fig.add_trace(
        go.Scatter(x=x, y=best_fit, mode="lines", name=name, line=dict(dash="dash")),
        row=row, col=col
    )

# Add best fit lines to each subplot
add_best_fit_line(fig, x_rewards, reward_moving_average, row=1, col=1, name="Reward Trend")
add_best_fit_line(fig, x_lengths, length_moving_average, row=1, col=2, name="Length Trend")
add_best_fit_line(fig, x_errors, training_error_moving_average, row=1, col=3, name="Error Trend")

# ...existing code...
fig.update_layout(
    width=1200, height=500,
    showlegend=True,  # Show legend to distinguish best fit lines
    margin=dict(l=50, r=30, t=40, b=40)
)
# ...existing code...

# Save to an interactive HTML file (best for WSL)
fig.write_html("training_results.html", include_plotlyjs="cdn")
print("Wrote training_results.html — open it in your browser.")

# Optional: also save a static PNG (needs 'kaleido' installed)
# fig.write_image("training_results.png", width=1200, height=500, scale=2)

import pickle

# Save Q-table to file
with open("connect4_qtable.pkl", "wb") as f:
    pickle.dump(dict(agent.q_values), f)
print("Q-table saved to connect4_qtable.pkl")
