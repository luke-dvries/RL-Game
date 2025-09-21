import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os

def load_training_logs(log_dir="./logs"):
    """Load training metrics from the logs directory."""
    evaluations_path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(evaluations_path):
        raise FileNotFoundError(f"No evaluation logs found at {evaluations_path}")
    
    data = np.load(evaluations_path)
    return {
        'timesteps': data['timesteps'],
        'results': data['results'],
        'ep_lengths': data['ep_lengths']
    }

def create_training_dashboard(logs):
    """Create an interactive dashboard with Plotly."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Average Reward per Episode',
            'Episode Lengths',
            'Reward Distribution',
            'Training Progress'
        )
    )

    # Average reward over time
    fig.add_trace(
        go.Scatter(
            x=logs['timesteps'],
            y=np.mean(logs['results'], axis=1),
            mode='lines',
            name='Average Reward',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # Episode lengths over time
    fig.add_trace(
        go.Scatter(
            x=logs['timesteps'],
            y=np.mean(logs['ep_lengths'], axis=1),
            mode='lines',
            name='Episode Length',
            line=dict(color='green')
        ),
        row=1, col=2
    )

    # Reward distribution (latest evaluation)
    fig.add_trace(
        go.Histogram(
            x=logs['results'][-1],
            name='Reward Distribution',
            nbinsx=20,
            marker_color='red'
        ),
        row=2, col=1
    )

    # Training progress (cumulative average reward)
    cumulative_avg = np.cumsum(np.mean(logs['results'], axis=1)) / np.arange(1, len(logs['results']) + 1)
    fig.add_trace(
        go.Scatter(
            x=logs['timesteps'],
            y=cumulative_avg,
            mode='lines',
            name='Cumulative Average',
            line=dict(color='purple')
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="Connect4 Training Metrics",
        height=800,
        showlegend=True,
        template="plotly_dark"
    )

    return fig

def save_dashboard(fig, output_path="training_results.html"):
    """Save the dashboard to an HTML file."""
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True
    )

if __name__ == "__main__":
    # Load training logs
    logs = load_training_logs()
    
    # Create dashboard
    fig = create_training_dashboard(logs)
    
    # Save to HTML
    save_dashboard(fig)
    
    # Start a local server to view the dashboard
    # import http.server
    # import socketserver
    
    # PORT = 8000
    
    # class Handler(http.server.SimpleHTTPRequestHandler):
    #     def end_headers(self):
    #         self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
    #         super().end_headers()
    
    # print(f"Starting server at http://localhost:{PORT}")
    # print("Open your browser and navigate to the above URL to view the dashboard")
    # print("Press Ctrl+C to stop the server")
    
    # with socketserver.TCPServer(("", PORT), Handler) as httpd:
    #     httpd.serve_forever()