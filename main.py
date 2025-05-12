import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm 
import framework
import utils
import time

# Import the agent classes
from agent_tabular import TabularQLAgent
from agent_linear import LinearQLAgent
from agent_dqn import DeepQLAgent

# --- Configuration ---
NUM_RUNS = 5 # Number of independent runs for averaging
AGENT_TYPES = ['tabular', 'linear', 'dqn'] # Agents to run

# Agent-specific configurations
AGENT_CONFIGS = {
    'tabular': {'class': TabularQLAgent, 'num_epochs': 200, 'alpha': 0.1, 'training_ep': 0.5},
    'linear': {'class': LinearQLAgent, 'num_epochs': 600, 'alpha': 0.01, 'training_ep': 0.5},
    'dqn': {'class': DeepQLAgent, 'num_epochs': 300, 'alpha': 0.1, 'training_ep': 0.5, 'hidden_size': 128}
}

# --- Common Setup ---
print("Loading game data and preparing dictionaries...")
# Load game data 
framework.load_game_data()

# Tabular Agent Specific Setup
(dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
NUM_ROOM_DESC = len(dict_room_desc)
NUM_QUESTS = len(dict_quest_desc)

# Linear/DQN Agent Specific Setup
state_texts = utils.load_data('game.tsv')
dictionary = utils.bag_of_words(state_texts)
STATE_DIM = len(dictionary) # Dimension for feature vectors
print("Setup complete.")

# --- Data Storage ---
all_results = {agent: [] for agent in AGENT_TYPES}

# --- Running Experiments ---
start_time_total = time.time()
print(f"\nStarting {NUM_RUNS} runs for each agent: {', '.join(AGENT_TYPES)}")
print("WARNING: This may take a very long time, especially for DQN!")

for agent_type in AGENT_TYPES:
    start_time_agent = time.time()
    print(f"\n--- Running Agent: {agent_type.upper()} ---")
    config = AGENT_CONFIGS[agent_type]
    agent_class = config['class']
    agent_runs_rewards = []

    pbar_runs = tqdm(range(NUM_RUNS), desc=f"Runs ({agent_type})", ncols=100)
    for i in pbar_runs:
        # Instantiate agent using config
        if agent_type == 'tabular':
            agent = agent_class(
                num_epochs=config['num_epochs'],
                alpha=config['alpha'],
                training_ep=config['training_ep']
            )
            # Run the agent - Redirect tqdm output from agent.run if desired
            run_rewards = agent.run(NUM_ROOM_DESC, NUM_QUESTS, dict_room_desc, dict_quest_desc)
        elif agent_type == 'linear':
            agent = agent_class(
                num_epochs=config['num_epochs'],
                alpha=config['alpha'],
                training_ep=config['training_ep']
            )
            run_rewards = agent.run(dictionary, STATE_DIM)
        elif agent_type == 'dqn':
            agent = agent_class(
                num_epochs=config['num_epochs'],
                alpha=config['alpha'],
                training_ep=config['training_ep'],
                hidden_size=config['hidden_size']
            )
            run_rewards = agent.run(dictionary, STATE_DIM)
        else:
             raise ValueError(f"Unknown agent type in config: {agent_type}")

        # Ensure the run produced results of the expected length
        if len(run_rewards) == config['num_epochs']:
            agent_runs_rewards.append(run_rewards)
        else:
             print(f"Warning: Run {i+1} for {agent_type} produced {len(run_rewards)} epochs, expected {config['num_epochs']}. Skipping run.")


        if agent_runs_rewards:
             last_run_avg = np.mean(agent_runs_rewards[-1])
             pbar_runs.set_postfix_str(f"Last run avg reward: {last_run_avg:.4f}")

    all_results[agent_type] = agent_runs_rewards
    end_time_agent = time.time()
    print(f"--- Finished {agent_type.upper()} runs in {end_time_agent - start_time_agent:.2f} seconds ---")


end_time_total = time.time()
print(f"\n--- All {NUM_RUNS} runs for all agents completed in {end_time_total - start_time_total:.2f} seconds ---")
print("Calculating statistics and generating plot...")

# --- Plotting with Plotly ---
fig = go.Figure()

colors_rgba = [
    'rgba(31, 119, 180, {alpha})',   # Muted Blue
    'rgba(255, 127, 14, {alpha})',   # Safety Orange
    'rgba(44, 160, 44, {alpha})',    # Cooked Asparagus Green
    'rgba(214, 39, 40, {alpha})',    # Brick Red
    'rgba(148, 103, 189, {alpha})',  # Muted Purple
]

plot_successful = False # Flag to check if any data was plotted

for i, agent_type in enumerate(AGENT_TYPES):
    agent_data = all_results[agent_type]
    config = AGENT_CONFIGS[agent_type]
    num_epochs = config['num_epochs']

    if not agent_data:
        print(f"No valid results found for agent type: {agent_type}. Skipping plot.")
        continue

    # Convert list of runs to a NumPy array for easier calculations
    # Shape: (num_valid_runs, num_epochs)
    results_array = np.array(agent_data)
    num_valid_runs = results_array.shape[0]

    if num_valid_runs == 0:
        print(f"No valid runs after filtering for agent type: {agent_type}. Skipping plot.")
        continue

    print(f"Plotting {agent_type.upper()} with {num_valid_runs} valid runs.")
    plot_successful = True

    # Calculate mean and standard deviation across runs for each epoch
    mean_rewards = np.mean(results_array, axis=0)
    std_rewards = np.std(results_array, axis=0)
    epochs_x = np.arange(num_epochs) # X-axis for this agent

    color_template = colors_rgba[i % len(colors_rgba)] # Cycle through defined colors

    # --- Add Standard Deviation Band ---
    # Upper bound trace (invisible line)
    fig.add_trace(go.Scatter(
        x=epochs_x,
        y=mean_rewards + std_rewards,
        mode='lines',
        line=dict(width=0), # Make line invisible
        showlegend=False,   # Don't show this trace in legend
        hoverinfo='none'    # No hover text for bounds
    ))
    # Lower bound trace (invisible line, fills to the upper bound)
    fig.add_trace(go.Scatter(
        x=epochs_x,
        y=mean_rewards - std_rewards,
        mode='lines',
        line=dict(width=0),
        fillcolor=color_template.format(alpha=0.2), # Fill color with transparency
        fill='tonexty', # Fill area between this trace and the previous one (upper bound)
        showlegend=False,
        hoverinfo='none',
        name=f'{agent_type.upper()} Std Dev' # Name for potential hover template later
    ))

    # --- Add Mean Reward Line ---
    # Added last so it appears on top of the shaded area
    fig.add_trace(go.Scatter(
        x=epochs_x,
        y=mean_rewards,
        mode='lines',
        name=f'{agent_type.upper()} Mean Reward',
        line=dict(color=color_template.format(alpha=1.0)), # Solid line color
        hovertemplate = f'<b>{agent_type.upper()}</b><br>Epoch: %{{x}}<br>Mean Reward: %{{y:.4f}}<extra></extra>' # Custom hover text
    ))


# --- Configure Plot Layout ---
fig.update_layout(
    title=f'Agent Performance Comparison (Mean Reward over {NUM_RUNS} runs)',
    xaxis_title='Epochs',
    yaxis_title='Average Discounted Test Reward per Epoch',
    hovermode="x unified", # Show hover info for all traces at a given x-coordinate
    legend_title_text='Agent',
    legend=dict(
        traceorder="normal", # Order legend items as they were added (Mean lines first conceptually)
        # Adjust legend position if needed:
        # yanchor="top", y=0.99, xanchor="left", x=0.01
    ),
    
    yaxis_range=[0, max(np.max(np.mean(np.array(all_results[agent]), axis=0) + np.std(np.array(all_results[agent]), axis=0)) for agent in AGENT_TYPES if all_results[agent]) * 1.1] # Auto-adjust slightly above max std dev
)

# --- Show Plot ---
if plot_successful:
    print("\nDisplaying interactive plot...")
    fig.show()
    fig.write_html("interactive_plot.html")
else:
    print("\nPlotting skipped as no valid results were generated for any agent.")