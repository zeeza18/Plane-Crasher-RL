# Import PyTorch for deep learning functionality
import torch
# Import NumPy for numerical operations and array manipulations
import numpy as np
# Import the custom FlappyBird environment
from environment.env import FlappyBirdEnv
# Import the Double DQN agent implementation
from models.ddqn import DoubleDQNAgent
# Import matplotlib for creating visualizations and plots
import matplotlib.pyplot as plt
# Import seaborn for enhanced visualization aesthetics
import seaborn as sns
# Import TensorBoard for tracking and visualizing metrics during training
from torch.utils.tensorboard import SummaryWriter
# Import OS module for file and directory operations
import os
# Import datetime for timestamping and time calculations
from datetime import datetime, timedelta
# Import time module for tracking training duration
import time
# Import tqdm for progress bar visualization
from tqdm import tqdm
# Import pandas for data manipulation and analysis
import pandas as pd
# Import GridSpec for creating complex plot layouts
from matplotlib.gridspec import GridSpec

# Function to create comprehensive visualization of training results
# This provides a detailed dashboard of the agent's learning progress
def create_advanced_plots(scores, avg_scores, epsilons, steps_per_episode, losses=None):
    """
    Create advanced plots that clearly show the learning progress and trends.
    Simplified to only include the comprehensive dashboard.
    
    Args:
        scores: List of episode scores (building-passed scores)
        avg_scores: List of average scores
        epsilons: List of epsilon values during training
        steps_per_episode: List of steps taken in each episode
        losses: Optional list of loss values
    """
    # Create directory for saving plots if it doesn't exist
    # Note: Using DDQN-specific directory to separate from DQN results
    os.makedirs('plots_ddqn', exist_ok=True)
    
    # Configure seaborn visualization style for better aesthetics
    sns.set(style="whitegrid")
    # Update matplotlib parameters for consistent and readable plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })
    
    # Create a pandas DataFrame from training metrics for easier manipulation
    data = pd.DataFrame({
        'Episode': np.arange(len(scores)),       # Episode numbers (0 to n-1)
        'Score': scores,                         # Buildings passed per episode
        'AvgScore': avg_scores,                  # Rolling average score (100 episodes)
        'Steps': steps_per_episode,              # Steps survived per episode
        'Epsilon': epsilons                      # Exploration rate per episode
    })
    
    # Add loss data if provided (handle None values by converting to NaN)
    if losses:
        valid_losses = [l if l is not None else np.nan for l in losses]
        data['Loss'] = valid_losses
    
    # Apply smoothing to raw metrics for better trend visualization
    # Window size adapts based on total episodes (larger dataset = larger window)
    window_size = min(25, len(scores) // 4) if len(scores) > 100 else 10
    # Calculate rolling means with specified window size
    data['SmoothScore'] = data['Score'].rolling(window=window_size, min_periods=1).mean()
    data['SmoothSteps'] = data['Steps'].rolling(window=window_size, min_periods=1).mean()
    
    # Calculate running maximum score (best score achieved up to each episode)
    data['CumMaxScore'] = data['Score'].cummax()  # Cumulative maximum
    
    # Create comprehensive dashboard with multiple plots
    # Figure with adequate size for detailed visualization
    fig = plt.figure(figsize=(16, 12))
    # Create grid specification for complex plot layout (3 rows, 2 columns)
    gs = GridSpec(3, 2, figure=fig)
    
    # TOP LEFT PLOT: Score progression with trends
    ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
    # Plot raw scores with low opacity
    ax1.plot(data['Episode'], data['Score'], alpha=0.4, color='royalblue', label='Buildings Passed')
    # Plot 100-episode moving average
    ax1.plot(data['Episode'], data['AvgScore'], color='navy', linewidth=2, label='100-Episode Average')
    # Plot smoothed trend line
    ax1.plot(data['Episode'], data['SmoothScore'], color='darkorange', linewidth=2, label=f'{window_size}-Episode Trend')
    # Plot best score achieved so far
    ax1.plot(data['Episode'], data['CumMaxScore'], color='crimson', linestyle='--', linewidth=1.5, label='Best Score')
    ax1.set_ylabel('Buildings Passed')  # Y-axis label
    # Title specifically mentions DDQN to differentiate from DQN plots
    ax1.set_title('Buildings Passed Over Training (DDQN)')  
    ax1.legend()  # Show legend
    
    # TOP RIGHT PLOT: Steps per episode progression
    ax2 = fig.add_subplot(gs[0, 1])  # First row, second column
    # Plot raw steps with low opacity
    ax2.plot(data['Episode'], data['Steps'], alpha=0.4, color='forestgreen', label='Steps per Episode')
    # Plot smoothed trend line
    ax2.plot(data['Episode'], data['SmoothSteps'], color='darkgreen', linewidth=2, label=f'{window_size}-Episode Trend')
    ax2.set_ylabel('Survival Steps')  # Y-axis label
    # Title specifically mentions DDQN
    ax2.set_title('Survival Time Progression (DDQN)')  
    ax2.legend()  # Show legend
    
    # MIDDLE LEFT PLOT: Score vs Epsilon relationship
    ax3 = fig.add_subplot(gs[1, 0])  # Second row, first column
    # Scatter plot with points colored by episode number
    scatter = ax3.scatter(data['Epsilon'], data['Score'], alpha=0.6, s=30, c=data['Episode'], cmap='viridis')
    ax3.set_xlabel('Epsilon (Exploration Rate)')  # X-axis label
    ax3.set_ylabel('Buildings Passed')  # Y-axis label
    # Title specifically mentions DDQN
    ax3.set_title('Buildings Passed vs Exploration Rate (DDQN)')  
    # Add color bar to show episode progression
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Episode')
    
    # MIDDLE RIGHT PLOT: Steps vs Score relationship
    ax4 = fig.add_subplot(gs[1, 1])  # Second row, second column
    # Scatter plot with points colored by episode number
    scatter2 = ax4.scatter(data['Steps'], data['Score'], alpha=0.6, s=30, c=data['Episode'], cmap='viridis')
    ax4.set_xlabel('Survival Steps')  # X-axis label
    ax4.set_ylabel('Buildings Passed')  # Y-axis label
    # Title specifically mentions DDQN
    ax4.set_title('Buildings Passed vs Survival Time (DDQN)')  
    
    # Add trend line to show correlation between steps and score
    z = np.polyfit(data['Steps'], data['Score'], 1)  # Linear fit
    p = np.poly1d(z)  # Create polynomial function from fit
    ax4.plot(sorted(data['Steps']), p(sorted(data['Steps'])), 
             color='red', linestyle='--', linewidth=2)  # Plot trend line
    
    # Add color bar to show episode progression
    cbar2 = plt.colorbar(scatter2, ax=ax4)
    cbar2.set_label('Episode')
    
    # BOTTOM PLOT: Epsilon decay and loss (if available)
    if losses:  # If loss data was provided
        # Create subplot spanning bottom row
        ax5 = fig.add_subplot(gs[2, :])
        # Create twin axis for plotting two y-scales
        ax5_twin = ax5.twinx()
        
        # Plot epsilon on primary y-axis
        ax5.plot(data['Episode'], data['Epsilon'], color='darkred', linewidth=2, label='Epsilon')
        ax5.set_xlabel('Episode')  # X-axis label
        ax5.set_ylabel('Epsilon', color='darkred')  # Y-axis label
        ax5.tick_params(axis='y', colors='darkred')  # Red tick marks for epsilon
        
        # Plot loss on secondary y-axis if available
        if 'Loss' in data.columns:
            # Drop rows with NaN loss values
            loss_data = data.dropna(subset=['Loss'])
            if len(loss_data) > 1:  # Ensure there's enough data to plot
                loss_series = loss_data['Loss']
                # Calculate appropriate smoothing window based on data size
                smooth_window = min(25, len(loss_data) // 4) if len(loss_data) > 100 else 10
                # Apply smoothing for trend visualization
                smoothed_loss = loss_series.rolling(window=smooth_window, min_periods=1).mean()
                
                # Plot raw loss values with low opacity
                ax5_twin.plot(loss_data['Episode'], loss_data['Loss'], alpha=0.3, color='purple', label='Loss')
                # Plot smoothed loss for trend visibility
                ax5_twin.plot(loss_data['Episode'], smoothed_loss, color='purple', linewidth=2, label=f'Smoothed Loss')
                ax5_twin.set_ylabel('Loss', color='purple')  # Y-axis label for loss
                ax5_twin.tick_params(axis='y', colors='purple')  # Purple tick marks for loss
                ax5_twin.legend(loc='upper right')  # Show legend for loss
        
        ax5.legend(loc='upper left')  # Show legend for epsilon
        # Title specifically mentions DDQN
        ax5.set_title('Exploration Rate Decay and Training Loss (DDQN)')  
    else:  # If no loss data was provided
        # Create subplot spanning bottom row
        ax5 = fig.add_subplot(gs[2, :])
        # Plot only epsilon decay
        ax5.plot(data['Episode'], data['Epsilon'], color='darkred', linewidth=2)
        ax5.set_xlabel('Episode')  # X-axis label
        ax5.set_ylabel('Epsilon')  # Y-axis label
        # Title specifically mentions DDQN
        ax5.set_title('Exploration Rate (Epsilon) Decay (DDQN)')  
    
    # Adjust layout for optimal spacing
    plt.tight_layout()
    # Save high-resolution figure to DDQN-specific file
    plt.savefig('plots_ddqn/training_dashboard.png', dpi=300)
    # Display figure
    plt.show()
    
    # Create a text summary of training results
    with open('plots_ddqn/training_summary.txt', 'w') as f:
        # Header specifically mentions DDQN
        f.write("======== FLAPPY BIRD DDQN TRAINING SUMMARY ========\n\n")
        
        # Write overall statistics
        f.write(f"Total Episodes: {len(scores)}\n")
        f.write(f"Best Score (Buildings Passed): {max(scores)}\n")
        f.write(f"Best 100-Episode Average: {max(avg_scores):.2f}\n")
        f.write(f"Final Exploration Rate (Epsilon): {epsilons[-1]:.4f}\n\n")
        
        # Write performance progression in chunks
        # Divide training into ~10 windows to show progression
        window_size = max(len(scores) // 10, 1)
        for i in range(0, len(scores), window_size):
            end_idx = min(i + window_size, len(scores))
            window_scores = scores[i:end_idx]
            window_steps = steps_per_episode[i:end_idx]
            
            # Write statistics for this window
            f.write(f"Episodes {i}-{end_idx-1}:\n")
            f.write(f"  Avg Buildings Passed: {np.mean(window_scores):.2f}\n")
            f.write(f"  Max Buildings Passed: {max(window_scores)}\n")
            f.write(f"  Avg Steps: {np.mean(window_steps):.2f}\n")
            f.write(f"  Max Steps: {max(window_steps)}\n\n")
        
        f.write("======== END OF SUMMARY ========\n")

# Main training function
def train():
    # Create directories for saving training artifacts
    # Note: Using DDQN-specific directories to separate from DQN results
    os.makedirs('checkpoints_ddqn', exist_ok=True)  # For model checkpoints
    os.makedirs('plots_ddqn', exist_ok=True)  # For plots
    
    # Generate timestamp for unique run identification
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Create log directory with timestamp and DDQN identifier
    log_dir = os.path.join('runs', f'flappy_bird_ddqn_{current_time}')
    # Initialize TensorBoard writer for logging metrics
    writer = SummaryWriter(log_dir)
    
    # Initialize the FlappyBird environment
    env = FlappyBirdEnv()
    # Get state and action dimensions from environment
    state_size = env.observation_space.shape[0]  # Input features (5)
    action_size = env.action_space.n  # Output actions (2: flap/don't flap)
    # Initialize Double DQN agent with appropriate dimensions
    agent = DoubleDQNAgent(state_size=state_size, action_size=action_size)
    
    # Define training hyperparameters
    n_episodes = 2500  # Total episodes to train
    max_steps = 2000   # Maximum steps per episode
    batch_size = 64    # Batch size for training updates
    
    # Initialize lists to track metrics
    scores = []         # Buildings passed per episode
    avg_scores = []     # Running average of buildings passed
    best_avg_score = -np.inf  # Best average score achieved
    
    # Additional metrics for enhanced analysis
    steps_per_episode = []  # Steps survived per episode
    epsilons = []           # Exploration rate per episode
    episode_losses = []     # Average loss per episode
    
    # Print message indicating start of DDQN training
    print("Starting DDQN training...")
    
    # Track timing information for ETA calculation
    start_time = time.time()  # Record training start time
    episode_times = []        # List to track duration of each episode
    
    try:
        # Main training loop
        for episode in range(1, n_episodes + 1):
            # Record episode start time
            episode_start = time.time()
            # Reset environment to start new episode
            state, _ = env.reset()
            # Initialize episode metrics
            reward_score = 0   # Cumulative reward (internal)
            game_score = 0     # Buildings passed (visible score)
            episode_loss = 0   # Cumulative loss
            n_steps = 0        # Step counter
            
            # Episode loop - run until done or max_steps reached
            for step in range(max_steps):
                # Select action based on current state
                action = agent.select_action(state)
                # Execute action in environment
                next_state, reward, done, _, info = env.step(action)
                
                # Track the actual game score (buildings passed)
                if 'score' in info:
                    game_score = info['score']
                
                # Store transition in replay buffer
                agent.memory.push(state, action, reward, next_state, done)
                
                # Perform training step if enough samples are available
                if len(agent.memory) > batch_size:
                    loss = agent.train_step()
                    # Accumulate loss (if available)
                    episode_loss += loss if loss is not None else 0
                
                # Update state and metrics
                state = next_state
                reward_score += reward
                n_steps += 1
                
                # Render the game (visual feedback)
                env.render()
                
                # Exit episode loop if done
                if done:
                    break
            
            # Update target network periodically
            if episode % agent.target_update == 0:
                agent.update_target_network()
            
            # Store metrics for this episode
            scores.append(game_score)  # Buildings passed
            steps_per_episode.append(n_steps)  # Steps survived
            epsilons.append(agent.epsilon)  # Exploration rate
            
            # Calculate average loss (if any training occurred)
            if n_steps > 0 and episode_loss > 0:
                episode_losses.append(episode_loss / n_steps)
            else:
                episode_losses.append(None)
                
            # Calculate average game score (last 100 episodes)
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            avg_scores.append(avg_score)
            
            # Calculate average steps
            avg_steps = np.mean(steps_per_episode[-100:]) if len(steps_per_episode) >= 100 else np.mean(steps_per_episode)
            
            # Log metrics to TensorBoard
            writer.add_scalar('Buildings Passed', game_score, episode)
            writer.add_scalar('Average Buildings Passed', avg_score, episode)
            writer.add_scalar('Reward', reward_score, episode)
            writer.add_scalar('Epsilon', agent.epsilon, episode)
            writer.add_scalar('Steps', n_steps, episode)
            # Only log loss if training occurred
            if episode_loss > 0:
                writer.add_scalar('Loss', episode_loss/n_steps, episode)
            
            # Save best model based on average game score
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save(f'checkpoints_ddqn/best_model.pth')
            
            # Track episode time for ETA calculation
            episode_end = time.time()
            episode_time = episode_end - episode_start
            episode_times.append(episode_time)
            
            # Calculate estimated time remaining
            # Use recent episodes for more accurate estimate
            if len(episode_times) > 10:
                recent_times = episode_times[-10:]
            else:
                recent_times = episode_times
            # Calculate average episode time
            avg_episode_time = sum(recent_times) / len(recent_times)
            # Calculate remaining episodes
            episodes_remaining = n_episodes - episode
            # Estimate seconds remaining
            eta_seconds = avg_episode_time * episodes_remaining
            # Format as hours:minutes:seconds
            eta = str(timedelta(seconds=int(eta_seconds)))
            
            # Print progress information
            print(f"Episode {episode}: Steps: {n_steps}, Avg Steps: {avg_steps:.1f}, Buildings Passed: {game_score}, Avg Buildings: {avg_score:.2f}, ε: {agent.epsilon:.4f}, Best: {best_avg_score:.2f} ({episode}/{n_episodes}), ETA: {eta}")
            
            # Save checkpoint periodically (every 100 episodes)
            if episode % 100 == 0:
                agent.save(f'checkpoints_ddqn/model_episode_{episode}.pth')
                
    # Handle user interruption (Ctrl+C)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    # Ensure cleanup regardless of how training ends
    finally:
        env.close()  # Close environment
        writer.close()  # Close TensorBoard writer
    
    # Calculate and print total training time
    total_training_time = time.time() - start_time
    training_hours = total_training_time // 3600
    training_minutes = (total_training_time % 3600) // 60
    training_seconds = total_training_time % 60
    
    print(f"\nDDQN Training completed in {int(training_hours)}h {int(training_minutes)}m {int(training_seconds)}s")
    print(f"Best average buildings passed: {best_avg_score:.2f}")
    
    # Save final model
    agent.save('checkpoints_ddqn/final_model.pth')
    
    # Generate performance analysis plots
    create_advanced_plots(scores, avg_scores, epsilons, steps_per_episode, episode_losses)
    
    # Return agent and metrics for further analysis
    return agent, scores, avg_scores

# Execute training if script is run directly
if __name__ == "__main__":
    agent, scores, avg_scores = train()