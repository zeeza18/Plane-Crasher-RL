import torch
import numpy as np
from environment.env import FlappyBirdEnv
from models.dqn import DQNAgent
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import time
from tqdm import tqdm
import pandas as pd
from matplotlib.gridspec import GridSpec

def create_advanced_plots(scores, avg_scores, epsilons, steps_per_episode, losses=None):
    """
    Create advanced plots that clearly show the learning progress and trends.
    
    Args:
        scores: List of episode scores
        avg_scores: List of average scores
        epsilons: List of epsilon values during training
        steps_per_episode: List of steps taken in each episode
        losses: Optional list of loss values
    """
    # Make sure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })
    
    # Create a DataFrame for easier manipulation
    data = pd.DataFrame({
        'Episode': np.arange(len(scores)),
        'Score': scores,
        'AvgScore': avg_scores,
        'Steps': steps_per_episode,
        'Epsilon': epsilons
    })
    
    if losses:
        valid_losses = [l if l is not None else np.nan for l in losses]
        data['Loss'] = valid_losses
    
    # Apply smoothing to better show trends
    window_size = min(25, len(scores) // 4) if len(scores) > 100 else 10
    data['SmoothScore'] = data['Score'].rolling(window=window_size, min_periods=1).mean()
    data['SmoothSteps'] = data['Steps'].rolling(window=window_size, min_periods=1).mean()
    
    # Calculate additional metrics
    data['CumMaxScore'] = data['Score'].cummax()  # Best score so far
    
    # PLOT 1: COMBINED LEARNING PROGRESS
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # Top left: Score progression with trends
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data['Episode'], data['Score'], alpha=0.4, color='royalblue', label='Episode Score')
    ax1.plot(data['Episode'], data['AvgScore'], color='navy', linewidth=2, label='100-Episode Average')
    ax1.plot(data['Episode'], data['SmoothScore'], color='darkorange', linewidth=2, label=f'{window_size}-Episode Trend')
    ax1.plot(data['Episode'], data['CumMaxScore'], color='crimson', linestyle='--', linewidth=1.5, label='Best Score')
    ax1.set_ylabel('Score')
    ax1.set_title('Score Progression with Trends')
    ax1.legend()
    
    # Top right: Steps progression with trends
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(data['Episode'], data['Steps'], alpha=0.4, color='forestgreen', label='Steps per Episode')
    ax2.plot(data['Episode'], data['SmoothSteps'], color='darkgreen', linewidth=2, label=f'{window_size}-Episode Trend')
    ax2.set_ylabel('Survival Steps')
    ax2.set_title('Survival Time Progression')
    ax2.legend()
    
    # Middle left: Score vs Epsilon (How exploration affects performance)
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(data['Epsilon'], data['Score'], alpha=0.6, s=30, c=data['Episode'], cmap='viridis')
    ax3.set_xlabel('Epsilon (Exploration Rate)')
    ax3.set_ylabel('Score')
    ax3.set_title('Score vs Exploration Rate')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Episode')
    
    # Middle right: Steps vs Score relationship
    ax4 = fig.add_subplot(gs[1, 1])
    scatter2 = ax4.scatter(data['Steps'], data['Score'], alpha=0.6, s=30, c=data['Episode'], cmap='viridis')
    ax4.set_xlabel('Survival Steps')
    ax4.set_ylabel('Score')
    ax4.set_title('Score vs Survival Time')
    # Add a best fit line to show correlation
    z = np.polyfit(data['Steps'], data['Score'], 1)
    p = np.poly1d(z)
    ax4.plot(sorted(data['Steps']), p(sorted(data['Steps'])), 
             color='red', linestyle='--', linewidth=2)
    cbar2 = plt.colorbar(scatter2, ax=ax4)
    cbar2.set_label('Episode')
    
    # Bottom: Epsilon decay and loss if available
    if losses:
        ax5 = fig.add_subplot(gs[2, :])
        ax5_twin = ax5.twinx()
        
        # Plot epsilon on primary y-axis
        ax5.plot(data['Episode'], data['Epsilon'], color='darkred', linewidth=2, label='Epsilon')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Epsilon', color='darkred')
        ax5.tick_params(axis='y', colors='darkred')
        
        # Plot smoothed loss on secondary y-axis
        if 'Loss' in data.columns:
            loss_data = data.dropna(subset=['Loss'])
            if len(loss_data) > 1:
                loss_series = loss_data['Loss']
                smooth_window = min(25, len(loss_data) // 4) if len(loss_data) > 100 else 10
                smoothed_loss = loss_series.rolling(window=smooth_window, min_periods=1).mean()
                
                ax5_twin.plot(loss_data['Episode'], loss_data['Loss'], alpha=0.3, color='purple', label='Loss')
                ax5_twin.plot(loss_data['Episode'], smoothed_loss, color='purple', linewidth=2, label=f'Smoothed Loss')
                ax5_twin.set_ylabel('Loss', color='purple')
                ax5_twin.tick_params(axis='y', colors='purple')
                ax5_twin.legend(loc='upper right')
        
        ax5.legend(loc='upper left')
        ax5.set_title('Exploration Rate Decay and Training Loss')
    else:
        ax5 = fig.add_subplot(gs[2, :])
        ax5.plot(data['Episode'], data['Epsilon'], color='darkred', linewidth=2)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Epsilon')
        ax5.set_title('Exploration Rate (Epsilon) Decay')
    
    plt.tight_layout()
    plt.savefig('plots/advanced_training_analysis.png', dpi=300)
    plt.show()
    
    # PLOT 2: LEARNING PROGRESS OVER TIME (FACETS)
    plt.figure(figsize=(12, 6))
    plt.plot(data['Episode'], data['Score'], alpha=0.5, color='cornflowerblue', label='Episode Score')
    plt.plot(data['Episode'], data['AvgScore'], color='navy', linewidth=2, label='100-Episode Average')
    plt.plot(data['Episode'], data['SmoothScore'], color='darkorange', linewidth=2, label=f'{window_size}-Episode Trend')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Score Progression Over Training', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/score_progression.png', dpi=300)
    plt.show()
    
    # PLOT 3: HEATMAP SHOWING PERFORMANCE PATTERNS
    # Create a binned version of the data
    bin_size = max(1, len(scores) // 50)  # Try to get around 50 bins
    data['EpisodeBin'] = data['Episode'] // bin_size
    
    # Group by bin and calculate statistics
    binned_data = data.groupby('EpisodeBin').agg({
        'Score': ['mean', 'max'],
        'Steps': ['mean', 'max'],
        'Epsilon': 'mean'
    })
    
    # Flatten the column names
    binned_data.columns = ['_'.join(col).strip() for col in binned_data.columns.values]
    
    # Reset index to get EpisodeBin as a column
    binned_data = binned_data.reset_index()
    
    # Create a pivot table for the heatmap
    if len(binned_data) > 4:  # Only create heatmap if we have enough bins
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a pivot table
        heatmap_data = binned_data.melt(id_vars='EpisodeBin', 
                                        value_vars=['Score_mean', 'Score_max', 'Steps_mean', 'Steps_max'])
        
        # Map episode bins to actual episode numbers (middle of bin)
        binned_data['EpisodeNum'] = binned_data['EpisodeBin'] * bin_size + bin_size // 2
        heatmap_data = pd.merge(heatmap_data, binned_data[['EpisodeBin', 'EpisodeNum']], on='EpisodeBin')
        
        # Create the heatmap
        pivot_data = heatmap_data.pivot(index='variable', columns='EpisodeNum', values='value')
        
        # Normalize each row
        normalized_data = pivot_data.copy()
        for idx in normalized_data.index:
            row_min = normalized_data.loc[idx].min()
            row_max = normalized_data.loc[idx].max()
            if row_max > row_min:  # Avoid division by zero
                normalized_data.loc[idx] = (normalized_data.loc[idx] - row_min) / (row_max - row_min)
        
        # Plot the heatmap
        sns.heatmap(normalized_data, cmap='viridis', ax=ax, cbar_kws={'label': 'Normalized Value'})
        ax.set_title('Training Progress Heatmap (Normalized)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Metric')
        
        # Adjust y-tick labels
        y_labels = {'Score_mean': 'Avg Score', 'Score_max': 'Max Score', 
                   'Steps_mean': 'Avg Steps', 'Steps_max': 'Max Steps'}
        ax.set_yticklabels([y_labels.get(idx, idx) for idx in normalized_data.index])
        
        plt.tight_layout()
        plt.savefig('plots/training_heatmap.png', dpi=300)
        plt.show()
    
    # Create a training summary text file
    with open('plots/training_summary.txt', 'w') as f:
        f.write("======== FLAPPY BIRD DQN TRAINING SUMMARY ========\n\n")
        
        # Overall stats
        f.write(f"Total Episodes: {len(scores)}\n")
        f.write(f"Best Score: {max(scores):.2f}\n")
        f.write(f"Best 100-Episode Average: {max(avg_scores):.2f}\n")
        f.write(f"Final Exploration Rate (Epsilon): {epsilons[-1]:.4f}\n\n")
        
        # Performance progression
        window_size = max(len(scores) // 10, 1)  # Split into ~10 windows
        for i in range(0, len(scores), window_size):
            end_idx = min(i + window_size, len(scores))
            window_scores = scores[i:end_idx]
            window_steps = steps_per_episode[i:end_idx]
            
            f.write(f"Episodes {i}-{end_idx-1}:\n")
            f.write(f"  Avg Score: {np.mean(window_scores):.2f}\n")
            f.write(f"  Max Score: {max(window_scores):.2f}\n")
            f.write(f"  Avg Steps: {np.mean(window_steps):.2f}\n")
            f.write(f"  Max Steps: {max(window_steps)}\n\n")
        
        f.write("======== END OF SUMMARY ========\n")

def plot_scores(scores, avg_scores):
    """Original simple plotting function (kept for compatibility)"""
    plt.figure(figsize=(10,5))
    plt.plot(scores, label='Score')
    plt.plot(avg_scores, label='Average Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('plots/simple_score_plot.png', dpi=300)
    plt.show()

def train():
    # Create directories for checkpoints, logs, and plots
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'flappy_bird_{current_time}')
    writer = SummaryWriter(log_dir)
    
    # Initialize environment and agent
    env = FlappyBirdEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Training parameters
    n_episodes = 5000
    max_steps = 2000
    batch_size = 64
    scores = []
    avg_scores = []
    best_avg_score = -np.inf
    
    # Additional metrics for enhanced analysis
    steps_per_episode = []
    epsilons = []
    episode_losses = []
    
    print("Starting training...")
    # Use tqdm for progress bar
    progress_bar = tqdm(range(n_episodes), desc="Training")
    
    # Track total training time
    start_time = time.time()
    
    try:
        for episode in progress_bar:
            state, _ = env.reset()
            score = 0
            episode_loss = 0
            n_steps = 0
            
            # Episode loop
            for step in range(max_steps):
                action = agent.select_action(state)
                next_state, reward, done, _, info = env.step(action)
                
                agent.memory.push(state, action, reward, next_state, done)
                
                if len(agent.memory) > batch_size:
                    loss = agent.train_step()
                    episode_loss += loss if loss is not None else 0
                
                state = next_state
                score += reward
                n_steps += 1
                
                # Render without interrupting
                env.render()
                
                if done:
                    break
            
            # Update target network
            if episode % agent.target_update == 0:
                agent.update_target_network()
            
            # Store metrics for this episode
            scores.append(score)
            steps_per_episode.append(n_steps)
            epsilons.append(agent.epsilon)
            
            if n_steps > 0 and episode_loss > 0:
                episode_losses.append(episode_loss / n_steps)
            else:
                episode_losses.append(None)
                
            # Calculate average score (last 100 episodes)
            avg_score = np.mean(scores[-100:])  # Moving average of last 100 episodes
            avg_scores.append(avg_score)
            
            # Log to tensorboard
            writer.add_scalar('Score', score, episode)
            writer.add_scalar('Average Score', avg_score, episode)
            writer.add_scalar('Epsilon', agent.epsilon, episode)
            writer.add_scalar('Steps', n_steps, episode)
            if episode_loss > 0:
                writer.add_scalar('Loss', episode_loss/n_steps, episode)
            
            # Save best model
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save(f'checkpoints/best_model.pth')
            
            # Update progress bar description
            if episode % 10 == 0:
                progress_bar.set_description(
                    f"Score: {score:.2f} Avg: {avg_score:.2f} Best: {best_avg_score:.2f} ε: {agent.epsilon:.2f}"
                )
            
            # Save checkpoint every 100 episodes
            if episode % 100 == 0:
                agent.save(f'checkpoints/model_episode_{episode}.pth')
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        env.close()
        writer.close()
    
    # Calculate total training time
    total_training_time = time.time() - start_time
    
    # Final save
    agent.save('checkpoints/final_model.pth')
    
    # Create enhanced analysis plots
    create_advanced_plots(scores, avg_scores, epsilons, steps_per_episode, episode_losses)
    
    # Original plotting function (kept for compatibility)
    plot_scores(scores, avg_scores)
    
    return agent, scores, avg_scores

if __name__ == "__main__":
    agent, scores, avg_scores = train()