import torch
import numpy as np
from environment.env import FlappyBirdEnv
from models.dqn import DQNAgent
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import pandas as pd
from matplotlib.gridspec import GridSpec

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
    # Make sure plots directory exists
    os.makedirs('plots_dqn', exist_ok=True)
    
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
    
    # COMPREHENSIVE DASHBOARD
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Top left: Score progression with trends
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data['Episode'], data['Score'], alpha=0.4, color='royalblue', label='Buildings Passed')
    ax1.plot(data['Episode'], data['AvgScore'], color='navy', linewidth=2, label='100-Episode Average')
    ax1.plot(data['Episode'], data['SmoothScore'], color='darkorange', linewidth=2, label=f'{window_size}-Episode Trend')
    ax1.plot(data['Episode'], data['CumMaxScore'], color='crimson', linestyle='--', linewidth=1.5, label='Best Score')
    ax1.set_ylabel('Buildings Passed')
    ax1.set_title('Buildings Passed Over Training (DQN)')
    ax1.legend()
    
    # Top right: Steps progression with trends
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(data['Episode'], data['Steps'], alpha=0.4, color='forestgreen', label='Steps per Episode')
    ax2.plot(data['Episode'], data['SmoothSteps'], color='darkgreen', linewidth=2, label=f'{window_size}-Episode Trend')
    ax2.set_ylabel('Survival Steps')
    ax2.set_title('Survival Time Progression (DQN)')
    ax2.legend()
    
    # Middle left: Score vs Epsilon (How exploration affects performance)
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(data['Epsilon'], data['Score'], alpha=0.6, s=30, c=data['Episode'], cmap='viridis')
    ax3.set_xlabel('Epsilon (Exploration Rate)')
    ax3.set_ylabel('Buildings Passed')
    ax3.set_title('Buildings Passed vs Exploration Rate (DQN)')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Episode')
    
    # Middle right: Steps vs Score relationship
    ax4 = fig.add_subplot(gs[1, 1])
    scatter2 = ax4.scatter(data['Steps'], data['Score'], alpha=0.6, s=30, c=data['Episode'], cmap='viridis')
    ax4.set_xlabel('Survival Steps')
    ax4.set_ylabel('Buildings Passed')
    ax4.set_title('Buildings Passed vs Survival Time (DQN)')
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
        ax5.set_title('Exploration Rate Decay and Training Loss (DQN)')
    else:
        ax5 = fig.add_subplot(gs[2, :])
        ax5.plot(data['Episode'], data['Epsilon'], color='darkred', linewidth=2)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Epsilon')
        ax5.set_title('Exploration Rate (Epsilon) Decay (DQN)')
    
    plt.tight_layout()
    plt.savefig('plots_dqn/training_dashboard.png', dpi=300)
    plt.show()
    
    # Create a training summary text file
    with open('plots_dqn/training_summary.txt', 'w') as f:
        f.write("======== FLAPPY BIRD DQN TRAINING SUMMARY ========\n\n")
        
        # Overall stats
        f.write(f"Total Episodes: {len(scores)}\n")
        f.write(f"Best Score (Buildings Passed): {max(scores)}\n")
        f.write(f"Best 100-Episode Average: {max(avg_scores):.2f}\n")
        f.write(f"Final Exploration Rate (Epsilon): {epsilons[-1]:.4f}\n\n")
        
        # Performance progression
        window_size = max(len(scores) // 10, 1)  # Split into ~10 windows
        for i in range(0, len(scores), window_size):
            end_idx = min(i + window_size, len(scores))
            window_scores = scores[i:end_idx]
            window_steps = steps_per_episode[i:end_idx]
            
            f.write(f"Episodes {i}-{end_idx-1}:\n")
            f.write(f"  Avg Buildings Passed: {np.mean(window_scores):.2f}\n")
            f.write(f"  Max Buildings Passed: {max(window_scores)}\n")
            f.write(f"  Avg Steps: {np.mean(window_steps):.2f}\n")
            f.write(f"  Max Steps: {max(window_steps)}\n\n")
        
        f.write("======== END OF SUMMARY ========\n")

def train():
    # Create directories for checkpoints, logs, and plots
    os.makedirs('checkpoints_dqn', exist_ok=True)
    os.makedirs('plots_dqn', exist_ok=True)
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'flappy_bird_dqn_{current_time}')
    writer = SummaryWriter(log_dir)
    
    # Initialize environment and agent
    env = FlappyBirdEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Training parameters
    n_episodes = 2500
    max_steps = 2000
    batch_size = 64
    scores = []  # Will store actual game scores (buildings passed)
    avg_scores = []  # Average of game scores
    best_avg_score = -np.inf
    
    # Additional metrics for enhanced analysis
    steps_per_episode = []
    epsilons = []
    episode_losses = []
    
    print("Starting DQN training...")
    
    # Track total training time and estimate completion
    start_time = time.time()
    episode_times = []
    
    try:
        for episode in range(1, n_episodes + 1):
            episode_start = time.time()
            state, _ = env.reset()
            reward_score = 0  # Cumulative reward (for internal use)
            game_score = 0  # Buildings passed (the visible score)
            episode_loss = 0
            n_steps = 0
            
            # Episode loop
            for step in range(max_steps):
                action = agent.select_action(state)
                next_state, reward, done, _, info = env.step(action)
                
                # Track the actual game score (buildings passed)
                if 'score' in info:
                    game_score = info['score']
                
                agent.memory.push(state, action, reward, next_state, done)
                
                if len(agent.memory) > batch_size:
                    loss = agent.train_step()
                    episode_loss += loss if loss is not None else 0
                
                state = next_state
                reward_score += reward
                n_steps += 1
                
                # Render without interrupting
                env.render()
                
                if done:
                    break
            
            # Update target network
            if episode % agent.target_update == 0:
                agent.update_target_network()
            
            # Store metrics for this episode
            scores.append(game_score)  # Store the buildings passed
            steps_per_episode.append(n_steps)
            epsilons.append(agent.epsilon)
            
            if n_steps > 0 and episode_loss > 0:
                episode_losses.append(episode_loss / n_steps)
            else:
                episode_losses.append(None)
                
            # Calculate average game score (last 100 episodes)
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            avg_scores.append(avg_score)
            
            # Calculate average steps
            avg_steps = np.mean(steps_per_episode[-100:]) if len(steps_per_episode) >= 100 else np.mean(steps_per_episode)
            
            # Log to tensorboard
            writer.add_scalar('Buildings Passed', game_score, episode)
            writer.add_scalar('Average Buildings Passed', avg_score, episode)
            writer.add_scalar('Reward', reward_score, episode)
            writer.add_scalar('Epsilon', agent.epsilon, episode)
            writer.add_scalar('Steps', n_steps, episode)
            if episode_loss > 0:
                writer.add_scalar('Loss', episode_loss/n_steps, episode)
            
            # Save best model based on game score
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save(f'checkpoints_dqn/best_model.pth')
            
            # Track episode time for ETA calculation
            episode_end = time.time()
            episode_time = episode_end - episode_start
            episode_times.append(episode_time)
            
            # Calculate ETA (using the average of recent episodes)
            if len(episode_times) > 10:
                recent_times = episode_times[-10:]
            else:
                recent_times = episode_times
            avg_episode_time = sum(recent_times) / len(recent_times)
            episodes_remaining = n_episodes - episode
            eta_seconds = avg_episode_time * episodes_remaining
            eta = str(timedelta(seconds=int(eta_seconds)))
            
            # Print the episode information with game score
            print(f"Episode {episode}: Steps: {n_steps}, Avg Steps: {avg_steps:.1f}, Buildings Passed: {game_score}, Avg Buildings: {avg_score:.2f}, ε: {agent.epsilon:.4f}, Best: {best_avg_score:.2f} ({episode}/{n_episodes}), ETA: {eta}")
            
            # Save checkpoint every 100 episodes
            if episode % 100 == 0:
                agent.save(f'checkpoints_dqn/model_episode_{episode}.pth')
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        env.close()
        writer.close()
    
    # Calculate total training time
    total_training_time = time.time() - start_time
    training_hours = total_training_time // 3600
    training_minutes = (total_training_time % 3600) // 60
    training_seconds = total_training_time % 60
    
    print(f"\nDQN Training completed in {int(training_hours)}h {int(training_minutes)}m {int(training_seconds)}s")
    print(f"Best average buildings passed: {best_avg_score:.2f}")
    
    # Final save
    agent.save('checkpoints_dqn/final_model.pth')
    
    # Create enhanced analysis plots with game scores (buildings passed)
    create_advanced_plots(scores, avg_scores, epsilons, steps_per_episode, episode_losses)
    
    return agent, scores, avg_scores

if __name__ == "__main__":
    agent, scores, avg_scores = train()