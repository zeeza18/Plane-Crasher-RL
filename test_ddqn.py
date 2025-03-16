# Import PyTorch for deep learning functionality
import torch
# Import NumPy for numerical operations
import numpy as np
# Import time module for sleep and timing functionality
import time
# Import argparse for command-line argument parsing
import argparse
# Import the FlappyBird environment
from environment.env import FlappyBirdEnv
# Import the Double DQN agent implementation
from models.ddqn import DoubleDQNAgent
# Import pygame for handling events during rendering
import pygame

def test_agent(model_path, num_episodes=5, render=True, fps=30, verbose=True):
    """
    Test a trained Double DQN agent on the FlappyBird environment.
    
    Args:
        model_path: Path to the saved model checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render the game
        fps: Frames per second for visualization (lower for slower gameplay)
        verbose: Whether to print episode information
    """
    # Initialize environment and agent
    env = FlappyBirdEnv()
    # Get state and action dimensions from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # Initialize Double DQN agent with appropriate dimensions
    agent = DoubleDQNAgent(state_size=state_size, action_size=action_size)
    
    # Load the trained model from specified path
    print(f"Loading Double DQN model from {model_path}")
    agent.load(model_path)
    
    # Set epsilon to a small value for minimal exploration during testing
    # This allows occasional random actions to demonstrate robustness
    agent.epsilon = 0.01
    
    # Initialize lists to track performance metrics
    scores = []        # Reward scores
    steps_list = []    # Steps per episode
    game_scores = []   # Buildings passed per episode
    
    try:
        # Run the specified number of test episodes
        for episode in range(1, num_episodes + 1):
            # Reset environment to start new episode
            state, _ = env.reset()
            # Initialize episode metrics
            score = 0
            steps = 0
            done = False
            
            # Episode loop - run until game over
            while not done:
                # Select action with minimal exploration (training=False)
                # Using deterministic policy for consistent performance assessment
                action = agent.select_action(state, training=False)
                
                # Execute action in environment
                next_state, reward, done, _, info = env.step(action)
                
                # Update state and metrics
                state = next_state
                score += reward
                steps += 1
                
                # Render game if enabled (for visual feedback)
                if render:
                    env.render()
                    # Control playback speed with FPS setting
                    # Lower FPS = slower gameplay for easier viewing
                    time.sleep(1/fps)
                
                # Handle game over scenario with special rendering
                if done and render:
                    # Play crash sound effect when game ends
                    # Only attempt if sound manager exists and is loaded
                    if hasattr(env, 'sound_manager') and env.sound_manager.loaded:
                        env.sound_manager.play_crash_sound_as_background()
                    
                    # Show game over animation for 2 seconds
                    # This gives time to see the crash and game over screen
                    animation_start = time.time()
                    animation_duration = 2.0
                    
                    # Animation loop for game over sequence
                    while time.time() - animation_start < animation_duration:
                        # Process pygame events to keep window responsive
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                # Allow user to exit during animation
                                pygame.quit()
                                return scores, steps_list
                        
                        # Continue rendering to show animation progress
                        env.render()
                        time.sleep(1/fps)
                    
                    # Background music will resume in next reset automatically
            
            # Record metrics for this episode
            scores.append(score)           # Total reward
            steps_list.append(steps)       # Steps survived
            # Get actual game score (buildings passed), defaulting to 0 if not in info
            game_score = info.get('score', 0)
            game_scores.append(game_score)
            
            # Print episode stats if verbose mode is enabled
            if verbose:
                print(f"Episode {episode}/{num_episodes} - Game Score: {game_score} - Steps: {steps}")
        
        # Calculate and print overall performance statistics
        avg_score = np.mean(scores)       # Average reward
        avg_steps = np.mean(steps_list)   # Average survival time
        max_game_score = max(game_scores) if game_scores else 0  # Best performance
        
        # Display summary of test results
        print("\n===== Double DQN Test Results =====")
        print(f"Episodes: {num_episodes}")
        print(f"Average Reward: {avg_score:.2f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Best Game Score: {max_game_score}")
        print(f"Model used: {model_path}")
        print("=================================")
        
    # Handle user interruption gracefully
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    # Ensure environment is closed properly regardless of outcome
    finally:
        env.close()
    
    # Return performance metrics for potential further analysis
    return scores, steps_list

# Execute test if script is run directly (not imported)
if __name__ == "__main__":
    # Set up command-line argument parser with descriptive help
    parser = argparse.ArgumentParser(description='Test a trained Double DQN agent on FlappyBird')
    
    # Add various command-line options with defaults
    parser.add_argument('--model', type=str, default='checkpoints_ddqn\model_episode_2300.pth',
                        help='Path to the model checkpoint (default: checkpoints_ddqn\model_episode_2300.pth)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for rendering (default: 30)')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable verbose output')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call test_agent with the parsed arguments
    test_agent(
        model_path=args.model,           # Path to trained model
        num_episodes=args.episodes,      # Number of test episodes
        render=not args.no_render,       # Whether to show visualization
        fps=args.fps,                    # Playback speed
        verbose=not args.quiet           # Whether to print detailed logs
    )