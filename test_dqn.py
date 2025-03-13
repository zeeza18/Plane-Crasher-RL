import torch
import numpy as np
import time
import argparse
from environment.env import FlappyBirdEnv
from models.dqn import DQNAgent
import pygame

def test_agent(model_path, num_episodes=10, render=True, fps=30, verbose=True):
    """
    Test a trained DQN agent on the FlappyBird environment.
    
    Args:
        model_path: Path to the saved model checkpoint
        num_episodes: Number of episodes to run
        render: Whether to render the game
        fps: Frames per second for visualization (lower for slower gameplay)
        verbose: Whether to print episode information
    """
    # Initialize environment and agent
    env = FlappyBirdEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Load the trained model
    print(f"Loading model from {model_path}")
    agent.load(model_path)
    
    # Set epsilon to a small value for minimal exploration during testing
    agent.epsilon = 0.01
    
    # Track statistics
    scores = []
    steps_list = []
    game_scores = []  # Track actual game scores
    
    try:
        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            score = 0
            steps = 0
            done = False
            
            # Episode loop
            while not done:
                # Select action without exploration (training=False)
                action = agent.select_action(state, training=False)
                
                # Take the action
                next_state, reward, done, _, info = env.step(action)
                
                # Update state and score
                state = next_state
                score += reward
                steps += 1
                
                # Render if enabled
                if render:
                    env.render()
                    time.sleep(1/fps)
                
                # If game is over
                if done and render:
                    # Play crash sound as background for the game over screen
                    if hasattr(env, 'sound_manager') and env.sound_manager.loaded:
                        env.sound_manager.play_crash_sound_as_background()
                    
                    # Show game over animation for 2 seconds
                    animation_start = time.time()
                    animation_duration = 2.0
                    
                    while time.time() - animation_start < animation_duration:
                        # Process events
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                return scores, steps_list
                        
                        # Continue rendering to show animation progress
                        env.render()
                        time.sleep(1/fps)
                    
                    # Resume background music after game over (will happen naturally in reset)
            
            # Record this episode
            scores.append(score)
            steps_list.append(steps)
            game_score = info.get('score', 0)  # Safely get the game score
            game_scores.append(game_score)
            
            # Print episode stats
            if verbose:
                print(f"Episode {episode}/{num_episodes} - Game Score: {game_score} - Steps: {steps}")
        
        # Print overall stats
        avg_score = np.mean(scores)
        avg_steps = np.mean(steps_list)
        max_game_score = max(game_scores) if game_scores else 0
        
        print("\n===== Test Results =====")
        print(f"Episodes: {num_episodes}")
        print(f"Average Reward: {avg_score:.2f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Best Game Score: {max_game_score}")
        print(f"Model used: {model_path}")
        print("=======================")
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    finally:
        env.close()
    
    return scores, steps_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a trained DQN agent on FlappyBird')
    parser.add_argument('--model', type=str, default='checkpoints_dqn/model_episode_2200.pth',
                        help='Path to the model checkpoint (default: checkpoints_dqn/model_episode_2200.pth)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for rendering (default: 30)')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable verbose output')
    
    args = parser.parse_args()
    
    test_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
        fps=args.fps,
        verbose=not args.quiet
    )