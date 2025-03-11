from environment.env import FlappyBirdEnv
import time

def test_environment():
    # Create environment
    env = FlappyBirdEnv()
    
    # Run 5 episodes
    for episode in range(5):
        observation, _ = env.reset()
        episode_reward = 0
        steps = 0
        terminated = False
        
        print(f"\nEpisode {episode + 1}")
        
        while not terminated:
            # Random action
            action = env.action_space.sample()
            
            # Take step
            observation, reward, terminated, _, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Render
            env.render()
            time.sleep(0.03)  # Add small delay to make it viewable
            
            # Print current state
            if steps % 20 == 0:  # Print every 20 steps
                print(f"Step {steps}, Score: {info['score']}, Reward: {episode_reward:.2f}")
        
        print(f"Episode {episode + 1} finished after {steps} steps with reward {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    test_environment()