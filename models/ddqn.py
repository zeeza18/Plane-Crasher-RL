# Import PyTorch library for deep learning functionality
import torch
# Import neural network modules from PyTorch
import torch.nn as nn
# Import functional interface for operations like activation functions
import torch.nn.functional as F
# Import NumPy for numerical operations and array handling
import numpy as np
# Import deque (double-ended queue) for efficient replay buffer implementation
from collections import deque
# Import random module for exploration and sampling from replay buffer
import random

# Define the Deep Q-Network neural network architecture
# This class inherits from nn.Module, which is the base class for all neural networks in PyTorch
class DQN(nn.Module):
    # Constructor method
    # Input parameters:
    #   input_size: dimension of state space (number of features in state)
    #   hidden_size: number of neurons in hidden layers
    #   output_size: dimension of action space (number of possible actions)
    def __init__(self, input_size, hidden_size, output_size):
        # Call the parent class constructor to initialize base functionality
        super(DQN, self).__init__()
        # Create first fully connected layer: state features -> hidden neurons
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Create second fully connected layer: hidden -> hidden (deeper representation)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Create output layer: hidden -> action values (Q-values for each action)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    # Forward pass method - defines how input data flows through the network
    # Input parameters:
    #   x: input tensor representing the state
    # Returns: predicted Q-values for each possible action
    def forward(self, x):
        # Apply first linear layer followed by ReLU activation function
        # ReLU function: f(x) = max(0,x) - keeps positive values unchanged, sets negatives to 0
        x = F.relu(self.fc1(x))
        # Apply second linear layer followed by ReLU activation
        x = F.relu(self.fc2(x))
        # Apply final linear layer without activation (raw Q-values)
        return self.fc3(x)

# Replay Buffer for experience replay - stores and samples transitions
# Experience replay breaks correlation between consecutive samples and improves learning stability
class ReplayBuffer:
    # Constructor method
    # Input parameters:
    #   capacity: maximum number of transitions to store (oldest get replaced when full)
    def __init__(self, capacity):
        # Initialize deque with maximum length = capacity
        # When buffer exceeds capacity, oldest elements are automatically removed
        self.buffer = deque(maxlen=capacity)
    
    # Add a new transition to the buffer
    # Input parameters:
    #   state: current state observation
    #   action: action taken
    #   reward: reward received
    #   next_state: resulting state after action
    #   done: boolean indicating if episode terminated
    def push(self, state, action, reward, next_state, done):
        # Add tuple of (state, action, reward, next_state, done) to buffer
        self.buffer.append((state, action, reward, next_state, done))
    
    # Sample a random batch of transitions for training
    # Input parameters:
    #   batch_size: number of transitions to sample
    # Returns: batch of transitions as PyTorch tensors
    def sample(self, batch_size):
        # Randomly sample 'batch_size' transitions from buffer
        batch = random.sample(self.buffer, batch_size)
        
        # Extract and organize each component across all sampled transitions
        # Convert lists to numpy arrays for efficient batch processing
        state = np.array([x[0] for x in batch])
        action = np.array([x[1] for x in batch])
        reward = np.array([x[2] for x in batch])
        next_state = np.array([x[3] for x in batch])
        done = np.array([x[4] for x in batch])
        
        # Convert numpy arrays to PyTorch tensors and return
        # Each tensor has shape [batch_size, component_dimension]
        return (torch.FloatTensor(state),           # States as float tensor
                torch.LongTensor(action),           # Actions as long (integer) tensor
                torch.FloatTensor(reward),          # Rewards as float tensor
                torch.FloatTensor(next_state),      # Next states as float tensor
                torch.FloatTensor(done))            # Done flags as float tensor
    
    # Return current size of buffer
    # Returns: number of transitions currently stored
    def __len__(self):
        return len(self.buffer)

# Double DQN Agent that makes decisions and learns from experience
# Double DQN reduces overestimation of Q-values by decoupling action selection and evaluation
class DoubleDQNAgent:
    # Constructor method
    # Input parameters:
    #   state_size: dimension of state space
    #   action_size: dimension of action space
    #   hidden_size: neurons in hidden layers (default 128)
    #   lr: learning rate for optimizer (default 0.0001)
    def __init__(self, state_size, action_size, hidden_size=128, lr=1e-4):
        # Set device to GPU if available, otherwise CPU
        # This accelerates training when a compatible GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize neural networks
        # Policy network: makes action decisions and gets updated directly
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        # Target network: provides stable Q-value targets (updated less frequently)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        # Initialize target network with same weights as policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Setup training components
        # Adam optimizer for policy network with specified learning rate
        # Adam adapts learning rates for each parameter based on past gradients
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        # Initialize replay buffer with capacity of 100,000 transitions
        self.memory = ReplayBuffer(100000)
        
        # Set hyperparameters
        self.batch_size = 64         # Number of samples per training batch
        self.gamma = 0.99            # Discount factor for future rewards (0.99 = 99% of future value retained)
        self.epsilon = 1.0           # Initial exploration rate (100% random actions)
        self.epsilon_min = 0.01      # Minimum exploration rate (1% random actions)
        self.epsilon_decay = 0.9995  # Decay multiplier for exploration rate
        self.epsilon_decay_start = 1000  # Start decaying epsilon after this many samples
        self.target_update = 10      # Update target network every N episodes
        
    # Select an action given a state
    # Input parameters:
    #   state: current state observation
    #   training: boolean flag indicating if in training mode (affects exploration)
    # Returns: selected action index
    def select_action(self, state, training=True):
        # Exploration (epsilon-greedy): take random action with probability epsilon
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)  # Random action (0 or 1 for Flappy Bird)
        
        # Exploitation: use network to choose best action
        with torch.no_grad():  # No need to track gradients for action selection (saves memory)
            # Convert state to tensor and add batch dimension
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Get Q-values for all actions from policy network
            q_values = self.policy_net(state)
            # Return action with highest Q-value
            # max(1) gets maximum along action dimension
            # [1] gets the indices, .item() converts single-item tensor to Python number
            return q_values.max(1)[1].item()
    
    # Perform one training step (batch update)
    # No input parameters
    # Returns: loss value (or None if not enough samples)
    def train_step(self):
        # Skip if not enough samples in replay buffer
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of transitions from replay buffer
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # Move all tensors to the selected device (GPU/CPU)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # Compute current Q-values: Q(s_t, a)
        # policy_net(state) gives Q-values for all actions in each state
        # gather(1, action.unsqueeze(1)) selects Q-values for the actions actually taken
        current_q = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        # Double DQN: Use policy_net to SELECT action and target_net to EVALUATE action
        # This reduces overestimation bias in Q-learning
        with torch.no_grad():  # Don't track gradients for target computation
            # Select action using policy network (newer, more up-to-date network)
            next_action = self.policy_net(next_state).max(1)[1].unsqueeze(1)
            
            # Evaluate action using target network (more stable, less frequently updated)
            # This decoupling of selection and evaluation reduces overestimation
            next_q = self.target_net(next_state).gather(1, next_action).squeeze(1)
            
            # Compute target Q value using Bellman equation
            # For terminal states (done=1), target is just the reward
            # For non-terminal states, target is reward + discounted future value
            target_q = reward + (1 - done) * self.gamma * next_q
        
        # Compute loss between current Q-values and target Q-values
        # Huber loss (smooth_l1_loss) is more robust to outliers than MSE
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        # Gradient descent step
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()             # Compute gradients
        # Clip gradients to prevent explosion (numerical stability)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()       # Update weights
        
        # Update exploration rate (epsilon) after initial collection period
        if len(self.memory) > self.epsilon_decay_start:
            # Decay epsilon but don't go below minimum value
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Return loss value for monitoring
        return loss.item()
    
    # Update target network with current policy network weights
    # No input parameters
    # No return value
    def update_target_network(self):
        # Copy parameters from policy network to target network
        # This stabilizes training by keeping target values consistent
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    # Save agent state to disk
    # Input parameters:
    #   path: file path to save to
    # No return value
    def save(self, path):
        # Save networks, optimizer state, and current exploration rate
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),    # Policy network weights
            'target_net_state_dict': self.target_net.state_dict(),    # Target network weights
            'optimizer_state_dict': self.optimizer.state_dict(),      # Optimizer state
            'epsilon': self.epsilon                                   # Current exploration rate
        }, path)
    
    # Load agent state from disk
    # Input parameters:
    #   path: file path to load from
    # No return value
    def load(self, path):
        # Load saved state
        checkpoint = torch.load(path)
        # Restore networks, optimizer state, and exploration rate
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']