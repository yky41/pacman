import torch
import numpy as np
import os
from stable_baselines3 import DQN
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import logging
import pygame
from utils.helpers import pixel_to_grid, collides_with_wall
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('imitation_model')

class FixedImitationModel(nn.Module):
    """Imitation learning model with the exact same architecture as DQN's q_net."""
    def __init__(self, observation_dim=42, action_dim=4):
        super(FixedImitationModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Use the exact same architecture as DQN's q_net
        self.q_net = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.q_net(x)
    
    def act(self, obs, temperature=10.0):  # Much higher temperature to flatten distribution
        """Get action from observation with additional debugging and temperature scaling."""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_net(obs_tensor)
            
            # Apply temperature scaling before softmax
            scaled_q_values = q_values / temperature
            
            # Add debugging to print all action probabilities
            probs = torch.softmax(scaled_q_values, dim=1)[0].numpy()
            action_names = ["Right", "Left", "Down", "Up"]
            print(f"Action probabilities (temperature={temperature}):")
            for i, prob in enumerate(probs):
                print(f"  {action_names[i]}: {prob:.6f}")
            
            # Sample from the distribution instead of taking argmax
            probs_tensor = torch.softmax(scaled_q_values, dim=1)[0]
            action = torch.multinomial(probs_tensor, 1).item()
            return action

def convert_imitation_to_dqn_fixed(imitation_path, dqn_output_path):
    """Improved conversion from imitation learning model to DQN format."""
    print(f"Loading imitation model from {imitation_path}")
    if not os.path.exists(imitation_path):
        raise FileNotFoundError(f"Model file {imitation_path} not found")
        
    # Load the imitation model checkpoint
    checkpoint = torch.load(imitation_path, map_location=torch.device('cpu'))
    observation_dim = checkpoint['observation_dim']
    action_dim = checkpoint['action_dim']
    
    # Create a dummy environment with the right dimensions
    class DummyEnv(gym.Env):
        def __init__(self, obs_dim, act_dim):
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(act_dim)
            
        def reset(self, **kwargs):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {}
            
        def step(self, action):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}
    
    dummy_env = DummyEnv(observation_dim, action_dim)
    
    # Create a fixed imitation model that matches DQN architecture exactly
    fixed_imitation = FixedImitationModel(observation_dim, action_dim)
    
    # Create a new DQN model with identical architecture
    dqn_model = DQN(
        policy="MlpPolicy",
        env=dummy_env,
        learning_rate=0.0001,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        tau=1.0,  # Set to 1.0 to ensure exact copying of weights
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        ),
        verbose=1
    )
    
    # Extract the imitation model state dict
    imitation_state_dict = {}
    for name, param in checkpoint['model_state_dict'].items():
        # Convert layer names to match the DQN model naming convention
        if '0.weight' in name:
            imitation_state_dict['q_net.0.weight'] = param
        elif '0.bias' in name:
            imitation_state_dict['q_net.0.bias'] = param
        elif '3.weight' in name:
            imitation_state_dict['q_net.2.weight'] = param
        elif '3.bias' in name:
            imitation_state_dict['q_net.2.bias'] = param
        elif '6.weight' in name:
            imitation_state_dict['q_net.4.weight'] = param
        elif '6.bias' in name:
            imitation_state_dict['q_net.4.bias'] = param
        elif '8.weight' in name:
            imitation_state_dict['q_net.6.weight'] = param
        elif '8.bias' in name:
            imitation_state_dict['q_net.6.bias'] = param
    
    # Manually set weights for the new fixed imitation model
    fixed_imitation_dict = fixed_imitation.state_dict()
    for name, param in imitation_state_dict.items():
        if name in fixed_imitation_dict:
            fixed_imitation_dict[name] = param
    
    fixed_imitation.load_state_dict(fixed_imitation_dict)
    
    # Test the fixed model with a random observation
    print("\nTesting fixed imitation model with random observation:")
    test_obs = np.random.rand(observation_dim).astype(np.float32)
    action = fixed_imitation.act(test_obs)
    print(f"Predicted action: {action}")
    
    # Copy weights to DQN q_net
    dqn_state_dict = dqn_model.policy.state_dict()
    
    # Print layer shapes for verification
    print("\nDQN model parameter shapes:")
    for name, param in dqn_model.policy.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print("\nFixed imitation model parameter shapes:")
    for name, param in fixed_imitation.named_parameters():
        if 'q_net' in name:
            print(f"  {name}: {param.shape}")
    
    # Direct copy of q_net parameters from fixed model to DQN
    for dqn_name, dqn_param in dqn_model.policy.q_net.named_parameters():
        fixed_name = dqn_name  # Same naming convention in fixed model
        if hasattr(fixed_imitation.q_net, fixed_name.split('.')[0]):
            # Get corresponding parameter from fixed model
            for name, param in fixed_imitation.q_net.named_parameters():
                if name == fixed_name:
                    print(f"Copying {name} -> {dqn_name}")
                    dqn_param.data.copy_(param.data)
                    break
    
    # Save the updated DQN model
    dqn_model.save(dqn_output_path)
    print(f"Fixed DQN model saved to {dqn_output_path}")
    
    # Test the DQN model
    print("\nTesting DQN model with the same observation:")
    obs_tensor = torch.as_tensor(test_obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = dqn_model.q_net(obs_tensor)
        dqn_action = torch.argmax(q_values, dim=1).item()
        # Print q-values
        print("DQN q-values:", q_values.numpy())
        print(f"DQN predicted action: {dqn_action}")
    
    return dqn_model, fixed_imitation

def load_imitation_model(model_path="outputs/models/imitation_learning_simple.pth", debug=True):
    """Load the imitation learning model with improved error handling and debugging.
    
    Args:
        model_path: Path to the model file
        debug: Whether to print debug information
        
    Returns:
        model: Loaded imitation learning model or None if loading fails
    """
    try:
        logger.info(f"Loading imitation model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Extract model parameters
        observation_dim = checkpoint.get('observation_dim', 42)
        action_dim = checkpoint.get('action_dim', 4)
        
        # Create fixed model
        model = FixedImitationModel(observation_dim, action_dim)
        
        if debug:
            logger.info(f"Created model with observation_dim={observation_dim}, action_dim={action_dim}")
        
        # Check if the loaded model uses the old architecture
        original_state_dict = checkpoint['model_state_dict']
        
        # Create a mapping from old architecture to new one
        mapped_state_dict = {}
        
        # Map old keys to new keys
        key_mapping = {
            '0.weight': 'q_net.0.weight',
            '0.bias': 'q_net.0.bias',
            '3.weight': 'q_net.2.weight',
            '3.bias': 'q_net.2.bias',
            '6.weight': 'q_net.4.weight',
            '6.bias': 'q_net.4.bias',
            '8.weight': 'q_net.6.weight',
            '8.bias': 'q_net.6.bias'
        }
        
        # Check if model is already in new format (has 'q_net' prefix)
        if any('q_net' in key for key in original_state_dict.keys()):
            # Already in correct format
            mapped_state_dict = original_state_dict
        else:
            # Map old format to new format
            for old_key, new_key in key_mapping.items():
                if old_key in original_state_dict:
                    mapped_state_dict[new_key] = original_state_dict[old_key]
        
        # Log weights before loading
        if debug:
            logger.info("Model parameters before loading:")
            for name, param in model.named_parameters():
                logger.info(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
        
        if missing_keys and debug:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys and debug:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        # Log weights after loading
        if debug:
            logger.info("Model parameters after loading:")
            for name, param in model.named_parameters():
                logger.info(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
        
        # Verify the model works
        if debug:
            test_observation = np.random.rand(observation_dim).astype(np.float32)
            logger.info(f"Testing model with random observation...")
            action = model.act(test_observation)
            logger.info(f"Test prediction successful, action: {action}")
        
        logger.info("Imitation model loaded successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Error loading imitation model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def apply_imitation_model(pacman, active_ghosts, pellet_grid, maze, model):
    """Apply the imitation model to control Pac-Man."""
    try:
        # Get observation and current position
        observation = pacman._create_observation(active_ghosts, pellet_grid, maze)
        current_pos = pixel_to_grid(pacman.x, pacman.y)
        
        # Get valid actions that won't hit walls
        valid_actions = []
        direction_map = {
            0: pygame.Vector2(1, 0),   # Right
            1: pygame.Vector2(-1, 0),  # Left
            2: pygame.Vector2(0, 1),   # Down
            3: pygame.Vector2(0, -1)   # Up
        }
        
        for action, direction in direction_map.items():
            # Test if this direction is valid (doesn't hit a wall)
            test_x = pacman.x + direction.x * pacman.speed
            test_y = pacman.y + direction.y * pacman.speed
            if not collides_with_wall(test_x, test_y, pacman.radius, maze):
                valid_actions.append(action)
        
        # If no valid actions, just return current direction
        if not valid_actions:
            return pacman.last_action if hasattr(pacman, 'last_action') else 0, pacman.direction
        
        # Use model to predict action
        _, probs = model.predict(observation, exploration=False)
        
        # Choose the highest probability valid action
        # Sort actions by probability
        action_probs = [(a, probs[a]) for a in valid_actions]
        action_probs.sort(key=lambda x: x[1], reverse=True)
        
        # With 70% probability, take best valid action
        # With 30% probability, take random valid action
        if random.random() < 0.7:
            action = action_probs[0][0]  # Best valid action
        else:
            action = random.choice(valid_actions)  # Random valid action
            
        # Save the action
        pacman.last_action = action
        
        return action, direction_map[action]
        
    except Exception as e:
        # If any error, take a random valid action
        # Get valid actions
        valid_actions = []
        direction_map = {
            0: pygame.Vector2(1, 0),   # Right
            1: pygame.Vector2(-1, 0),  # Left
            2: pygame.Vector2(0, 1),   # Down
            3: pygame.Vector2(0, -1)   # Up
        }
        
        for action, direction in direction_map.items():
            test_x = pacman.x + direction.x * pacman.speed
            test_y = pacman.y + direction.y * pacman.speed
            if not collides_with_wall(test_x, test_y, pacman.radius, maze):
                valid_actions.append(action)
        
        # Take a random valid action
        action = random.choice(valid_actions) if valid_actions else 0
        return action, direction_map[action]

def detect_oscillation(positions, window=6):
    """Detect if the agent is oscillating between positions.
    
    Returns:
        bool: True if oscillation is detected, False otherwise
    """
    if len(positions) < window:
        return False
    
    # Check for simple A-B-A-B pattern
    recent = positions[-window:]
    if (recent[0] == recent[2] == recent[4] and 
        recent[1] == recent[3] == recent[5]):
        return True
    
    # Check for A-B-C-A-B-C pattern
    if window >= 6 and len(set(recent)) <= 3:
        # Count unique positions
        if len(set(recent[::2])) == 1 and len(set(recent[1::2])) <= 2:
            return True
    
    return False