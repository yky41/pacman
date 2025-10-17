import os
import torch
import constants
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

# Import our custom components
from dqn.monitors import PacmanMonitor
from dqn.callbacks import TensorboardScoreCallback
from dqn.environment import PacmanEnv

class LearningRateSchedulerCallback(BaseCallback):
    """
    Callback for dynamic learning rate adjustment
    """
    def __init__(self, schedule_freq=50000, initial_lr=0.0001, min_lr=0.00001, verbose=1):
        super(LearningRateSchedulerCallback, self).__init__(verbose)
        self.schedule_freq = schedule_freq
        self.initial_lr = initial_lr
        self.min_lr = min_lr
    
    def _on_step(self):
        # Implement cosine annealing with warm restarts
        if self.num_timesteps % self.schedule_freq == 0 and self.num_timesteps > 0:
            # Calculate current learning rate
            progress = min(1.0, self.num_timesteps / (2 * 1000000))  # 2 million steps max
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())
            lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
            
            # Apply new learning rate
            self.model.learning_rate = lr
            
            if self.verbose > 0:
                print(f"\nAdjusting learning rate to {lr:.6f}")
        
        return True

def train_dqn_model(maze, pellet_grid, timesteps=400000, force_retrain=False, log_dir="./outputs/logs", use_enhanced_features=True):
    """
    Train DQN model with improved hyperparameters for better ghost avoidance
    """
    print(f"\n=== TRAINING DQN MODEL ===")
    print(f"Maze type: {constants.CURRENT_MAZE_TYPE}")
    print(f"Maze dimensions: {constants.ROWS}x{constants.COLS}")
    print(f"Enhanced features: {use_enhanced_features}")
    
    # Create log directory if needed
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_log = os.path.join(log_dir, "tensorboard")
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # Create models directory if needed
    models_dir = os.path.join("./outputs/logs", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if model exists and not forcing retrain
    model_path = f"outputs/models/pacman_dqn_{constants.CURRENT_MAZE_TYPE.lower()}"
    if os.path.exists(model_path + ".zip") and not force_retrain:
        print(f"Model already exists at {model_path}. Use force_retrain=True to train new model.")
        return load_dqn_model(model_path)
    
    print(f"Training new DQN model with {'enhanced' if use_enhanced_features else 'standard'} features...")
    
    # Create environment with specified feature set
    env = PacmanEnv(maze, pellet_grid, use_enhanced_features=use_enhanced_features)
    
    # Use custom monitor to track scores and completion rates
    env = PacmanMonitor(env, os.path.join(log_dir, "monitor"), 
                       info_keywords=('score', 'pellets_remaining', 'won', 'revisit_rate'))
    
    # Create evaluation environment
    eval_env = PacmanMonitor(
        PacmanEnv(maze, pellet_grid, use_enhanced_features=use_enhanced_features),
        os.path.join(log_dir, "eval_monitor"),
        info_keywords=('score', 'pellets_remaining', 'won', 'revisit_rate')
    )
    
    # Use optimised hyperparameters based on maze type
    if constants.CURRENT_MAZE_TYPE == "SIMPLE":
        learning_rate = 0.00015
        buffer_size = 100000 
        learning_starts = 10000
        batch_size = 128
        exploration_fraction = 0.4  # Faster exploration decay
        target_update_interval = 1000
        net_arch = [256, 128, 64]
        weight_decay = 1e-6
        min_training_steps = 250000
        
    else:
        # Original hyperparameters for complex maze
        learning_rate = 0.0001
        buffer_size = 250000
        learning_starts = 25000
        batch_size = 256
        exploration_fraction = 0.6  # Adjusted for better exploration
        target_update_interval = 1500
        net_arch = [512, 256, 128]
        weight_decay = 1e-5
        min_training_steps = 100000
    
    print(f"Using hyperparameters for {constants.CURRENT_MAZE_TYPE} maze")
    
    # Create model with TensorBoard logging and improved hyperparameters
    model = DQN(
        "MlpPolicy", 
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=0.99,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=target_update_interval,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs=dict(
            net_arch=net_arch,
            activation_fn=torch.nn.ReLU,
            optimizer_kwargs=dict(weight_decay=weight_decay)
        ),
        tensorboard_log=tensorboard_log,
        verbose=1
    )
    
    # Set up checkpoints to save during training
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=models_dir,
        name_prefix=f"dqn_pacman_{constants.CURRENT_MAZE_TYPE.lower()}_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Evaluation callback to track performance
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_results"),
        eval_freq=20000,
        deterministic=True,
        render=False,
        n_eval_episodes=15
    )
    
    # Create TensorBoard metrics callback
    tensorboard_callback = TensorboardScoreCallback()
    
    # Create learning rate scheduler callback
    lr_callback = LearningRateSchedulerCallback(
        schedule_freq=500000,
        initial_lr=learning_rate,
        min_lr=learning_rate/10
    )
    
    # Create combined callback
    combined_callbacks = [checkpoint_callback, eval_callback, tensorboard_callback, lr_callback]
    
    # Ensure appropriate training time based on maze type
    actual_timesteps = max(timesteps, min_training_steps)
    
    # Train model
    model.learn(
        total_timesteps=actual_timesteps, 
        callback=combined_callbacks,
        progress_bar=True
    )
    
    # Save model with maze type in filename
    model.save(model_path)
    print(f"DQN model trained and saved to {model_path}!")
    
    # Save best model separately
    best_model_path = os.path.join(models_dir, "best_model/best_model.zip")
    if os.path.exists(best_model_path):
        best_model = DQN.load(best_model_path)
        best_model.save(f"pacman_dqn_{constants.CURRENT_MAZE_TYPE.lower()}_best")
        print(f"Best model during training saved to pacman_dqn_{constants.CURRENT_MAZE_TYPE.lower()}_best.zip")
        return best_model
    
    return model

def load_dqn_model(model_path=f"./outputs/pacman_dqn_{constants.CURRENT_MAZE_TYPE.lower()}", strict=False):

    # Try best model first
    best_path = model_path + "_best"
    if os.path.exists(best_path + ".zip"):
        try:
            print(f"Loading best DQN model from {best_path}...")
            model = DQN.load(best_path)
            print("Best DQN model loaded successfully!")
            return model
        except Exception as e:
            print(f"Could not load best model: {e}")
            # Fall back to regular model
    
    # Try regular model
    try:
        print(f"Loading DQN model from {model_path}...")
        model = DQN.load(model_path)
        print("DQN model loaded successfully!")
        return model
    except Exception as e:
        print(f"Could not load DQN model: {e}")
        if strict:
            raise FileNotFoundError(f"No model found at {model_path}. Please train model first.")
        print("You may need to train a model first.")
        return None

def evaluate_model(model, env, num_episodes=10):

    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=num_episodes,
        deterministic=True,
        render=False,
        return_episode_rewards=False
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward