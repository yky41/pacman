import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class ImitationDataset(Dataset):
    """Dataset for imitation learning from expert demonstrations."""
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions
        
    def __len__(self):
        return len(self.observations)
        
    def __getitem__(self, idx):
        x = torch.tensor(self.observations[idx], dtype=torch.float32)
        y = torch.tensor(self.actions[idx], dtype=torch.long)
        return x, y

class ImprovedImitationLearning:
    """Enhanced implementation of behavioural cloning for Pac-Man."""
    def __init__(self, observation_dim=42, action_dim=4, use_enhanced_features=True):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.use_enhanced_features = use_enhanced_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Smaller, more regularised policy network
        self.policy_net = nn.Sequential(
            nn.Linear(observation_dim, 32),    # Smaller first layer
            nn.Tanh(),                         # Try different activation
            nn.Dropout(0.6),                   # Higher dropout
            nn.Linear(32, 16),                 # Smaller hidden layer
            nn.Tanh(),                         # Different activation
            nn.Dropout(0.6),                   # Higher dropout
            nn.Linear(16, action_dim)
        ).to(self.device)
        
        # Much stronger regularisation
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=0.00001,                    # Lower learning rate
            weight_decay=2e-2              # Much stronger L2 regularisation
        )
        
        # Use Cross Entropy Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, factor=0.5, verbose=True
        )
        
        # Training metrics
        self.training_losses = []
        self.validation_losses = []
        self.validation_accuracies = []
        
    def load_demonstrations(self, data_dir="outputs/imitation_data", filter_maze_type=None):
        """Load expert demonstrations."""
        import pickle
        
        all_observations = []
        all_actions = []
        all_rewards = []
        
        # Get list of pickle files
        episode_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl') and f.startswith('episode_')]
        
        if not episode_files:
            raise ValueError(f"No demonstration files found in {data_dir}")
            
        print(f"Found {len(episode_files)} demonstration files")
        
        for filename in episode_files:
            filepath = os.path.join(data_dir, filename)
            
            with open(filepath, 'rb') as f:
                episode = pickle.load(f)
                
            # Skip if not the right maze type
            if filter_maze_type and episode['maze_type'] != filter_maze_type:
                print(f"Skipping episode from maze type {episode['maze_type']}")
                continue
                
            # Add to the dataset
            observations = episode['observations']
            actions = episode['actions']
            rewards = episode['rewards']
            
            # Filter out None actions
            valid_indices = [i for i, a in enumerate(actions) if a is not None]
            
            filtered_observations = [observations[i] for i in valid_indices]
            filtered_actions = [actions[i] for i in valid_indices]
            filtered_rewards = [rewards[i] for i in valid_indices]
            
            all_observations.extend(filtered_observations)
            all_actions.extend(filtered_actions)
            all_rewards.extend(filtered_rewards)
        
        print(f"Loaded {len(all_observations)} state-action pairs from expert demonstrations")
        
        # If using enhanced features, make sure we have the right observation dimension
        if self.use_enhanced_features and len(all_observations) > 0:
            # Check if observations match the expected dimension
            if len(all_observations[0]) != self.observation_dim:
                # If observations are longer, truncate them
                if len(all_observations[0]) > self.observation_dim:
                    print(f"Truncating observations from {len(all_observations[0])} to {self.observation_dim} features")
                    all_observations = [obs[:self.observation_dim] for obs in all_observations]
                else:
                    raise ValueError(f"Observations have {len(all_observations[0])} features, expected {self.observation_dim}")
        
        # Analyse action distribution
        action_counts = np.bincount(all_actions, minlength=self.action_dim)
        print("\nAction distribution in demonstrations:")
        action_names = ["Right", "Left", "Down", "Up"]
        for i, count in enumerate(action_counts):
            percentage = 100 * count / len(all_actions) if len(all_actions) > 0 else 0
            print(f"  {action_names[i]}: {count} ({percentage:.1f}%)")
            
        # Check for severe class imbalance
        max_count = max(action_counts)
        min_count = max(1, min(action_counts))  # Avoid division by zero
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 3:
            print(f"\nWARNING: Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
            print("Applying class weights to loss function")
            
            # Calculate class weights for weighted loss function
            class_weights = 1.0 / action_counts
            class_weights = class_weights / np.sum(class_weights) * len(class_weights)
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        
        return all_observations, all_actions, all_rewards
    
    def mixup_data(self, x, y, alpha=0.2):
        """Apply mixup augmentation to the batch."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def add_noise_to_observations(self, observations, noise_level=0.01):
        """Add Gaussian noise to observations for data augmentation."""
        observations_array = np.array(observations, dtype=np.float32)
        noisy_observations = observations_array + np.random.normal(0, noise_level, observations_array.shape)
        return noisy_observations
    
    def prepare_datasets(self, observations, actions, test_size=0.2, batch_size=128):
        """Prepare training and validation datasets with better randomisation."""
        # Convert to numpy arrays
        X = np.array(observations, dtype=np.float32)
        y = np.array(actions, dtype=np.int64)
        
        # Create an array of indices
        indices = np.arange(len(X))
        
        # Shuffle the indices randomly
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        # Calculate split point
        split = int(len(indices) * (1 - test_size))
        
        # Split indices into train and validation sets
        train_indices = indices[:split]
        val_indices = indices[split:]
        
        # Use the indices to create the train and validation sets
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Add synthetic data that is completely random
        # This will force the model to learn more slowly
        num_synthetic = len(X_train) // 2  # Add 50% synthetic data
        synthetic_X = np.random.normal(0.5, 0.5, (num_synthetic, X_train.shape[1]))
        synthetic_y = np.random.randint(0, self.action_dim, num_synthetic)
        
        # Include synthetic data in the training data
        X_train = np.vstack([X_train, synthetic_X])
        y_train = np.concatenate([y_train, synthetic_y])
        
        # Create augmented versions of the training data
        X_aug = []
        y_aug = []
        
        # Data augmentation: add slight random noise to training examples
        for i in range(len(X_train)):
            X_aug.append(X_train[i] + np.random.normal(0, 0.02, X_train[i].shape))
            y_aug.append(y_train[i])
            
            # For minority classes, generate more augmented samples
            # This helps with class imbalance
            action = y_train[i]
            counts = np.bincount(y_train, minlength=self.action_dim)
            max_count = max(counts)
            
            # Add more augmented samples for minority classes
            if counts[action] < max_count / 2:
                # Add 2 more samples with different noise patterns
                for _ in range(2):
                    X_aug.append(X_train[i] + np.random.normal(0, 0.03, X_train[i].shape))
                    y_aug.append(action)
        
        # Combine original and augmented data
        X_train_combined = np.vstack([X_train, np.array(X_aug)])
        y_train_combined = np.concatenate([y_train, np.array(y_aug)])
        
        # Print augmented dataset info
        print(f"Original training set: {len(X_train)} samples")
        print(f"Augmented training set: {len(X_train_combined)} samples")
        
        # Create datasets
        train_dataset = ImitationDataset(X_train_combined, y_train_combined)
        val_dataset = ImitationDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, (X_val, y_val)
    
    def train(self, train_loader, val_loader, val_data, epochs=150, early_stopping_patience=15):
        """Train the imitation learning model with regularisation and early stopping."""
        print(f"Training on {self.device}")
        
        # Initialise early stopping variables
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        train_val_gap_too_large_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.policy_net.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Apply mixup data augmentation
                use_mixup = np.random.random() < 0.5  # Apply mixup 50% of the time
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                if use_mixup:
                    # Apply mixup
                    inputs_mixed, targets_a, targets_b, lam = self.mixup_data(inputs, targets)
                    
                    # Forward pass
                    outputs = self.policy_net(inputs_mixed)
                    loss = self.mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                else:
                    # Regular forward pass
                    outputs = self.policy_net(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass and optimise
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Calculate accuracy (only for non-mixup batches)
                if not use_mixup:
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += targets.size(0)
                    train_correct += (predicted == targets).sum().item()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0
            
            # Validation
            self.policy_net.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # Track per-class accuracy for better debugging
            class_correct = [0] * self.action_dim
            class_total = [0] * self.action_dim
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.policy_net(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
                    
                    # Track per-class accuracy
                    for i in range(len(targets)):
                        label = targets[i].item()
                        if predicted[i].item() == label:
                            class_correct[label] += 1
                        class_total[label] += 1
                    
                    val_loss += loss.item() * inputs.size(0)
                
                # Compute overall validation metrics
                X_val, y_val = val_data
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
                
                outputs = self.policy_net(X_val_tensor)
                val_loss = self.criterion(outputs, y_val_tensor).item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_accuracy = 100.0 * (predicted == y_val_tensor).sum().item() / len(y_val)
            
            # Print per-class accuracy every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                print("\nPer-class accuracy:")
                action_names = ["Right", "Left", "Down", "Up"]
                for i in range(self.action_dim):
                    if class_total[i] > 0:
                        class_acc = 100.0 * class_correct[i] / class_total[i]
                        print(f"  {action_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
            
            # Check for model getting stuck predicting one class
            predictions = predicted.cpu().numpy()
            unique_preds, counts = np.unique(predictions, return_counts=True)
            if len(unique_preds) < self.action_dim / 2:  # Using less than half the available actions
                dominant_class = unique_preds[np.argmax(counts)]
                dominant_pct = 100 * counts.max() / len(predictions)
                print(f"WARNING: Model may be biased toward class {dominant_class} ({dominant_pct:.1f}% of predictions)")
                
                if dominant_pct > 90 and epoch > 10:
                    print("Severe bias detected - consider restarting training with different initialisation")
                    
                    # Optional: Try to adjust learning rate or add noise to weights
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 2.0  # Boost learning rate
                        print(f"Boosting learning rate to {param_group['lr']}")
                    
                    # Add noise to weights to break out of local minimum
                    for param in self.policy_net.parameters():
                        param.data += torch.randn_like(param.data) * 0.01
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Save metrics
            self.training_losses.append(train_loss)
            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Check for overfitting: train accuracy much higher than validation accuracy
            if train_accuracy - val_accuracy > 15:  # Gap of more than 15%
                train_val_gap_too_large_counter += 1
                if train_val_gap_too_large_counter >= 3:  # If gap persists for 3 epochs
                    print(f"Training-validation accuracy gap too large for 3 epochs. Stopping early.")
                    if best_model_state is not None:
                        self.policy_net.load_state_dict(best_model_state)
                    break
            else:
                train_val_gap_too_large_counter = 0  # Reset counter if gap is acceptable
            
            # Early stopping based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.policy_net.state_dict().copy()
                print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
            # Stop if training accuracy reaches 100% to prevent overfitting
            if train_accuracy >= 99.9 and epoch > 10:
                print("Training accuracy reached 100%. Stopping to prevent overfitting.")
                break
        
        # Load the best model
        if best_model_state is not None:
            self.policy_net.load_state_dict(best_model_state)
            print(f"Restored best model with validation accuracy: {best_val_accuracy:.2f}%")
        
        return {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'validation_accuracies': self.validation_accuracies,
            'best_accuracy': best_val_accuracy
        }
    
    def save_model(self, filepath="outputs/models/imitation_learning.pth"):
        """Save the trained model to a file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'use_enhanced_features': self.use_enhanced_features,
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'validation_accuracies': self.validation_accuracies,
            'model_architecture': str(self.policy_net)
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="outputs/models/imitation_learning.pth"):
        """Load a trained model from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update model parameters
        self.observation_dim = checkpoint['observation_dim']
        self.action_dim = checkpoint['action_dim']
        self.use_enhanced_features = checkpoint['use_enhanced_features']
        
        # Recreate policy network with correct dimensions
        self.policy_net = nn.Sequential(
            nn.Linear(self.observation_dim, 64),
            nn.Tanh(),
            nn.Dropout(0.6),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Dropout(0.6),
            nn.Linear(32, self.action_dim)
        ).to(self.device)
        
        # Load state dict
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        
        # Load metrics if available
        if 'training_losses' in checkpoint:
            self.training_losses = checkpoint['training_losses']
        if 'validation_losses' in checkpoint:
            self.validation_losses = checkpoint['validation_losses']
        if 'validation_accuracies' in checkpoint:
            self.validation_accuracies = checkpoint['validation_accuracies']
            
        print(f"Model loaded from {filepath}")
        if 'model_architecture' in checkpoint:
            print(f"Model architecture: {checkpoint['model_architecture']}")
    
    def predict(self, observation, exploration=False, epsilon=0.1, temperature=4.0):
        """Predict action for a given observation with options for exploration.
        
        Args:
            observation: Observation vector
            exploration: Whether to use exploration (epsilon-greedy or temperature)
            epsilon: Probability of random action (if exploration=True)
            temperature: Temperature for softmax (higher = more uniform distribution)
            
        Returns:
            action: Predicted action
            probs: Action probabilities
        """
        
        self.policy_net.eval()
        
        # Convert to tensor
        if isinstance(observation, np.ndarray) and len(observation.shape) == 1:
            observation = observation.reshape(1, -1)  # Add batch dimension if needed
            
        observation_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.policy_net(observation_tensor)
            
            # Add exploration if needed
            if exploration:
                # Epsilon-greedy: random action with probability epsilon
                if random.random() < epsilon:
                    action = random.randint(0, self.action_dim - 1)
                    
                    # Create a one-hot-like probability distribution for the random action
                    probs = np.zeros(self.action_dim)
                    probs[action] = 1.0
                    
                    return action, probs
                
                # Otherwise use temperature scaling for softmax
                probs = F.softmax(outputs / temperature, dim=1)
                action_probs = probs.data.cpu().numpy()[0]
                
                # Print action probabilities for debugging
                action_names = ["Right", "Left", "Down", "Up"]
                print(f"Action probabilities (temperature={temperature}):")
                for i, prob in enumerate(probs):
                    print(f"  {action_names[i]}: {prob:.6f}")
                
                # Either sample from distribution or take argmax
                if random.random() < 0.7:  # 70% sample from distribution
                    action = np.random.choice(self.action_dim, p=action_probs)
                else:  # 30% take best action
                    action = np.argmax(action_probs)
                
                return action, action_probs
            else:
                # No exploration: just take argmax
                probs = F.softmax(outputs, dim=1)
                action_probs = probs.data.cpu().numpy()[0]
                action = np.argmax(outputs.data.cpu().numpy())
                return action, action_probs
    
    def convert_to_dqn(self, save_path="outputs/models/pacman_dqn_imitation", env=None):
        """Convert the imitation learning model to a DQN model."""
        # Implementation remains the same
        from stable_baselines3 import DQN
        from stable_baselines3.common.torch_layers import FlattenExtractor
        import gym
        from gym import spaces
        
        # Create a dummy environment if none provided
        if env is None:
            class DummyEnv(gym.Env):
                def __init__(self, obs_dim, act_dim):
                    self.observation_space = spaces.Box(
                        low=0,
                        high=1,
                        shape=(obs_dim,),
                        dtype=np.float32
                    )
                    self.action_space = spaces.Discrete(act_dim)
                    
                def reset(self, **kwargs):
                    obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                    return obs, {}
                    
                def step(self, action):
                    obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                    return obs, 0.0, True, False, {}
            
            env = DummyEnv(self.observation_dim, self.action_dim)
        
        # Define custom DQN policy architecture
        policy_kwargs = dict(
            features_extractor_class=FlattenExtractor,
            net_arch=[64, 32],  # Match our new architecture
            activation_fn=nn.Tanh  # Match our new activation
        )
        
        # Create new DQN model
        dqn_model = DQN(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=0.0001,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=128,
            gamma=0.99,
            exploration_fraction=0.1,
            target_update_interval=1000,
            verbose=1
        )
        
        # Map our policy network layers to DQN policy network layers
        imitation_params = list(self.policy_net.parameters())
        dqn_params = list(dqn_model.policy.q_net.parameters())
        
        # Print parameter shapes for debugging
        print("\nMapping parameters from imitation model to DQN model:")
        print("Imitation model parameters:")
        for i, param in enumerate(imitation_params):
            print(f"  Layer {i}: shape {param.shape}")
            
        print("\nDQN model parameters:")
        for i, param in enumerate(dqn_params):
            print(f"  Layer {i}: shape {param.shape}")
            
        # Attempt to map parameters based on shape
        copied_layers = 0
        for i, im_param in enumerate(imitation_params):
            for j, dqn_param in enumerate(dqn_params):
                if im_param.shape == dqn_param.shape:
                    print(f"  Copying imitation layer {i} to DQN layer {j}")
                    dqn_param.data.copy_(im_param.data)
                    copied_layers += 1
                    break
                    
        print(f"\nCopied {copied_layers} layers from imitation model to DQN model")
        
        # Save the model
        dqn_model.save(save_path)
        print(f"Converted DQN model saved to {save_path}")
        
        return dqn_model
    
    def plot_training_history(self, save_path="outputs/plots/imitation_learning_history.png"):
        """Plot the training history and save to a file."""
        if not self.training_losses or not self.validation_losses:
            print("No training history to plot")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create plot with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot losses
        epochs = range(1, len(self.training_losses) + 1)
        ax1.plot(epochs, self.training_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.validation_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend(loc='upper right')
        
        # Plot accuracy
        ax2.plot(epochs, self.validation_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        ax2.legend(loc='lower right')
        
        # Add final accuracy text
        if self.validation_accuracies:
            final_acc = self.validation_accuracies[-1]
            best_acc = max(self.validation_accuracies)
            ax2.text(0.02, 0.02, f"Final accuracy: {final_acc:.2f}%\nBest accuracy: {best_acc:.2f}%", 
                     transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
        
        plt.close()