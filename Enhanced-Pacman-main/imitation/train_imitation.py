# imitation/train_imitation.py
import os
import time
import constants
import argparse
import numpy as np
from datetime import datetime
import torch

from imitation.behavioral_cloning import ImprovedImitationLearning
from stable_baselines3 import DQN

def train_imitation_model(
    data_dir="outputs/imitation_data",
    maze_type="SIMPLE",
    observation_dim=42,
    action_dim=4,
    use_enhanced_features=True,
    epochs=150,
    batch_size=128,
    test_size=0.2,
    save_dir="outputs/models",
    plots_dir="outputs/plots",
    learning_rate=0.0001
):
    """Train an imitation learning model from expert demonstrations.
    
    Args:
        data_dir: Directory containing demonstration data
        maze_type: Type of maze used for training (SIMPLE or COMPLEX)
        observation_dim: Dimension of observation vector
        action_dim: Number of possible actions
        use_enhanced_features: Whether to use enhanced features
        epochs: Number of training epochs
        batch_size: Batch size for training
        test_size: Fraction of data to use for testing
        save_dir: Directory to save the trained model
        plots_dir: Directory to save training plots
        learning_rate: Learning rate for training
    
    Returns:
        imitation_model: Trained imitation learning model
        dqn_model: Converted DQN model
    """
    # Set the maze type in constants
    constants.set_maze_type(maze_type)
    
    # Create save directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\n==== TRAINING IMITATION LEARNING MODEL ====")
    print(f"Maze type: {maze_type}")
    print(f"Observation dimension: {observation_dim}")
    print(f"Enhanced features: {use_enhanced_features}")
    
    # Create a run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"imitation_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save hyperparameters
    hyperparams = {
        'data_dir': data_dir,
        'maze_type': maze_type,
        'observation_dim': observation_dim,
        'action_dim': action_dim,
        'use_enhanced_features': use_enhanced_features,
        'epochs': epochs,
        'batch_size': batch_size,
        'test_size': test_size,
        'learning_rate': learning_rate,
        'timestamp': timestamp
    }
    
    with open(os.path.join(run_dir, "hyperparameters.txt"), 'w') as f:
        for k, v in hyperparams.items():
            f.write(f"{k}: {v}\n")
    
    # Initialise model
    imitation_learner = ImprovedImitationLearning(
        observation_dim=observation_dim,
        action_dim=action_dim,
        use_enhanced_features=use_enhanced_features
    )
    
    # Override optimizer with specified learning rate
    imitation_learner.optimizer = torch.optim.Adam(
        imitation_learner.policy_net.parameters(), 
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    # Load expert demonstrations
    print(f"\nLoading expert demonstrations from {data_dir}...")
    try:
        observations, actions, rewards = imitation_learner.load_demonstrations(
            data_dir=data_dir,
            filter_maze_type=maze_type
        )
    except Exception as e:
        print(f"Error loading demonstrations: {e}")
        return None, None
    
    if len(observations) < 100:
        print(f"WARNING: Only {len(observations)} state-action pairs found. ")
        print("This may not be enough for effective training. Consider recording more demonstrations.")
        
        if len(observations) < 20:
            print("ERROR: Insufficient data for training. Please record more demonstrations first.")
            return None, None
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_loader, val_loader, val_data = imitation_learner.prepare_datasets(
        observations, actions, 
        test_size=test_size, 
        batch_size=batch_size
    )
    
    # Train model
    print(f"\nTraining model for {epochs} epochs...")
    print(f"Using early stopping with patience=15")
    start_time = time.time()
    history = imitation_learner.train(
        train_loader, val_loader, val_data, 
        epochs=epochs,
        early_stopping_patience=15
    )
    training_time = time.time() - start_time
    
    # Save model and plot training history
    model_name = f"imitation_learning_{maze_type.lower()}"
    model_path = os.path.join(run_dir, f"{model_name}.pth")
    dqn_path = os.path.join(run_dir, f"pacman_dqn_imitation_{maze_type.lower()}")
    plot_path = os.path.join(plots_dir, f"{model_name}_history_{timestamp}.png")
    
    print(f"\nSaving model to {model_path}...")
    imitation_learner.save_model(model_path)
    
    # Also save to main models directory for easy access
    main_model_path = os.path.join(save_dir, f"{model_name}.pth")
    imitation_learner.save_model(main_model_path)
    
    print(f"Plotting training history to {plot_path}...")
    imitation_learner.plot_training_history(plot_path)
    
    # Test the model on validation data
    print("\nTesting model predictions on validation data...")
    X_val, y_val = val_data
    correct = 0
    class_correct = [0] * action_dim
    class_total = [0] * action_dim
    
    for i in range(len(X_val)):
        pred, probs = imitation_learner.predict(X_val[i])
        true = y_val[i]
        
        if pred == true:
            correct += 1
            class_correct[true] += 1
        class_total[true] += 1
    
    accuracy = 100 * correct / len(X_val)
    print(f"Overall accuracy: {accuracy:.2f}%")
    print("Per-class accuracy:")
    action_names = ["Right", "Left", "Down", "Up"]
    for i in range(action_dim):
        class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  {action_names[i]}: {class_acc:.2f}%")
    
    # Sample predictions on a few examples
    print("\nSample predictions:")
    for i in range(min(5, len(X_val))):
        pred, probs = imitation_learner.predict(X_val[i])
        true = y_val[i]
        print(f"  Sample {i+1} - True: {action_names[true]} ({true}), Predicted: {action_names[pred]} ({pred})")
        print(f"    Probabilities: " + ", ".join([f"{action_names[j]}: {p:.4f}" for j, p in enumerate(probs)]))
    
    # Convert to DQN model for gameplay
    print(f"\nConverting to DQN format for compatibility...")
    dqn_model = imitation_learner.convert_to_dqn(dqn_path)
    
    # Also save to main DQN model directory for easy access
    main_dqn_path = os.path.join(save_dir, f"pacman_dqn_imitation_{maze_type.lower()}")
    dqn_model.save(main_dqn_path)
    
    # Print training summary
    print("\n==== TRAINING SUMMARY ====")
    print(f"Model trained on {len(observations)} state-action pairs")
    print(f"Final validation accuracy: {accuracy:.2f}%")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Model saved to {model_path}")
    print(f"DQN version saved to {main_dqn_path}")
    
    # Quality evaluation
    if accuracy < 50:
        print("\nWARNING: Model accuracy is low. Recommendations:")
        print("1. Record more demonstrations with clear, consistent gameplay")
        print("2. Try training for more epochs (200-300)")
        print("3. Experiment with different learning rates")
        print("4. Make sure your demonstrations have good coverage of the maze")
    elif accuracy < 70:
        print("\nModel accuracy is moderate. Consider:")
        print("1. Recording additional demonstrations")
        print("2. Fine-tuning with a lower learning rate")
    else:
        print("\nModel accuracy is good. Ready for gameplay testing!")
        
    # Create a test script for the model
    test_script_path = os.path.join(run_dir, f"test_{model_name}.py")
    with open(test_script_path, 'w') as f:
        f.write(f"""
# Test script for imitation learning model
import os
import torch
import numpy as np
from behavioral_cloning import ImprovedImitationLearning

# Load the model
model_path = "{model_path}"
model = ImprovedImitationLearning()
model.load_model(model_path)

# Create a random observation
obs = np.random.random(model.observation_dim).astype(np.float32)

# Make a prediction
action, probs = model.predict(obs)
print(f"Predicted action: {{action}}")
print(f"Action probabilities: {{probs}}")

# Try predicting the same action 10 times to check consistency
print("\\nTesting prediction consistency:")
for i in range(10):
    action, _ = model.predict(obs)
    print(f"Prediction {{i+1}}: {{action}}")
""")
    
    print(f"\nTest script created at {test_script_path}")
    
    return imitation_learner, dqn_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an imitation learning model for Pac-Man")
    parser.add_argument("--data_dir", type=str, default="outputs/imitation_data", help="Directory containing demonstration data")
    parser.add_argument("--maze_type", type=str, default="SIMPLE", choices=["SIMPLE", "COMPLEX"], help="Type of maze used for training")
    parser.add_argument("--use_enhanced", action="store_true", help="Use enhanced features")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training")
    
    args = parser.parse_args()
    
    # Set observation dimension based on feature set
    obs_dim = 42 if args.use_enhanced else 18
    
    train_imitation_model(
        data_dir=args.data_dir,
        maze_type=args.maze_type,
        observation_dim=obs_dim,
        use_enhanced_features=args.use_enhanced,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )