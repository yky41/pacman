import os
import sys
# Add the root directory to the Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from stable_baselines3.common.evaluation import evaluate_policy

from dqn.training import train_dqn_model
from dqn.environment import PacmanEnv
from dqn.monitors import PacmanMonitor
from imitation.behavioral_cloning import ImprovedImitationLearning
from imitation.data_collector import ImprovedDataCollector
import constants

def run_convergence_test(maze_type="SIMPLE", num_runs=5, max_timesteps=100000, num_demonstrations=20, use_existing_demos=False):
    """Run comparative tests between RL and imitation learning agents to measure convergence speed and sample efficiency.
    
    Args:
        maze_type: Type of maze to test on ("SIMPLE" or "COMPLEX")
        num_runs: Number of independent runs for each method
        max_timesteps: Maximum number of training timesteps
        num_demonstrations: Number of expert demonstrations to collect/use
        use_existing_demos: Whether to use existing demonstrations if available
    """
    # Set maze type
    constants.set_maze_type(maze_type)
    maze = [[1 if char == "#" else 0 for char in row] for row in constants.MAZE_LAYOUT]
    
    # Initialise results storage
    results = {
        'rl_scores': [],
        'rl_timesteps': [],
        'imitation_scores': [],
        'imitation_timesteps': [],
        'rl_convergence_points': [],
        'imitation_convergence_points': []
    }
    
    # Create output directories
    os.makedirs("outputs/convergence_tests", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/convergence_tests/{maze_type}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing demonstrations
    existing_demos_dir = "outputs/imitation_data"
    has_existing_demos = use_existing_demos and os.path.exists(existing_demos_dir)
    
    for run in range(num_runs):
        print(f"\n=== Run {run + 1}/{num_runs} ===")
        
        # Reset pellet grid for each run
        pellet_grid = [[True if char == '.' else False for char in row] 
                      for row in constants.MAZE_LAYOUT]
        
        # === RL Training ===
        print("\nTraining RL agent...")
        env = PacmanEnv(maze, pellet_grid, use_enhanced_features=True)
        env = PacmanMonitor(env, f"{output_dir}/rl_run_{run}")
        
        # Train RL agent
        model = train_dqn_model(
            maze, 
            pellet_grid, 
            timesteps=max_timesteps,
            force_retrain=True,
            log_dir=f"{output_dir}/rl_run_{run}"
        )
        
        # Evaluate RL agent
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        results['rl_scores'].append(mean_reward)
        results['rl_timesteps'].append(max_timesteps)
        
        # === Imitation Learning ===
        print("\nTraining Imitation agent...")
        
        # Handle demonstrations
        demo_dir = f"{output_dir}/expert_data_run_{run}"
        if has_existing_demos:
            print(f"Using existing demonstrations from {existing_demos_dir}")
            demo_dir = existing_demos_dir
        else:
            print(f"Collecting {num_demonstrations} new demonstrations...")
            collector = ImprovedDataCollector()
            expert_data = collector.collect_demonstrations(
                num_episodes=num_demonstrations,
                save_dir=demo_dir
            )
        
        # Train imitation agent
        imitation_agent = ImprovedImitationLearning(
            observation_dim=42,
            action_dim=4,
            use_enhanced_features=True
        )
        
        # Load and prepare demonstrations
        demonstration_data = imitation_agent.load_demonstrations(
            data_dir=demo_dir,
            filter_maze_type=maze_type
        )
        observations = demonstration_data[0]  # First element is observations
        actions = demonstration_data[1]      # Second element is actions
        
        # Train imitation model
        train_loader, val_loader, val_data = imitation_agent.prepare_datasets(
            observations, actions
        )
        
        imitation_agent.train(
            train_loader,
            val_loader,
            val_data,
            epochs=100  # Increased epochs for better training
        )
        
        # Convert to DQN format for evaluation
        imitation_model = imitation_agent.convert_to_dqn(
            save_path=f"{output_dir}/imitation_model_run_{run}"
        )
        
        # Evaluate imitation agent
        mean_reward, std_reward = evaluate_policy(imitation_model, env, n_eval_episodes=10)
        results['imitation_scores'].append(mean_reward)
        results['imitation_timesteps'].append(len(observations))
        
        # Save intermediate results
        save_results(results, output_dir)
        
        # Plot learning curves
        plot_learning_curves(results, output_dir, run)
    
    # Generate final analysis
    generate_analysis(results, output_dir)
    
    return results

def save_results(results, output_dir):
    """Save results to CSV file."""
    # Ensure all arrays have the same length
    max_length = max(len(results['rl_scores']), 
                    len(results['rl_timesteps']),
                    len(results['imitation_scores']),
                    len(results['imitation_timesteps']))
    
    # Pad shorter arrays with None
    for key in results:
        if len(results[key]) < max_length:
            results[key].extend([None] * (max_length - len(results[key])))
    
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/results.csv", index=False)

def plot_learning_curves(results, output_dir, run):
    """Plot learning curves for RL and imitation learning."""
    plt.figure(figsize=(12, 6))
    
    # Get valid data points (excluding None values)
    rl_timesteps = [t for t in results['rl_timesteps'][:run+1] if t is not None]
    rl_scores = [s for s in results['rl_scores'][:run+1] if s is not None]
    imitation_timesteps = [t for t in results['imitation_timesteps'][:run+1] if t is not None]
    imitation_scores = [s for s in results['imitation_scores'][:run+1] if s is not None]
    
    # Plot RL scores
    if rl_timesteps and rl_scores:
        plt.plot(rl_timesteps, rl_scores, 
                 label='RL Agent', color='blue', alpha=0.5)
    
    # Plot imitation scores
    if imitation_timesteps and imitation_scores:
        plt.plot(imitation_timesteps, imitation_scores, 
                 label='Imitation Agent', color='red', alpha=0.5)
    
    plt.xlabel('Training Samples')
    plt.ylabel('Average Score')
    plt.title(f'Learning Curves - Run {run + 1}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/learning_curves_run_{run}.png")
    plt.close()

def generate_analysis(results, output_dir):
    """Generate statistical analysis of the results."""
    # Filter out None values and get valid data
    rl_scores = [s for s in results['rl_scores'] if s is not None]
    rl_timesteps = [t for t in results['rl_timesteps'] if t is not None]
    imitation_scores = [s for s in results['imitation_scores'] if s is not None]
    imitation_timesteps = [t for t in results['imitation_timesteps'] if t is not None]
    
    # Calculate statistics only if we have valid data
    analysis = {}
    
    if rl_scores:
        analysis['RL Mean Score'] = np.mean(rl_scores)
        analysis['RL Score Std'] = np.std(rl_scores)
    else:
        analysis['RL Mean Score'] = 0
        analysis['RL Score Std'] = 0
    
    if rl_timesteps:
        analysis['RL Mean Samples'] = np.mean(rl_timesteps)
    else:
        analysis['RL Mean Samples'] = 0
    
    if imitation_scores:
        analysis['Imitation Mean Score'] = np.mean(imitation_scores)
        analysis['Imitation Score Std'] = np.std(imitation_scores)
    else:
        analysis['Imitation Mean Score'] = 0
        analysis['Imitation Score Std'] = 0
    
    if imitation_timesteps:
        analysis['Imitation Mean Samples'] = np.mean(imitation_timesteps)
    else:
        analysis['Imitation Mean Samples'] = 0
    
    # Calculate sample efficiency ratio safely
    if imitation_timesteps and rl_timesteps:
        analysis['Sample Efficiency Ratio'] = np.mean(rl_timesteps) / np.mean(imitation_timesteps)
    else:
        analysis['Sample Efficiency Ratio'] = 0
    
    # Save analysis
    with open(f"{output_dir}/analysis.txt", 'w') as f:
        for key, value in analysis.items():
            f.write(f"{key}: {value:.2f}\n")
    
    # Create comparison plots only if we have valid data
    if any(analysis.values()):
        # Plot 1: Performance vs Samples
        plt.figure(figsize=(10, 6))
        plt.scatter(analysis['RL Mean Samples'], analysis['RL Mean Score'], 
                   color='blue', label='RL Agent', s=100)
        plt.scatter(analysis['Imitation Mean Samples'], analysis['Imitation Mean Score'], 
                   color='red', label='Imitation Agent', s=100)
        
        plt.xlabel('Number of Training Samples')
        plt.ylabel('Mean Score')
        plt.title('Performance vs Sample Efficiency')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/performance_vs_samples.png")
        plt.close()
        
        # Plot 2: Sample Efficiency Ratio
        plt.figure(figsize=(15, 6))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Samples needed
        methods = ['RL Agent', 'Imitation Agent']
        samples = [analysis['RL Mean Samples'], analysis['Imitation Mean Samples']]
        scores = [analysis['RL Mean Score'], analysis['Imitation Mean Score']]
        
        bars1 = ax1.bar(methods, samples, color=['blue', 'red'])
        ax1.set_ylabel('Number of Training Samples')
        ax1.set_title('Training Samples Required')
        ax1.grid(True, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                    f'{int(height):,}',
                    ha='center', va='bottom')
        
        # Plot 2: Final scores
        bars2 = ax2.bar(methods, scores, color=['blue', 'red'])
        ax2.set_ylabel('Final Score')
        ax2.set_title('Final Performance')
        ax2.grid(True, axis='y')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        # Add efficiency ratio as text
        plt.figtext(0.5, 0.01, 
                    f'Sample Efficiency Ratio (RL/Imitation): {analysis["Sample Efficiency Ratio"]:.1f}x\n' +
                    f'This means RL requires {analysis["Sample Efficiency Ratio"]:.1f} times more samples than Imitation Learning',
                    ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sample_efficiency_ratio.png")
        plt.close()

if __name__ == "__main__":
    # Run tests for both simple and complex mazes
    print("Running convergence tests for simple maze...")
    simple_results = run_convergence_test(
        maze_type="SIMPLE",
        num_runs=5,  # Multiple runs for statistical significance
        max_timesteps=100000,  # Sufficient training time
        num_demonstrations=20,  # More demonstrations for better imitation learning
        use_existing_demos=True  # Use existing demos if available
    )
    
    print("\nRunning convergence tests for complex maze...")
    complex_results = run_convergence_test(
        maze_type="COMPLEX",
        num_runs=5,  # Multiple runs for statistical significance
        max_timesteps=100000,  # Sufficient training time
        num_demonstrations=20,  # More demonstrations for better imitation learning
        use_existing_demos=True  # Use existing demos if available
    ) 