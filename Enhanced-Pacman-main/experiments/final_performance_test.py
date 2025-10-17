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

def run_final_performance_test(maze_type="SIMPLE", num_runs=5, max_timesteps=100000, 
                             num_demonstrations=20, use_existing_demos=False):
    """Run comparative tests between RL and imitation learning agents to measure final performance in terms of score and survival time.
    
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
        'rl_survival_times': [],
        'imitation_scores': [],
        'imitation_survival_times': [],
        'rl_timesteps': [],
        'imitation_timesteps': []
    }
    
    # Create output directories
    os.makedirs("outputs/final_performance_tests", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/final_performance_tests/{maze_type}_{timestamp}"
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
        
        # Get survival time from monitor
        survival_time = env.get_survival_time()
        results['rl_survival_times'].append(survival_time)
        
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
        observations = demonstration_data[0]
        actions = demonstration_data[1]
        
        # Train imitation model
        train_loader, val_loader, val_data = imitation_agent.prepare_datasets(
            observations, actions
        )
        
        imitation_agent.train(
            train_loader,
            val_loader,
            val_data,
            epochs=100
        )
        
        # Convert to DQN format for evaluation
        imitation_model = imitation_agent.convert_to_dqn(
            save_path=f"{output_dir}/imitation_model_run_{run}"
        )
        
        # Evaluate imitation agent
        mean_reward, std_reward = evaluate_policy(imitation_model, env, n_eval_episodes=10)
        results['imitation_scores'].append(mean_reward)
        results['imitation_timesteps'].append(len(observations))
        
        # Get survival time from monitor
        survival_time = env.get_survival_time()
        results['imitation_survival_times'].append(survival_time)
        
        # Save intermediate results
        save_results(results, output_dir)
        
        # Plot performance curves
        plot_performance_curves(results, output_dir, run)
    
    # Generate final analysis
    generate_performance_analysis(results, output_dir)
    
    return results

def save_results(results, output_dir):
    """Save results to CSV file."""
    # Ensure all arrays have the same length
    max_length = max(len(results['rl_scores']), 
                    len(results['rl_survival_times']),
                    len(results['imitation_scores']),
                    len(results['imitation_survival_times']),
                    len(results['rl_timesteps']),
                    len(results['imitation_timesteps']))
    
    # Pad shorter arrays with None
    for key in results:
        if len(results[key]) < max_length:
            results[key].extend([None] * (max_length - len(results[key])))
    
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/results.csv", index=False)

def plot_performance_curves(results, output_dir, run):
    """Plot performance curves for RL and imitation learning."""
    plt.figure(figsize=(15, 10))
    
    # Plot scores
    plt.subplot(2, 1, 1)
    rl_scores = [s for s in results['rl_scores'][:run+1] if s is not None]
    imitation_scores = [s for s in results['imitation_scores'][:run+1] if s is not None]
    
    if rl_scores:
        plt.plot(range(len(rl_scores)), rl_scores, 
                 label='RL Agent', color='blue', alpha=0.5)
    if imitation_scores:
        plt.plot(range(len(imitation_scores)), imitation_scores, 
                 label='Imitation Agent', color='red', alpha=0.5)
    
    plt.xlabel('Run')
    plt.ylabel('Average Score')
    plt.title('Final Scores')
    plt.legend()
    plt.grid(True)
    
    # Plot survival times
    plt.subplot(2, 1, 2)
    rl_times = [t for t in results['rl_survival_times'][:run+1] if t is not None]
    imitation_times = [t for t in results['imitation_survival_times'][:run+1] if t is not None]
    
    if rl_times:
        plt.plot(range(len(rl_times)), rl_times, 
                 label='RL Agent', color='blue', alpha=0.5)
    if imitation_times:
        plt.plot(range(len(imitation_times)), imitation_times, 
                 label='Imitation Agent', color='red', alpha=0.5)
    
    plt.xlabel('Run')
    plt.ylabel('Average Survival Time')
    plt.title('Survival Times')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_curves_run_{run}.png")
    plt.close()

def generate_performance_analysis(results, output_dir):
    """Generate statistical analysis of the performance results."""
    # Filter out None values
    rl_scores = [s for s in results['rl_scores'] if s is not None]
    imitation_scores = [s for s in results['imitation_scores'] if s is not None]
    rl_times = [t for t in results['rl_survival_times'] if t is not None]
    imitation_times = [t for t in results['imitation_survival_times'] if t is not None]
    
    # Calculate statistics
    analysis = {
        'RL Mean Score': np.mean(rl_scores),
        'RL Score Std': np.std(rl_scores),
        'RL Mean Survival Time': np.mean(rl_times),
        'RL Time Std': np.std(rl_times),
        'Imitation Mean Score': np.mean(imitation_scores),
        'Imitation Score Std': np.std(imitation_scores),
        'Imitation Mean Survival Time': np.mean(imitation_times),
        'Imitation Time Std': np.std(imitation_times)
    }
    
    # Save analysis
    with open(f"{output_dir}/analysis.txt", 'w') as f:
        for key, value in analysis.items():
            f.write(f"{key}: {value:.2f}\n")
    
    # Create comparison plots
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Final Scores
    plt.subplot(1, 2, 1)
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, 
            [analysis['RL Mean Score'], analysis['RL Mean Survival Time']],
            width, label='RL Agent', color='blue')
    
    plt.bar(x + width/2, 
            [analysis['Imitation Mean Score'], analysis['Imitation Mean Survival Time']],
            width, label='Imitation Agent', color='red')
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Final Performance Comparison')
    plt.xticks(x, ['Score', 'Survival Time'])
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Standard Deviations
    plt.subplot(1, 2, 2)
    
    plt.bar(x - width/2, 
            [analysis['RL Score Std'], analysis['RL Time Std']],
            width, label='RL Agent', color='blue')
    
    plt.bar(x + width/2, 
            [analysis['Imitation Score Std'], analysis['Imitation Time Std']],
            width, label='Imitation Agent', color='red')
    
    plt.xlabel('Metric')
    plt.ylabel('Standard Deviation')
    plt.title('Performance Variability')
    plt.xticks(x, ['Score', 'Survival Time'])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png")
    plt.close()

if __name__ == "__main__":
    # Run tests for both simple and complex mazes
    print("Running final performance tests for simple maze...")
    simple_results = run_final_performance_test(
        maze_type="SIMPLE",
        num_runs=5,
        max_timesteps=100000,
        num_demonstrations=20,
        use_existing_demos=True
    )
    
    print("\nRunning final performance tests for complex maze...")
    complex_results = run_final_performance_test(
        maze_type="COMPLEX",
        num_runs=5,
        max_timesteps=100000,
        num_demonstrations=20,
        use_existing_demos=True
    ) 