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

def run_adaptability_test(maze_type="SIMPLE", num_runs=5, max_timesteps=100000, 
                         num_demonstrations=20, use_existing_demos=False):
    """Run comparative tests between RL and imitation learning agents to measure adaptability to new challenges.
    
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
        'rl_visibility_scores': [],  # Scores with limited visibility
        'imitation_visibility_scores': [],
        'rl_complex_scores': [],  # Scores on complex maze
        'imitation_complex_scores': []
    }
    
    # Create output directories
    os.makedirs("outputs/adaptability_tests", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/adaptability_tests/{maze_type}_{timestamp}"
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
        
        # Evaluate RL agent on original environment
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        results['rl_scores'].append(mean_reward)
        results['rl_timesteps'].append(max_timesteps)
        
        # Evaluate RL agent with limited visibility
        env_limited_vis = PacmanEnv(maze, pellet_grid, use_enhanced_features=True, 
                                  visibility_radius=2)  # Limited visibility
        mean_reward_vis, _ = evaluate_policy(model, env_limited_vis, n_eval_episodes=10)
        results['rl_visibility_scores'].append(mean_reward_vis)
        
        # Evaluate RL agent on complex maze
        constants.set_maze_type("COMPLEX")
        complex_maze = [[1 if char == "#" else 0 for char in row] for row in constants.MAZE_LAYOUT]
        complex_pellet_grid = [[True if char == '.' else False for char in row] 
                             for row in constants.MAZE_LAYOUT]
        env_complex = PacmanEnv(complex_maze, complex_pellet_grid, use_enhanced_features=True)
        mean_reward_complex, _ = evaluate_policy(model, env_complex, n_eval_episodes=10)
        results['rl_complex_scores'].append(mean_reward_complex)
        
        # Reset maze type back to original
        constants.set_maze_type(maze_type)
        
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
        
        # Evaluate imitation agent on original environment
        mean_reward, std_reward = evaluate_policy(imitation_model, env, n_eval_episodes=10)
        results['imitation_scores'].append(mean_reward)
        results['imitation_timesteps'].append(len(observations))
        
        # Evaluate imitation agent with limited visibility
        mean_reward_vis, _ = evaluate_policy(imitation_model, env_limited_vis, n_eval_episodes=10)
        results['imitation_visibility_scores'].append(mean_reward_vis)
        
        # Evaluate imitation agent on complex maze
        mean_reward_complex, _ = evaluate_policy(imitation_model, env_complex, n_eval_episodes=10)
        results['imitation_complex_scores'].append(mean_reward_complex)
        
        # Save intermediate results
        save_results(results, output_dir)
        
        # Plot learning curves
        plot_adaptability_curves(results, output_dir, run)
    
    # Generate final analysis
    generate_adaptability_analysis(results, output_dir)
    
    return results

def save_results(results, output_dir):
    """Save results to CSV file."""
    # Ensure all arrays have the same length
    max_length = max(len(results['rl_scores']), 
                    len(results['rl_timesteps']),
                    len(results['imitation_scores']),
                    len(results['imitation_timesteps']),
                    len(results['rl_visibility_scores']),
                    len(results['imitation_visibility_scores']),
                    len(results['rl_complex_scores']),
                    len(results['imitation_complex_scores']))
    
    # Pad shorter arrays with None
    for key in results:
        if len(results[key]) < max_length:
            results[key].extend([None] * (max_length - len(results[key])))
    
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/results.csv", index=False)

def plot_adaptability_curves(results, output_dir, run):
    """Plot adaptability curves for RL and imitation learning."""
    plt.figure(figsize=(15, 10))
    
    # Plot original environment scores
    plt.subplot(2, 2, 1)
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
    plt.title('Original Environment')
    plt.legend()
    plt.grid(True)
    
    # Plot limited visibility scores
    plt.subplot(2, 2, 2)
    rl_vis_scores = [s for s in results['rl_visibility_scores'][:run+1] if s is not None]
    imitation_vis_scores = [s for s in results['imitation_visibility_scores'][:run+1] if s is not None]
    
    if rl_vis_scores:
        plt.plot(range(len(rl_vis_scores)), rl_vis_scores, 
                 label='RL Agent', color='blue', alpha=0.5)
    if imitation_vis_scores:
        plt.plot(range(len(imitation_vis_scores)), imitation_vis_scores, 
                 label='Imitation Agent', color='red', alpha=0.5)
    
    plt.xlabel('Run')
    plt.ylabel('Average Score')
    plt.title('Limited Visibility')
    plt.legend()
    plt.grid(True)
    
    # Plot complex maze scores
    plt.subplot(2, 2, 3)
    rl_complex_scores = [s for s in results['rl_complex_scores'][:run+1] if s is not None]
    imitation_complex_scores = [s for s in results['imitation_complex_scores'][:run+1] if s is not None]
    
    if rl_complex_scores:
        plt.plot(range(len(rl_complex_scores)), rl_complex_scores, 
                 label='RL Agent', color='blue', alpha=0.5)
    if imitation_complex_scores:
        plt.plot(range(len(imitation_complex_scores)), imitation_complex_scores, 
                 label='Imitation Agent', color='red', alpha=0.5)
    
    plt.xlabel('Run')
    plt.ylabel('Average Score')
    plt.title('Complex Maze')
    plt.legend()
    plt.grid(True)
    
    # Plot adaptability ratios
    plt.subplot(2, 2, 4)
    if rl_scores and rl_vis_scores and rl_complex_scores:
        rl_vis_ratio = np.mean(rl_vis_scores) / np.mean(rl_scores)
        rl_complex_ratio = np.mean(rl_complex_scores) / np.mean(rl_scores)
        plt.bar(['Limited Visibility', 'Complex Maze'], 
                [rl_vis_ratio, rl_complex_ratio], 
                color='blue', alpha=0.5, label='RL Agent')
    
    if imitation_scores and imitation_vis_scores and imitation_complex_scores:
        imitation_vis_ratio = np.mean(imitation_vis_scores) / np.mean(imitation_scores)
        imitation_complex_ratio = np.mean(imitation_complex_scores) / np.mean(imitation_scores)
        plt.bar(['Limited Visibility', 'Complex Maze'], 
                [imitation_vis_ratio, imitation_complex_ratio], 
                color='red', alpha=0.5, label='Imitation Agent')
    
    plt.xlabel('Challenge Type')
    plt.ylabel('Performance Ratio (Challenge/Original)')
    plt.title('Adaptability Ratios')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/adaptability_curves_run_{run}.png")
    plt.close()

def generate_adaptability_analysis(results, output_dir):
    """Generate statistical analysis of the adaptability results."""
    # Filter out None values
    rl_scores = [s for s in results['rl_scores'] if s is not None]
    imitation_scores = [s for s in results['imitation_scores'] if s is not None]
    rl_vis_scores = [s for s in results['rl_visibility_scores'] if s is not None]
    imitation_vis_scores = [s for s in results['imitation_visibility_scores'] if s is not None]
    rl_complex_scores = [s for s in results['rl_complex_scores'] if s is not None]
    imitation_complex_scores = [s for s in results['imitation_complex_scores'] if s is not None]
    
    # Calculate statistics
    analysis = {
        'RL Original Score': np.mean(rl_scores),
        'RL Original Std': np.std(rl_scores),
        'RL Visibility Score': np.mean(rl_vis_scores),
        'RL Visibility Std': np.std(rl_vis_scores),
        'RL Complex Score': np.mean(rl_complex_scores),
        'RL Complex Std': np.std(rl_complex_scores),
        'Imitation Original Score': np.mean(imitation_scores),
        'Imitation Original Std': np.std(imitation_scores),
        'Imitation Visibility Score': np.mean(imitation_vis_scores),
        'Imitation Visibility Std': np.std(imitation_vis_scores),
        'Imitation Complex Score': np.mean(imitation_complex_scores),
        'Imitation Complex Std': np.std(imitation_complex_scores)
    }
    
    # Calculate adaptability ratios
    if rl_scores and rl_vis_scores and rl_complex_scores:
        analysis['RL Visibility Ratio'] = np.mean(rl_vis_scores) / np.mean(rl_scores)
        analysis['RL Complex Ratio'] = np.mean(rl_complex_scores) / np.mean(rl_scores)
    
    if imitation_scores and imitation_vis_scores and imitation_complex_scores:
        analysis['Imitation Visibility Ratio'] = np.mean(imitation_vis_scores) / np.mean(imitation_scores)
        analysis['Imitation Complex Ratio'] = np.mean(imitation_complex_scores) / np.mean(imitation_scores)
    
    # Save analysis
    with open(f"{output_dir}/analysis.txt", 'w') as f:
        for key, value in analysis.items():
            f.write(f"{key}: {value:.2f}\n")
    
    # Create comparison plots
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Original vs Challenging Environments
    plt.subplot(1, 2, 1)
    x = np.arange(3)
    width = 0.35
    
    plt.bar(x - width/2, 
            [analysis['RL Original Score'], 
             analysis['RL Visibility Score'], 
             analysis['RL Complex Score']],
            width, label='RL Agent', color='blue')
    
    plt.bar(x + width/2, 
            [analysis['Imitation Original Score'], 
             analysis['Imitation Visibility Score'], 
             analysis['Imitation Complex Score']],
            width, label='Imitation Agent', color='red')
    
    plt.xlabel('Environment')
    plt.ylabel('Average Score')
    plt.title('Performance Across Environments')
    plt.xticks(x, ['Original', 'Limited Visibility', 'Complex Maze'])
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Adaptability Ratios
    plt.subplot(1, 2, 2)
    x = np.arange(2)
    
    plt.bar(x - width/2, 
            [analysis['RL Visibility Ratio'], 
             analysis['RL Complex Ratio']],
            width, label='RL Agent', color='blue')
    
    plt.bar(x + width/2, 
            [analysis['Imitation Visibility Ratio'], 
             analysis['Imitation Complex Ratio']],
            width, label='Imitation Agent', color='red')
    
    plt.xlabel('Challenge Type')
    plt.ylabel('Performance Ratio (Challenge/Original)')
    plt.title('Adaptability Ratios')
    plt.xticks(x, ['Limited Visibility', 'Complex Maze'])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/adaptability_comparison.png")
    plt.close()

if __name__ == "__main__":
    # Run tests for both simple and complex mazes
    print("Running adaptability tests for simple maze...")
    simple_results = run_adaptability_test(
        maze_type="SIMPLE",
        num_runs=5,
        max_timesteps=100000,
        num_demonstrations=20,
        use_existing_demos=True
    )
    
    print("\nRunning adaptability tests for complex maze...")
    complex_results = run_adaptability_test(
        maze_type="COMPLEX",
        num_runs=5,
        max_timesteps=100000,
        num_demonstrations=20,
        use_existing_demos=True
    ) 