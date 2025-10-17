# dqn/testing.py
import pygame
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import json

from dqn.training import load_dqn_model
from dqn.environment import PacmanEnv
import constants

def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialisation"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def test_dqn_model(maze, pellet_grid, model=None, num_episodes=100, render=False,
                   use_enhanced_features=True, model_path="outputs/models/pacman_dqn",
                   save_results=True, results_dir="./outputs/test_results"):
    """
    Test DQN model for specified number of episodes and log results

    Parameters:
        maze: Maze grid
        pellet_grid: Initial pellet grid
        model: Pre-loaded DQN model (if None, will load from model_path)
        num_episodes: Number of episodes to run
        render: Whether to render game during testing
        use_enhanced_features: Whether to use enhanced features (must match model)
        model_path: Path to load model from if not provided
        save_results: Whether to save results to file
        results_dir: Directory to save results

    Returns:
        results_df: DataFrame with results of all episodes
    """
    # Generate timestamp for folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory if needed
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        run_dir = os.path.join(results_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = results_dir

    # Load model if not provided
    if model is None:
        model = load_dqn_model(model_path=model_path, strict=True)

    # Create testing environment
    env = PacmanEnv(maze, pellet_grid, use_enhanced_features=use_enhanced_features)

    # Initialise Pygame if rendering
    if render:
        pygame.init()
        screen = pygame.display.set_mode((constants.WIDTH, constants.HEIGHT))
        pygame.display.set_caption("PacMan DQN Testing")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 36)

    # Store results
    results = []

    print(f"Starting DQN model testing for {num_episodes} episodes...")
    for episode in range(1, num_episodes + 1):
        # Reset environment
        observation, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        step_count = 0
        pellets_eaten = 0
        initial_pellets = sum(sum(row) for row in env.pellet_grid)
        start_time = time.time()

        # Run episode
        while not done and not truncated:
            # Get action from model
            action, _states = model.predict(observation, deterministic=True)

            # Take action in environment
            observation, reward, done, truncated, info = env.step(action)

            # Update metrics
            total_reward += reward
            step_count += 1

            # Calculate pellets eaten
            current_pellets = sum(sum(row) for row in env.pellet_grid)
            pellets_eaten = initial_pellets - current_pellets

            # Handle rendering
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return pd.DataFrame(results)

                # Draw environment
                screen.fill((0, 0, 0))

                # Draw maze
                for row in range(constants.ROWS):
                    for col in range(constants.COLS):
                        if maze[row][col] == 1:
                            rect = pygame.Rect(
                                col * constants.TILE_SIZE,
                                row * constants.TILE_SIZE,
                                constants.TILE_SIZE,
                                constants.TILE_SIZE
                            )
                            pygame.draw.rect(screen, (0, 0, 255), rect)

                # Draw pellets
                for row in range(constants.ROWS):
                    for col in range(constants.COLS):
                        if env.pellet_grid[row][col]:
                            centre_x = col * constants.TILE_SIZE + constants.TILE_SIZE // 2
                            centre_y = row * constants.TILE_SIZE + constants.TILE_SIZE // 2
                            pygame.draw.circle(screen, (255, 255, 255), (centre_x, centre_y), 4)

                # Draw pacman and ghosts
                env.pacman.draw(screen, font)
                for ghost in env.active_ghosts:
                    ghost.draw(screen)

                # Draw UI information
                score_text = font.render(f"Score: {info['score']}", True, (255, 255, 255))
                episode_text = font.render(f"Episode: {episode}/{num_episodes}", True, (255, 255, 255))
                step_text = font.render(f"Steps: {step_count}", True, (255, 255, 255))

                screen.blit(score_text, (10, 10))
                screen.blit(episode_text, (constants.WIDTH - 200, 10))
                screen.blit(step_text, (10, 50))

                pygame.display.flip()
                clock.tick(30)  # Slower speed for visualisation

        # Calculate episode metrics
        episode_time = time.time() - start_time
        won = info.get('won', False)

        # Store results
        results.append({
            'episode': episode,
            'score': info['score'],
            'steps': step_count,
            'pellets_eaten': pellets_eaten,
            'total_pellets': initial_pellets,
            'completion_rate': pellets_eaten / initial_pellets if initial_pellets > 0 else 0,
            'reward': total_reward,
            'won': won,
            'time': episode_time
        })

        # Print progress
        status = "WIN" if won else "LOSS"
        print(f"Episode {episode}/{num_episodes} - Score: {info['score']} - Steps: {step_count} - Status: {status}")

    # Clean up pygame
    if render:
        pygame.quit()

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    if save_results and not results_df.empty:
        # Save raw data to CSV
        csv_path = os.path.join(run_dir, "dqn_test_results.csv")
        results_df.to_csv(csv_path, index=False)

        # Save summary statistics to JSON
        summary = {
            'total_episodes': num_episodes,
            'win_rate': results_df['won'].mean(),
            'avg_score': results_df['score'].mean(),
            'avg_steps': results_df['steps'].mean(),
            'avg_completion_rate': results_df['completion_rate'].mean(),
            'avg_reward': results_df['reward'].mean(),
            'max_score': results_df['score'].max(),
            'min_score': results_df['score'].min(),
            'timestamp': timestamp,
            'model_path': model_path,
            'enhanced_features': use_enhanced_features
        }

        summary_path = os.path.join(run_dir, "dqn_test_summary.json")
        # Convert numpy types to Python native types
        for key in summary:
            summary[key] = convert_to_json_serializable(summary[key])

        # Save to JSON
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        # Generate and save plots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot score distribution
        axs[0, 0].hist(results_df['score'], bins=20, alpha=0.7)
        axs[0, 0].set_title('Score Distribution')
        axs[0, 0].set_xlabel('Score')
        axs[0, 0].set_ylabel('Frequency')

        # Plot steps distribution
        axs[0, 1].hist(results_df['steps'], bins=20, alpha=0.7)
        axs[0, 1].set_title('Steps Distribution')
        axs[0, 1].set_xlabel('Steps')
        axs[0, 1].set_ylabel('Frequency')

        # Plot completion rate
        axs[1, 0].hist(results_df['completion_rate'], bins=10, alpha=0.7)
        axs[1, 0].set_title('Completion Rate Distribution')
        axs[1, 0].set_xlabel('Completion Rate')
        axs[1, 0].set_ylabel('Frequency')

        # Plot score over episodes
        axs[1, 1].plot(results_df['episode'], results_df['score'])
        axs[1, 1].set_title('Score Over Episodes')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Score')

        plt.tight_layout()
        plot_path = os.path.join(run_dir, "dqn_test_plots.png")
        plt.savefig(plot_path)

        print(f"Results saved to {run_dir}")
        print(f"Win rate: {summary['win_rate']:.2%}")
        print(f"Average score: {summary['avg_score']:.2f}")
        print(f"Average completion rate: {summary['avg_completion_rate']:.2%}")

    return results_df

