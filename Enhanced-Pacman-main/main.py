import pygame
import sys
import random
import numpy as np
import os
import torch
from collections import defaultdict, deque  # <--- 在这里添加这一行
import constants
from utils.helpers import collides_with_wall, pixel_to_grid, grid_to_pixel, check_collision_pacman_ghost
from utils.a_star import a_star
from entities.pacman import Pacman
from entities.ghost import Ghost
from rendering.renderer import render_game
from dqn.training import train_dqn_model, load_dqn_model
from dqn.testing import test_dqn_model
from imitation.data_collector import ImprovedDataCollector
from imitation.behavioral_cloning import ImprovedImitationLearning

# tensorboard --logdir=./outputs/logs/tensorboard

pygame.init()

# ----- Game Configuration -----
TRAINING_MODE = False  # Set to True to train a new model
TESTING_MODE = True# Set to True to run automated testing

# Imitation learning settings
IMITATION_MODE = False  # Set to True to use imitation learning model
RECORD_DEMONSTRATIONS = False  # Set to True to record expert demonstrations
TRAIN_IMITATION_MODEL = False  # Set to True to train a new imitation model

# Training parameters
TRAINING_TIMESTEPS = 500000  # Number of steps to train for
USE_ENHANCED_FEATURES = True  # Whether to use enhanced features for training

# Imitation learning parameters
IMITATION_EPOCHS = 150  # Number of epochs to train for
IMITATION_LEARNING_RATE = 0.0001  # Learning rate for training
IMITATION_BATCH_SIZE = 128  # Batch size for training

# Testing parameters
TEST_EPISODES = 30  # Number of episodes to test
TEST_RENDER = True # Whether to render the game during testing
# MODEL_PATHS = ["outputs/models/pacman_dqn", "outputs/models/logs/models/dqn_pacman_simple_checkpoint_250000_steps"]
MODEL_PATHS = ["outputs/models/pacman_dqn", "outputs/models/logs/models/dqn_pacman_simple_checkpoint_100000_steps"]
MODEL_NAMES = ["Final Model", "Checkpoint 40k"]

# Maze configuration
MAZE_TYPE = "SIMPLE"  # "COMPLEX" or "SIMPLE"
constants.set_maze_type(MAZE_TYPE)

def convert_maze_to_grid(maze_layout):
    """Convert maze layout to a grid of walls (1) and open spaces (0)"""
    maze = []
    for row in maze_layout:
        row_list = []
        for char in row:
            row_list.append(1 if char == "#" else 0)
        maze.append(row_list)
    return maze

# Initialise maze grid
maze = convert_maze_to_grid(constants.MAZE_LAYOUT)

def reset_pellet_grid():
    """Reset pellet grid to initial state"""
    pellet_grid = []
    for row in constants.MAZE_LAYOUT:
        pellet_row = []
        for char in row:
            # In the simple maze, 'G' marks ghost positions - no pellets there
            pellet_row.append(True if char == '.' else False)
        pellet_grid.append(pellet_row)
    return pellet_grid

# Initialise game state
pellet_grid = reset_pellet_grid()
score = 0

# Set up display
screen = pygame.display.set_mode((constants.WIDTH, constants.HEIGHT))
pygame.display.set_caption("PacMan with DQN")
clock = pygame.time.Clock()
score_font = pygame.font.SysFont(None, 36)

def init_game_objects():
    """Initialise and return game objects"""
    # Get spawn positions based on current maze
    pacman_spawn_pos = constants.get_pacman_spawn()
    pacman_spawn = grid_to_pixel(*pacman_spawn_pos)
    pacman = Pacman(*pacman_spawn)

    ghost_info = constants.get_ghost_house_info()
    ghost_spawn_pos = ghost_info["spawn"]
    ghost_spawn = grid_to_pixel(*ghost_spawn_pos)

    # Define ghost colours (Red, Pink, Cyan, Orange)
    ghost_colours = [(255, 0, 0), (255, 184, 255), (0, 255, 255), (255, 184, 82)]

    # Map each ghost colour to its scatter target
    ghost_targets = {
        (255, 0, 0): grid_to_pixel(constants.COLS - 1, 0),       # Red: Top-right corner
        (255, 184, 255): grid_to_pixel(0, 0),                    # Pink: Top-left corner
        (0, 255, 255): grid_to_pixel(constants.COLS - 1, constants.ROWS - 1), # Cyan: Bottom-right corner
        (255, 184, 82): grid_to_pixel(0, constants.ROWS - 1)     # Orange: Bottom-left corner
    }

    # Create active ghosts list
    active_ghosts = []
    ghosts_to_release = []
    
    if constants.CURRENT_MAZE_TYPE == "SIMPLE":
        ghost_count = min(constants.SIMPLE_MAZE_GHOST_COUNT, len(ghost_colours))
        
        # Place first ghost in original position
        top_right_pos = grid_to_pixel(constants.COLS - 3, 3)
        active_ghosts.append(
            Ghost(top_right_pos[0], top_right_pos[1], ghost_colours[0], 
                 scatter_target=ghost_targets[ghost_colours[0]])
        )
        
        if ghost_count > 1:
            bottom_left_pos = grid_to_pixel(3, constants.ROWS - 3)
            active_ghosts.append(
                Ghost(bottom_left_pos[0], bottom_left_pos[1], ghost_colours[1], 
                     scatter_target=ghost_targets[ghost_colours[1]])
            )
        
        for ghost in active_ghosts:
            ghost.exited = True
    else:
        active_ghosts.append(
            Ghost(ghost_spawn[0], ghost_spawn[1], ghost_colours[0], 
                 scatter_target=ghost_targets[ghost_colours[0]])
        )
        
        for i in range(1, len(ghost_colours)):
            ghosts_to_release.append(
                Ghost(ghost_spawn[0], ghost_spawn[1], ghost_colours[i], 
                     scatter_target=ghost_targets[ghost_colours[i]])
            )
    
    return pacman, active_ghosts, ghosts_to_release

def switch_maze(new_maze_type):
    """Switch between maze types and reset the game"""
    global maze, pellet_grid
    
    constants.set_maze_type(new_maze_type)
    maze = convert_maze_to_grid(constants.MAZE_LAYOUT)
    pellet_grid = reset_pellet_grid()
    pygame.display.set_mode((constants.WIDTH, constants.HEIGHT))
    
    return init_game_objects()

# ----- Main Game Setup -----
def main():
    global TRAINING_MODE, TRAINING_TIMESTEPS, USE_ENHANCED_FEATURES
    global TESTING_MODE, TEST_EPISODES, TEST_RENDER, COMPARE_MODELS
    global MAZE_TYPE, maze, pellet_grid, score, IMITATION_MODE  
    
    # Initialise game objects
    pacman, active_ghosts, ghosts_to_release = init_game_objects()
    
    # Set up ghost release timing
    release_interval = 3000  # Release one ghost every 3 seconds
    last_release_time = pygame.time.get_ticks()

    # Game state
    running = True
    game_over = False
    game_won = False
    score = 0
    restart_button = None

    # Initialise model variables
    dqn_model = None
    imitation_model = None
    
    # Handle DQN model loading or training
    if constants.GAME_MODE == "DQN":
        if TRAINING_MODE:
            print(f"Training mode enabled. Training a new model with {TRAINING_TIMESTEPS} timesteps...")
            print(f"Current maze type: {MAZE_TYPE}")
            dqn_model = train_dqn_model(
                maze=maze,
                pellet_grid=pellet_grid,
                timesteps=TRAINING_TIMESTEPS, 
                force_retrain=True,
                use_enhanced_features=USE_ENHANCED_FEATURES
            )
            if dqn_model:
                print("Training complete! New model will be used for gameplay.")
                pacman.dqn_model = dqn_model
            else:
                print("Training failed! Switching to A_STAR mode.")
                constants.GAME_MODE = "A_STAR"
        else:
            print("Loading existing model...")
            dqn_model = load_dqn_model()
            if dqn_model:
                print("Model loaded successfully!")
                pacman.dqn_model = dqn_model
            else:
                print("ERROR: No model found. Please train a model first by setting TRAINING_MODE = True")
                print("Exiting game.")
                pygame.quit()
                sys.exit()
    
    # Handle imitation learning setup
    data_collector = None

    if RECORD_DEMONSTRATIONS:
        data_collector = ImprovedDataCollector(maze=maze, pellet_grid=pellet_grid)
        print("Demonstration recording mode active. Use human mode (1) to play and record.")

    if TRAIN_IMITATION_MODEL:
        print("Training new imitation learning model...")
        
        try:
            from imitation.train_imitation import train_imitation_model
            
            imitation_learner, imitation_model = train_imitation_model(
                data_dir="outputs/imitation_data",
                maze_type=constants.CURRENT_MAZE_TYPE,
                observation_dim=42 if USE_ENHANCED_FEATURES else 18,
                use_enhanced_features=USE_ENHANCED_FEATURES,
                epochs=IMITATION_EPOCHS,
                batch_size=IMITATION_BATCH_SIZE,
                learning_rate=IMITATION_LEARNING_RATE
            )
            
            if imitation_model is None:
                print("Failed to train imitation model. Make sure you have recorded demonstrations first.")
                IMITATION_MODE = False
            else:
                print("Imitation learning model trained successfully!")
                if IMITATION_MODE:
                    constants.GAME_MODE = "IMITATION"
        except Exception as e:
            print(f"Error training imitation model: {e}")
            print("Make sure you have recorded demonstrations first (RECORD_DEMONSTRATIONS = True)")
            IMITATION_MODE = False

    # Load existing imitation model if not training a new one
    if IMITATION_MODE and not imitation_model:
        try:
            print("Loading imitation learning model...")
            
            model_path = f"outputs/models/imitation_learning_{constants.CURRENT_MAZE_TYPE.lower()}.pth"
            
            if os.path.exists(model_path):
                print(f"Found imitation model at {model_path}")
                imitation_learner = ImprovedImitationLearning()
                imitation_learner.load_model(model_path)
                
                test_obs = np.random.random(imitation_learner.observation_dim).astype(np.float32)
                test_action, test_probs = imitation_learner.predict(test_obs)
                print(f"Model loaded - test prediction: {test_action}")
                
                dqn_path = f"outputs/models/pacman_dqn_imitation_{constants.CURRENT_MAZE_TYPE.lower()}"
                if os.path.exists(dqn_path + ".zip"):
                    print(f"Loading DQN version of the imitation model from {dqn_path}")
                    imitation_model = load_dqn_model(model_path=dqn_path, strict=False)
                    
                    if imitation_model:
                        print("Imitation DQN model loaded successfully!")
                        pacman.imitation_model = imitation_model
                        if IMITATION_MODE:
                            constants.GAME_MODE = "IMITATION"
                    else:
                        print("Failed to load DQN version of imitation model.")
                        print("Will try to use the native imitation model.")
                        pacman.imitation_learner = imitation_learner
                        if IMITATION_MODE:
                            constants.GAME_MODE = "IMITATION"
                else:
                    print(f"DQN version not found at {dqn_path}")
                    print("Will use the native imitation model.")
                    pacman.imitation_learner = imitation_learner
                    if IMITATION_MODE:
                        constants.GAME_MODE = "IMITATION"
            else:
                print(f"No imitation model found at {model_path}")
                print("You need to train a model first (TRAIN_IMITATION_MODEL = True)")
                IMITATION_MODE = False
        except Exception as e:
            print(f"Error loading imitation model: {e}")
            print("You need to train an imitation model first (TRAIN_IMITATION_MODEL = True)")
            IMITATION_MODE = False

    # Handle testing mode
    if TESTING_MODE and constants.GAME_MODE == "DQN" and dqn_model:
        print("\n=== TESTING MODE ACTIVATED ===")
        print(f"Current maze type: {MAZE_TYPE}")
        print(f"Running {TEST_EPISODES} test episodes...")
        
        results = test_dqn_model(
            maze=maze,
            pellet_grid=pellet_grid,
            model=dqn_model,
            num_episodes=TEST_EPISODES,
            render=TEST_RENDER,
            use_enhanced_features=USE_ENHANCED_FEATURES
        )
        
        if not TEST_RENDER:
            print("Testing complete! Exiting...")
            pygame.quit()
            sys.exit()

    # Initialise A* path for Pac-Man
    if constants.GAME_MODE == "A_STAR":
        for r in range(constants.ROWS):
            for c in range(constants.COLS):
                if pellet_grid[r][c]:
                    pacman.target_pellet = (c, r)
                    pacman.path = a_star(pixel_to_grid(pacman.x, pacman.y), (c, r), maze)
                    if len(pacman.path) > 1:
                        next_cell = pacman.path[1]
                        current_cell = pacman.path[0]
                        dx = next_cell[0] - current_cell[0]
                        dy = next_cell[1] - current_cell[1]
                        pacman.direction = pygame.Vector2(dx, dy)
                        pacman.desired_direction = pygame.Vector2(dx, dy)
                    break
            if pacman.target_pellet:
                break

    # Function to restart the game
    def restart_game():
        """Restart the game with current settings"""
        nonlocal pacman, active_ghosts, ghosts_to_release, game_over, game_won
        nonlocal last_release_time, restart_button
        global pellet_grid, score

        game_over = False
        game_won = False
        score = 0
        restart_button = None
        
        pellet_grid = reset_pellet_grid()
        pacman, active_ghosts, ghosts_to_release = init_game_objects()
        
        if constants.GAME_MODE == "DQN" and dqn_model:
            pacman.dqn_model = dqn_model
        elif constants.GAME_MODE == "IMITATION" and imitation_model:
            pacman.imitation_model = imitation_model
        elif constants.GAME_MODE == "IMITATION" and hasattr(pacman, 'imitation_learner'):
            pacman.imitation_learner = imitation_learner
        
        last_release_time = pygame.time.get_ticks()
        
        if constants.GAME_MODE == "A_STAR":
            for r in range(constants.ROWS):
                for c in range(constants.COLS):
                    if pellet_grid[r][c]:
                        pacman.target_pellet = (c, r)
                        pacman.path = a_star(pixel_to_grid(pacman.x, pacman.y), (c, r), maze)
                        if len(pacman.path) > 1:
                            next_cell = pacman.path[1]
                            current_cell = pacman.path[0]
                            dx = next_cell[0] - current_cell[0]
                            dy = next_cell[1] - current_cell[1]
                            pacman.direction = pygame.Vector2(dx, dy)
                            pacman.desired_direction = pygame.Vector2(dx, dy)
                        break
                if pacman.target_pellet:
                    break

    # Main game loop
    while running:
        clock.tick(60)  # 60 FPS
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and (game_over or game_won):
                if restart_button and restart_button.collidepoint(event.pos):
                    restart_game()
            
            if event.type == pygame.KEYDOWN:
                if (game_over or game_won) and event.key == pygame.K_SPACE:
                    restart_game()
                
                elif not game_over and not game_won:
                    if event.key == pygame.K_1:
                        constants.GAME_MODE = "HUMAN"
                        print("Switched to HUMAN mode")
                    elif event.key == pygame.K_2:
                        constants.GAME_MODE = "A_STAR"
                        print("Switched to A_STAR mode")
                    elif event.key == pygame.K_3:
                        constants.GAME_MODE = "DQN"
                        print("Switched to DQN mode")
                        if not hasattr(pacman, 'dqn_model') or pacman.dqn_model is None:
                            dqn_model = load_dqn_model()
                            if dqn_model:
                                pacman.dqn_model = dqn_model
                            else:
                                print("Failed to load DQN model, staying in current mode")
                                constants.GAME_MODE = "A_STAR"
                    elif event.key == pygame.K_4:
                        if imitation_model or hasattr(pacman, 'imitation_learner'):
                            constants.GAME_MODE = "IMITATION"
                            print("Switched to IMITATION mode")
                            if imitation_model and (not hasattr(pacman, 'imitation_model') or pacman.imitation_model is None):
                                pacman.imitation_model = imitation_model
                        else:
                            print("No imitation model available")
                            
                    elif event.key == pygame.K_m:
                        if constants.CURRENT_MAZE_TYPE == "COMPLEX":
                            MAZE_TYPE = "SIMPLE"
                        else:
                            MAZE_TYPE = "COMPLEX"
                        
                        print(f"Switching to {MAZE_TYPE} maze")
                        pacman, active_ghosts, ghosts_to_release = switch_maze(MAZE_TYPE)
                        
                        if constants.GAME_MODE == "DQN" and dqn_model:
                            pacman.dqn_model = dqn_model
                        elif constants.GAME_MODE == "IMITATION" and imitation_model:
                            pacman.imitation_model = imitation_model
                        elif constants.GAME_MODE == "IMITATION" and hasattr(pacman, 'imitation_learner'):
                            pacman.imitation_learner = imitation_learner
                    
                    if RECORD_DEMONSTRATIONS and data_collector:
                        if event.key == pygame.K_r and constants.GAME_MODE == "HUMAN":
                            if not data_collector.is_recording:
                                data_collector.start_recording(pacman, active_ghosts)
                            else:
                                print("Already recording!")
                                
                        elif event.key == pygame.K_t:
                            if data_collector.is_recording:
                                data_collector.stop_recording(score=score, won=game_won)
                            else:
                                print("Not currently recording!")
                        
                        elif event.key == pygame.K_c and data_collector.is_recording:
                            data_collector.stop_recording(cancelled=True)
                            
                        elif event.key == pygame.K_a and not data_collector.is_recording:
                            data_collector.analyze_all_demonstrations()
                        
                    if constants.GAME_MODE == "HUMAN":
                        if event.key == pygame.K_UP:
                            pacman.desired_direction = pygame.Vector2(0, -1)
                        elif event.key == pygame.K_DOWN:
                            pacman.desired_direction = pygame.Vector2(0, 1)
                        elif event.key == pygame.K_LEFT:
                            pacman.desired_direction = pygame.Vector2(-1, 0)
                        elif event.key == pygame.K_RIGHT:
                            pacman.desired_direction = pygame.Vector2(1, 0)
        
        # Release ghosts in complex maze
        current_time = pygame.time.get_ticks()
        if constants.CURRENT_MAZE_TYPE == "COMPLEX" and ghosts_to_release and current_time - last_release_time >= release_interval:
            active_ghosts.append(ghosts_to_release.pop(0))
            last_release_time = current_time
        
        # Update game objects
        if not game_over and not game_won:
            # ==================  ==================
            # Check for pellet collection
            pacman_col = int(pacman.x // constants.TILE_SIZE)
            pacman_row = int(pacman.y // constants.TILE_SIZE)
            if 0 <= pacman_row < constants.ROWS and 0 <= pacman_col < constants.COLS:
                if pellet_grid[pacman_row][pacman_col]:
                    pellet_grid[pacman_row][pacman_col] = False
                    score += 10
            # =================================================================
            if constants.GAME_MODE == "DQN" and pacman.dqn_model:
                pacman.update(maze, active_ghosts, pellet_grid, score)
            elif constants.GAME_MODE == "IMITATION":
                if hasattr(pacman, 'imitation_model') and pacman.imitation_model:
                    try:
                        obs = pacman._create_observation(active_ghosts, pellet_grid, maze)
                        
                        if isinstance(obs, np.ndarray) and len(obs.shape) == 1:
                            obs = obs.reshape(1, -1)
                        
                        with torch.no_grad():
                            action, _ = pacman.imitation_model.predict(obs, deterministic=True)
                            
                            if hasattr(action, 'item'):
                                action = action.item()
                            
                            direction_map = {
                                0: pygame.Vector2(1, 0),   # Right
                                1: pygame.Vector2(-1, 0),  # Left
                                2: pygame.Vector2(0, 1),   # Down
                                3: pygame.Vector2(0, -1)   # Up
                            }
                            
                            action = max(0, min(3, int(action)))
                            
                            valid_actions = []
                            for action, direction in direction_map.items():
                                test_x = pacman.x + direction.x * pacman.speed
                                test_y = pacman.y + direction.y * pacman.speed
                                if not collides_with_wall(test_x, test_y, pacman.radius, maze):
                                    valid_actions.append(action)

                            if action in valid_actions:
                                pacman.desired_direction = direction_map[action]
                            else:
                                if valid_actions:
                                    action = random.choice(valid_actions)
                                    pacman.desired_direction = direction_map[action]
                            pacman.last_action = action
                            pacman.update(maze, active_ghosts, pellet_grid, score)
                    except Exception as e:
                        print(f"Error using imitation model: {e}")
                        constants.GAME_MODE = "A_STAR"
                        pacman.update(maze, active_ghosts, pellet_grid, score)
                
                elif hasattr(pacman, 'imitation_learner') and pacman.imitation_learner:
                    try:
                        obs = pacman._create_observation(active_ghosts, pellet_grid, maze)
                        action, probs = pacman.imitation_learner.predict(obs)
                        
                        direction_map = {
                            0: pygame.Vector2(1, 0),   # Right
                            1: pygame.Vector2(-1, 0),  # Left
                            2: pygame.Vector2(0, 1),   # Down
                            3: pygame.Vector2(0, -1)   # Up
                        }
                        
                        action = max(0, min(3, int(action)))
                        pacman.desired_direction = direction_map[action]
                        pacman.last_action = action
                        pacman.update(maze, active_ghosts, pellet_grid, score)
                    except Exception as e:
                        print(f"Error using native imitation learner: {e}")
                        constants.GAME_MODE = "A_STAR"
                        pacman.update(maze, active_ghosts, pellet_grid, score)
                else:
                    print("No imitation model available. Switching to A* mode.")
                    constants.GAME_MODE = "A_STAR"
                    pacman.update(maze, active_ghosts, pellet_grid, score)
            else:
                pacman.update(maze, active_ghosts, pellet_grid, score)
            
            # Record demonstration if in recording mode
            if RECORD_DEMONSTRATIONS and data_collector and data_collector.is_recording:
                reward = 0
                
                pacman_col, pacman_row = pixel_to_grid(pacman.x, pacman.y)
                pellet_eaten = False
                if 0 <= pacman_row < constants.ROWS and 0 <= pacman_col < constants.COLS:
                    if pellet_grid[pacman_row][pacman_col]:
                        reward += 10
                        pellet_eaten = True
                
                reward -= 0.1
                
                current_action = None
                if constants.GAME_MODE == "HUMAN":
                    if pacman.direction.x > 0:
                        current_action = 0  # Right
                    elif pacman.direction.x < 0:
                        current_action = 1  # Left
                    elif pacman.direction.y > 0:
                        current_action = 2  # Down
                    elif pacman.direction.y < 0:
                        current_action = 3  # Up
                
                data_collector.record_step(pacman, active_ghosts, current_action, reward)
                
            for ghost in active_ghosts:
                ghost.update(maze, pacman)
                


            # Check win condition
            if all(not pellet for row in pellet_grid for pellet in row):
                game_won = True
                if RECORD_DEMONSTRATIONS and data_collector and data_collector.is_recording:
                    print("Game won! Automatically stopping recording.")
                    data_collector.stop_recording(score=score, won=True)
            else:
                # Check collision with ghosts
                if not pacman.invisible_mode:
                    for ghost in active_ghosts:
                        if check_collision_pacman_ghost(pacman, ghost):
                            game_over = True
                            if RECORD_DEMONSTRATIONS and data_collector and data_collector.is_recording:
                                print("Game over! Automatically stopping recording.")
                                data_collector.stop_recording(score=score, won=False)
                            break
        
        # Render game
        restart_button = render_game(
            screen=screen,
            maze=maze,
            pellet_grid=pellet_grid,
            pacman=pacman,
            active_ghosts=active_ghosts,
            score=score,
            game_mode=constants.GAME_MODE,
            game_over=game_over,
            game_won=game_won,
            font=score_font,
            maze_type=constants.CURRENT_MAZE_TYPE
        )
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

# Run the game
if __name__ == "__main__":
    main()