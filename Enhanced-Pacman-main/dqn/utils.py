from utils.helpers import pixel_to_grid, grid_to_pixel, collides_with_wall
import numpy as np
import torch
import collections
import math
import pygame
import os
import json
import heapq  
import constants

# Define all global variables 
STUCK_POSITION_COUNTER = collections.defaultdict(int)
FORBIDDEN_ACTIONS_MAP = {}  # Maps positions to temporarily forbidden actions
POSITION_HISTORY = collections.deque(maxlen=20)  # Track recent positions
STUCK_THRESHOLD = 5  # How many visits to same position before considered "stuck"
KNOWN_TROUBLE_SPOTS = set()  # Set of known problematic positions
TROUBLE_SPOT_ACTIONS = {}  # Maps trouble spots to successful escape actions
DANGER_MAP = None  # Will store the current danger map
PREDICTION_STEPS = 2  # How many steps ahead to predict ghost movement
DANGER_RADIUS = 2  # How many tiles around a ghost are considered dangerous
GHOST_DANGER_THRESHOLD = 0.7  # Danger value threshold for avoiding actions
SAFE_PATHS = {}  # Cache of pre-computed safe paths

# Load known trouble spots if available
try:
    if os.path.exists("trouble_spots.json"):
        with open("trouble_spots.json", "r") as f:
            data = json.load(f)
            KNOWN_TROUBLE_SPOTS = set(tuple(pos) for pos in data["spots"])
            # Handle string keys properly
            actions_dict = {}
            for k, v in data.get("actions", {}).items():
                try:
                    # Convert string representation of list back to tuple
                    key_tuple = tuple(eval(k))
                    actions_dict[key_tuple] = v
                except:
                    print(f"Error converting key: {k}")
            TROUBLE_SPOT_ACTIONS.update(actions_dict)
except Exception as e:
    print(f"Could not load trouble spots: {e}")

def predict_ghost_positions(ghosts, maze, steps=PREDICTION_STEPS):
    """
    Predict ghost positions for next few steps
    Returns list of [(x, y, probability), ...] for each ghost
    """
    predictions = []
    
    for ghost in ghosts:
        # Get current ghost position and direction
        ghost_pos = pixel_to_grid(ghost.x, ghost.y)
        current_direction = (int(ghost.direction.x), int(ghost.direction.y))
        
        # Use default direction if ghost not moving
        if current_direction == (0, 0):
            current_direction = (1, 0)
            
        # Track positions and probabilities
        positions = {ghost_pos: 1.0}
        
        for step in range(steps):
            new_positions = {}
            
            for pos, prob in positions.items():
                if prob < 0.05:
                    continue
                    
                x, y = pos
                dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                dir_probs = {}
                
                for d in dirs:
                    nx, ny = x + d[0], y + d[1]
                    
                    # Skip wall collisions
                    if not (0 <= ny < len(maze) and 0 <= nx < len(maze[0])) or maze[ny][nx] == 1:
                        dir_probs[d] = 0.0
                    else:
                        # Ghosts favour current direction and avoid reversals
                        if d == current_direction:
                            dir_probs[d] = 0.7
                        elif d == (-current_direction[0], -current_direction[1]):
                            dir_probs[d] = 0.05
                        else:
                            dir_probs[d] = 0.25
                
                # Normalise probabilities
                total_prob = sum(dir_probs.values())
                if total_prob > 0:
                    for d in dir_probs:
                        dir_probs[d] /= total_prob
                
                # Add new positions with probabilities
                for d, d_prob in dir_probs.items():
                    if d_prob > 0:
                        nx, ny = x + d[0], y + d[1]
                        new_pos = (nx, ny)
                        new_prob = prob * d_prob
                        
                        if new_pos in new_positions:
                            new_positions[new_pos] += new_prob
                        else:
                            new_positions[new_pos] = new_prob
            
            positions = new_positions
        
        ghost_predictions = [(pos[0], pos[1], prob) for pos, prob in positions.items()]
        predictions.append(ghost_predictions)
    
    return predictions

def create_danger_map(ghosts, maze, pacman_pos=None):
    """
    Create 2D danger map based on ghost positions and predictions
    Returns numpy array with danger values (0.0 = safe, 1.0 = danger)
    """
    # Initialise danger map
    danger_map = np.zeros((len(maze), len(maze[0])))
    
    # Get ghost movement predictions
    ghost_predictions = predict_ghost_positions(ghosts, maze)
    
    for ghost_idx, ghost in enumerate(ghosts):
        # Mark current position as dangerous
        current_pos = pixel_to_grid(ghost.x, ghost.y)
        danger_map[current_pos[1]][current_pos[0]] = 1.0
        
        # Add danger from current position
        for dy in range(-DANGER_RADIUS, DANGER_RADIUS+1):
            for dx in range(-DANGER_RADIUS, DANGER_RADIUS+1):
                nx, ny = current_pos[0] + dx, current_pos[1] + dy
                if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]) and maze[ny][nx] == 0:
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= DANGER_RADIUS:
                        danger_value = max(0, 1.0 - (distance / DANGER_RADIUS))
                        danger_map[ny][nx] = max(danger_map[ny][nx], danger_value)
        
        # Add danger from predicted positions
        for x, y, prob in ghost_predictions[ghost_idx]:
            if prob >= 0.2:
                danger_value = min(1.0, prob * 1.2)
                danger_map[y][x] = max(danger_map[y][x], danger_value)
                
                # Add danger to adjacent cells
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]) and maze[ny][nx] == 0:
                            adj_danger = danger_value * 0.75
                            danger_map[ny][nx] = max(danger_map[ny][nx], adj_danger)
    
    # Add chase path danger if Pac-Man position provided
    if pacman_pos is not None:
        px, py = pacman_pos
        for ghost in ghosts:
            if getattr(ghost, 'chasing', False):
                ghost_pos = pixel_to_grid(ghost.x, ghost.y)
                gx, gy = ghost_pos
                
                distance = abs(px - gx) + abs(py - gy)
                
                if distance < 10:
                    path = find_path((gx, gy), (px, py), maze)
                    
                    for i, (x, y) in enumerate(path):
                        if i > 0:
                            path_danger = 0.9 * (1.0 - (i / len(path)))
                            danger_map[y][x] = max(danger_map[y][x], path_danger)
    
    return danger_map

def find_safe_route(start_pos, target_pos, maze, danger_map, max_danger=0.4):
    """
    Find safe route from start to target avoiding high danger areas
    Returns list of positions forming the path
    """
    if danger_map is None:
        danger_map = np.zeros((len(maze), len(maze[0])))
    
    # Use A* search with danger-aware cost function
    open_set = []
    closed_set = set()
    heapq.heappush(open_set, (0, start_pos))
    
    came_from = {}
    g_score = {start_pos: 0}
    f_score = {start_pos: heuristic(start_pos, target_pos)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == target_pos:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        closed_set.add(current)
        
        # Check neighbours
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = current[0] + dx, current[1] + dy
            neighbour = (nx, ny)
            
            # Skip invalid positions
            if not (0 <= ny < len(maze) and 0 <= nx < len(maze[0])) or maze[ny][nx] == 1:
                continue
            
            if neighbour in closed_set:
                continue
                
            # Skip high danger areas
            danger_level = danger_map[ny][nx]
            if danger_level > max_danger:
                continue
            
            # Calculate cost with danger penalty
            danger_penalty = 1.0 + (danger_level * 5.0)
            tentative_g_score = g_score[current] + danger_penalty
            
            if neighbour not in g_score or tentative_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g_score
                f_score[neighbour] = tentative_g_score + heuristic(neighbour, target_pos)
                
                # Add to open set if not already there
                if not any(pos == neighbour for _, pos in open_set):
                    heapq.heappush(open_set, (f_score[neighbour], neighbour))
    
    # No path found
    return []

def heuristic(a, b):
    """Manhattan distance heuristic for A* search"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_path(start, goal, maze):
    """Simple A* pathfinding from start to goal"""
    open_set = []
    closed_set = set()
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        closed_set.add(current)
        
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            
            if not (0 <= ny < len(maze) and 0 <= nx < len(maze[0])) or maze[ny][nx] == 1:
                continue
                
            if neighbor in closed_set:
                continue
                
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                
                if not any(pos == neighbor for _, pos in open_set):
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []

def evaluate_action_safety(pacman, action, ghost_predictions, danger_map):
    """
    Evaluate how safe an action is based on ghost predictions
    Returns a safety score (0.0 = dangerous, 1.0 = safe)
    """
    direction_map = {
        0: pygame.Vector2(1, 0),   # Right
        1: pygame.Vector2(-1, 0),  # Left
        2: pygame.Vector2(0, 1),   # Down
        3: pygame.Vector2(0, -1)   # Up
    }
    
    direction = direction_map[action]
    
    # Current position
    current_pos = pixel_to_grid(pacman.x, pacman.y)
    x, y = current_pos
    
    # Check immediate safety at current position
    current_danger = danger_map[y][x]
    
    # Next position after action
    next_x = x + int(direction.x)
    next_y = y + int(direction.y)
    
    # If next position is out of bounds or a wall, it's not valid
    if not (0 <= next_y < len(danger_map) and 0 <= next_x < len(danger_map[0])):
        return 0.0
    
    # Check danger at next position
    next_danger = danger_map[next_y][next_x]
    
    # Check danger at positions 2 and 3 steps ahead
    danger_ahead = 0.0
    count = 0
    
    for steps in range(2, 4):
        future_x = x + int(direction.x * steps)
        future_y = y + int(direction.y * steps)
        
        if (0 <= future_y < len(danger_map) and 0 <= future_x < len(danger_map[0])):
            danger_ahead += danger_map[future_y][future_x]
            count += 1
    
    # Average danger ahead (if any cells were valid)
    if count > 0:
        danger_ahead /= count
    
    # Compute safety score - weight immediate danger higher
    safety_score = 1.0 - (0.5 * next_danger + 0.3 * danger_ahead + 0.2 * current_danger)
    
    # Adjust for "dead ends" - discouraging actions that lead to traps
    # Check if we'd be moving into a corridor with few exits
    valid_moves_from_next = 0
    for check_dx, check_dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        check_x = next_x + check_dx
        check_y = next_y + check_dy
        
        if (0 <= check_y < len(danger_map) and 0 <= check_x < len(danger_map[0]) 
                and danger_map[check_y][check_x] < 0.8):  # Not a wall or very dangerous
            valid_moves_from_next += 1
    
    # Penalize actions that lead to locations with few exits
    if valid_moves_from_next <= 1:  # Only the way we came from
        safety_score *= 0.7  # Significant penalty for dead ends
    elif valid_moves_from_next == 2:
        safety_score *= 0.9  # Minor penalty for corridors
    
    return max(0.0, min(1.0, safety_score))

def find_nearest_pellet(pacman_pos, pellet_grid):
    """Find the nearest pellet to the given position"""
    px, py = pacman_pos
    best_dist = float('inf')
    best_pos = None
    
    for y in range(len(pellet_grid)):
        for x in range(len(pellet_grid[0])):
            if pellet_grid[y][x]:
                dist = abs(px - x) + abs(py - y)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (x, y)
    
    return best_pos

def find_nearest_safe_pellet(pacman_pos, pellet_grid, danger_map, max_danger=0.5):
    """Find the nearest pellet that can be reached safely"""
    px, py = pacman_pos
    best_pos = None
    best_score = float('inf')  # Lower is better (combined distance and danger)
    
    for y in range(len(pellet_grid)):
        for x in range(len(pellet_grid[0])):
            if pellet_grid[y][x]:
                # Calculate distance
                dist = abs(px - x) + abs(py - y)
                
                # Check danger level
                danger = danger_map[y][x]
                
                if danger <= max_danger:
                    # Score combines distance and danger
                    score = dist * (1.0 + danger)
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y)
    
    # If no safe pellet found, just return nearest pellet
    if best_pos is None:
        return find_nearest_pellet(pacman_pos, pellet_grid)
    
    return best_pos

def select_safest_action(pacman, valid_actions, ghost_predictions, danger_map, q_values=None):
    """
    Select the safest action from the valid actions
    Uses a combination of safety score and Q-values if provided
    """
    if not valid_actions:
        return 0  # Default action if none are valid
    
    # Calculate safety scores for all valid actions
    safety_scores = {}
    for action in valid_actions:
        safety_scores[action] = evaluate_action_safety(pacman, action, ghost_predictions, danger_map)
    
    # If Q-values provided, use them as a secondary factor
    if q_values is not None:
        # Normalize Q-values to 0-1 range
        valid_qs = np.array([q_values[a] for a in valid_actions])
        if len(valid_qs) > 1:
            min_q = np.min(valid_qs)
            max_q = np.max(valid_qs)
            range_q = max_q - min_q
            
            if range_q > 0:
                normalized_qs = (valid_qs - min_q) / range_q
            else:
                normalized_qs = np.ones_like(valid_qs)
        else:
            normalized_qs = np.ones_like(valid_qs)
        
        # Combined score: 70% safety, 30% Q-value
        combined_scores = {}
        for i, action in enumerate(valid_actions):
            combined_scores[action] = (0.7 * safety_scores[action]) + (0.3 * normalized_qs[i])
        
        # Choose action with highest combined score
        return max(combined_scores.items(), key=lambda x: x[1])[0]
    else:
        # Just use safety scores
        return max(safety_scores.items(), key=lambda x: x[1])[0]

def select_action_with_safety(pacman, valid_actions, ghost_predictions, danger_map, q_values, active_ghosts=None):
    """
    Select an action based on Q-values but with safety consideration
    Returns the best action that balances reward and safety
    """
    if not valid_actions:
        return np.argmax(q_values)
    
    # Get normalized Q-values (0-1 range)
    valid_qs = np.array([q_values[a] for a in valid_actions])
    if len(valid_qs) > 1:
        min_q = np.min(valid_qs)
        max_q = np.max(valid_qs)
        range_q = max_q - min_q
        
        if range_q > 0:
            normalized_qs = (valid_qs - min_q) / range_q
        else:
            normalized_qs = np.ones_like(valid_qs)
    else:
        normalized_qs = np.ones_like(valid_qs)
    
    # Calculate safety scores
    safety_scores = {}
    for action in valid_actions:
        safety_scores[action] = evaluate_action_safety(pacman, action, ghost_predictions, danger_map)
    
    # Calculate minimum ghost distance to determine safety weight
    min_ghost_distance = float('inf')
    if active_ghosts:
        for ghost in active_ghosts:
            ghost_dist = math.hypot(ghost.x - pacman.x, ghost.y - pacman.y) / constants.TILE_SIZE
            min_ghost_distance = min(min_ghost_distance, ghost_dist)
    
    # Dynamic safety weight based on ghost proximity
    # Base weight is 0.3, but increases significantly when ghosts are nearby
    base_safety_weight = 0.3
    safety_weight = base_safety_weight
    
    if min_ghost_distance < float('inf'):  # If we found at least one ghost
        if min_ghost_distance < 2.0:  # Very close - immediate danger
            safety_weight = 0.85  # Heavily prioritize safety
        elif min_ghost_distance < 3.0:  # Moderate danger
            safety_weight = 0.7   # Significantly prioritize safety
        elif min_ghost_distance < 5.0:  # Potential danger
            safety_weight = 0.5   # Balance safety and exploitation
    
    # Combine Q-values and safety with dynamic weighting
    combined_scores = {}
    for i, action in enumerate(valid_actions):
        q_weight = 1.0 - safety_weight  # Weight for the Q-value
        combined_scores[action] = (q_weight * normalized_qs[i]) + (safety_weight * safety_scores[action])
    
    # Return action with highest combined score
    return max(combined_scores.items(), key=lambda x: x[1])[0]

# Modified get_dqn_action function with ghost avoidance
# Modified get_dqn_action function with ghost avoidance
def get_dqn_action(pacman, active_ghosts, pellet_grid, maze, dqn_model):
    """
    Get the next action from the DQN model with enhanced ghost avoidance
    """
    global STUCK_POSITION_COUNTER, FORBIDDEN_ACTIONS_MAP, POSITION_HISTORY
    global KNOWN_TROUBLE_SPOTS, DANGER_MAP, SAFE_PATHS, TROUBLE_SPOT_ACTIONS, STUCK_THRESHOLD
    
    # Get current grid position
    current_pos = pixel_to_grid(pacman.x, pacman.y)
    current_pos_tuple = tuple(current_pos)
    
    # Track position history
    POSITION_HISTORY.append(current_pos_tuple)
    
    # Update stuck position counter for current position
    STUCK_POSITION_COUNTER[current_pos_tuple] += 1
    
    # Create danger map if needed
    DANGER_MAP = create_danger_map(active_ghosts, maze, current_pos)
    
    # Get expected observation from model
    model_obs_dim = dqn_model.policy.observation_space.shape[0]
    observation = pacman._create_observation(active_ghosts, pellet_grid, maze)
    
    if len(observation) > model_obs_dim:
        observation = observation[:model_obs_dim]
    
    # Get Q-values for all actions
    with torch.no_grad():
        q_values = dqn_model.policy.q_net(torch.as_tensor(observation).float())
        q_values = q_values.numpy()
    
    # Initialize action history if needed
    if not hasattr(pacman, 'action_history') or pacman.action_history is None:
        pacman.action_history = []
    
    # Get valid actions (those that don't immediately hit walls)
    valid_actions = []
    for action in range(4):  # 0=right, 1=left, 2=down, 3=up
        # Convert action to direction
        direction_map = {
            0: pygame.Vector2(1, 0),   # Right
            1: pygame.Vector2(-1, 0),  # Left
            2: pygame.Vector2(0, 1),   # Down
            3: pygame.Vector2(0, -1)   # Up
        }
        direction = direction_map[action]
        
        # Check if this action would hit a wall
        test_x = pacman.x + direction.x * pacman.speed
        test_y = pacman.y + direction.y * pacman.speed
        
        if not collides_with_wall(test_x, test_y, pacman.radius, maze):
            valid_actions.append(action)
    
    # If no valid actions, use the original best action
    if not valid_actions:
        action = np.argmax(q_values)
    else:
        # Check if we're in a known trouble spot
        in_trouble_spot = current_pos_tuple in KNOWN_TROUBLE_SPOTS
        
        # Check if we're stuck (repeating the same position)
        is_stuck = False
        if len(POSITION_HISTORY) >= 8:
            recent_positions = list(POSITION_HISTORY)[-8:]
            position_counts = collections.Counter(recent_positions)
            most_common_pos, count = position_counts.most_common(1)[0]
            is_stuck = count >= STUCK_THRESHOLD and most_common_pos == current_pos_tuple
        
        # Detect oscillation pattern
        is_oscillating = False
        if len(POSITION_HISTORY) >= 6:
            # Check for A-B-A-B-A-B pattern
            recent_six = list(POSITION_HISTORY)[-6:]
            is_oscillating = (
                recent_six[0] == recent_six[2] == recent_six[4] and
                recent_six[1] == recent_six[3] == recent_six[5] and
                recent_six[0] != recent_six[1]
            )
        
        # Check if this is the position where we keep getting stuck at 300 points
        score_approx_300 = (getattr(pacman, 'score', 0) in range(290, 310))
        at_common_stuck_score = score_approx_300
        
        # Get ghost predictions for safety evaluation
        ghost_predictions = predict_ghost_positions(active_ghosts, maze)
        
        # Determine action selection strategy
        if in_trouble_spot or is_stuck or is_oscillating or at_common_stuck_score:
            if in_trouble_spot and current_pos_tuple in TROUBLE_SPOT_ACTIONS:
                # Use previously successful escape actions for this trouble spot
                escape_actions = TROUBLE_SPOT_ACTIONS[current_pos_tuple]
                valid_escape_actions = [a for a in escape_actions if a in valid_actions]
                
                # Check safety of escape actions
                safe_escape_actions = []
                for a in valid_escape_actions:
                    safety = evaluate_action_safety(pacman, a, ghost_predictions, DANGER_MAP)
                    if safety > 0.6:  # Only use if reasonably safe
                        safe_escape_actions.append(a)
                
                if safe_escape_actions:
                    action = np.random.choice(safe_escape_actions)
                elif valid_escape_actions:
                    action = np.random.choice(valid_escape_actions)
                else:
                    # Fall back to safety-based selection
                    action = select_safest_action(pacman, valid_actions, ghost_predictions, DANGER_MAP, q_values)
                
                # Mark this as a known trouble spot
                if current_pos_tuple not in KNOWN_TROUBLE_SPOTS:
                    KNOWN_TROUBLE_SPOTS.add(current_pos_tuple)
                    print(f"Added new trouble spot: {current_pos_tuple}")
            else:
                # Apply random exploration for stuck situations
                # More aggressive randomness for oscillation patterns
                exploration_chance = 0.4
                if is_oscillating:
                    exploration_chance = 0.7
                
                if np.random.random() < exploration_chance:
                    # Choose random action from valid actions, avoiding recently used ones
                    recent_actions = pacman.action_history[-3:] if len(pacman.action_history) >= 3 else []
                    available_actions = [a for a in valid_actions if a not in recent_actions]
                    
                    # If all actions were recently used, just use all valid actions
                    if not available_actions:
                        available_actions = valid_actions
                    
                    # Filter actions by safety
                    safe_actions = []
                    for a in available_actions:
                        safety = evaluate_action_safety(pacman, a, ghost_predictions, DANGER_MAP)
                        if safety > 0.4:  # Consider reasonably safe actions
                            safe_actions.append((a, safety))
                    
                    if safe_actions:
                        # Weight selection by safety score
                        total_safety = sum(s for _, s in safe_actions)
                        probs = [s/total_safety for _, s in safe_actions]
                        action = np.random.choice([a for a, _ in safe_actions], p=probs)
                    else:
                        action = np.random.choice(available_actions)
                    
                    # Log this as a potential trouble spot
                    if current_pos_tuple not in KNOWN_TROUBLE_SPOTS:
                        KNOWN_TROUBLE_SPOTS.add(current_pos_tuple)
                        print(f"Added new trouble spot: {current_pos_tuple}")
                else:
                    # Use Q-values but with safety consideration
                    action = select_action_with_safety(pacman, valid_actions, ghost_predictions, DANGER_MAP, q_values, active_ghosts)
        else:
            # Normal case - not stuck but still consider safety
            # Is there an immediate ghost threat?
            any_ghost_close = False
            for ghost in active_ghosts:
                ghost_pos = pixel_to_grid(ghost.x, ghost.y)
                ghost_dist = abs(ghost_pos[0] - current_pos[0]) + abs(ghost_pos[1] - current_pos[1])
                if ghost_dist < 5:  # Ghost is close
                    any_ghost_close = True
                    break
            
            if any_ghost_close:
                # Prioritize safety when ghosts are nearby
                action = select_safest_action(pacman, valid_actions, ghost_predictions, DANGER_MAP, q_values)
            else:
                # Use standard selection with safety as a factor
                action = select_action_with_safety(pacman, valid_actions, ghost_predictions, DANGER_MAP, q_values, active_ghosts)
    
    # Update action history
    pacman.action_history.append(action)
    if len(pacman.action_history) > 20:
        pacman.action_history.pop(0)
    
    # Direction map for converting action to direction
    direction_map = {
        0: pygame.Vector2(1, 0),   # Right
        1: pygame.Vector2(-1, 0),  # Left
        2: pygame.Vector2(0, 1),   # Down
        3: pygame.Vector2(0, -1)   # Up
    }
    
    # Check if we're leaving a trouble spot
    if current_pos_tuple in KNOWN_TROUBLE_SPOTS:
        # If we just left a trouble spot, record the successful action
        next_x = current_pos[0] + int(direction_map[action].x)
        next_y = current_pos[1] + int(direction_map[action].y)
        
        # Will need to check this on next step to see if we escaped
        if not hasattr(pacman, 'leaving_trouble_spot'):
            pacman.leaving_trouble_spot = {}
        
        pacman.leaving_trouble_spot[current_pos_tuple] = {
            'action': action,
            'step': getattr(pacman, 'steps_taken', 0)
        }
    
    # Check if we successfully escaped a trouble spot
    if hasattr(pacman, 'leaving_trouble_spot'):
        for pos, data in list(pacman.leaving_trouble_spot.items()):
            # If we were in this position within the last 2 steps and now we're not
            if pos != current_pos_tuple and (getattr(pacman, 'steps_taken', 0) - data['step']) <= 2:
                # Successfully escaped! Record the action that worked
                action_used = data['action']
                if pos not in TROUBLE_SPOT_ACTIONS:
                    TROUBLE_SPOT_ACTIONS[pos] = []
                if action_used not in TROUBLE_SPOT_ACTIONS[pos]:
                    TROUBLE_SPOT_ACTIONS[pos].append(action_used)
                    # Save updated trouble spots
                    save_trouble_spots()
                
                # Remove from the leaving dict since we've processed it
                pacman.leaving_trouble_spot.pop(pos)
    
    return action, direction_map[action]

def save_trouble_spots():
    """Save known trouble spots and successful escape actions to a file"""
    try:
        data = {
            "spots": [list(pos) for pos in KNOWN_TROUBLE_SPOTS],
            "actions": {str(list(k)): v for k, v in TROUBLE_SPOT_ACTIONS.items()}
        }
        with open("trouble_spots.json", "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving trouble spots: {e}")