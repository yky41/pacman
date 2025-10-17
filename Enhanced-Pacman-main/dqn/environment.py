# Modified environment.py with improved reward function and tracking
import pygame
import random
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from collections import defaultdict, deque

import constants
from utils.helpers import pixel_to_grid, grid_to_pixel, check_collision_pacman_ghost
from entities.pacman import Pacman
from entities.ghost import Ghost

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PacmanEnv(gym.Env):
    """Custom environment that follows gym interface for DQN training"""
    metadata = {'render.modes': ['human']}

    def __init__(self, maze, pellet_grid, use_enhanced_features=False, visibility_radius=None):
        super(PacmanEnv, self).__init__()

        # Store environment data
        self.maze = maze
        self.original_pellet_grid = [row[:] for row in pellet_grid]  # Deep copy

        # Define action and observation space
        # 4 actions: 0=right, 1=left, 2=down, 3=up
        self.action_space = spaces.Discrete(4)

        # Determine observation space dimension based on whether to use enhanced features
        self.use_enhanced_features = use_enhanced_features

        # Set visibility radius (None means unlimited visibility)
        self.visibility_radius = visibility_radius

        if use_enhanced_features:
            # Enhanced observation with features:
            # - pacman position (2)
            # - direction (4)
            # - wall sensors (4)
            # - pellet sensors (4)
            # - ghost sensors (4)
            # - quadrant pellets (4)
            # - junction indicator (1)
            # - closest ghost distance (1)
            # - stuck indicator (1)
            # - oscillation indicator (1)
            # - valid moves (4)
            # - movement history (4)
            # - ghost direction (4)
            # - ghost in path (4)
            obs_dim = 43
        else:
            # Original observation
            obs_dim = 18

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Game state
        self.pellet_grid = None
        self.pacman = None
        self.active_ghosts = []
        self.ghosts_to_release = []

        # Get ghost house and spawn information based on current maze type
        ghost_info = constants.get_ghost_house_info()
        self.ghost_spawn = grid_to_pixel(*ghost_info["spawn"])

        # Get pacman spawn position based on current maze
        pacman_spawn_pos = constants.get_pacman_spawn()
        self.pacman_spawn = grid_to_pixel(*pacman_spawn_pos)

        self.score = 0
        self.steps_taken = 0

        # Adjust max_steps based on maze type
        if constants.CURRENT_MAZE_TYPE == "SIMPLE":
            self.max_steps = 1000  # Fewer steps for simple maze
        else:
            self.max_steps = 2000  # Original value for complex maze

        # Adjust stuck detection based on maze type
        if constants.CURRENT_MAZE_TYPE == "SIMPLE":
            self.stuck_threshold = 200  # Shorter threshold for simple maze
        else:
            self.stuck_threshold = 300  # Original threshold

        self.last_pellet_steps = 0  # Steps since last pellet eaten
        self.last_score = 0  # Track score change

        # Enhanced tracking for better exploration
        self.position_visit_counts = defaultdict(int)  # Track visit counts per position
        self.recent_positions = deque(maxlen=20)  # Track recent positions
        self.pellets_eaten = 0  # Count pellets eaten
        self.total_pellets = sum(sum(row) for row in self.original_pellet_grid)

        # Reward scaling and normalisation
        self.reward_history = []
        self.cumulative_reward = 0

        # Ghost colours and targets - adjusted based on maze type
        self.ghost_colors = [(255, 0, 0), (255, 184, 255), (0, 255, 255), (255, 184, 82)]

        # Set targets based on current maze dimensions
        self.ghost_targets = {
            (255, 0, 0): grid_to_pixel(constants.COLS - 1, 0),         # Red: Top-right
            (255, 184, 255): grid_to_pixel(0, 0),                      # Pink: Top-left
            (0, 255, 255): grid_to_pixel(constants.COLS - 1, constants.ROWS - 1),  # Cyan: Bottom-right
            (255, 184, 82): grid_to_pixel(0, constants.ROWS - 1)       # Orange: Bottom-left
        }

        self.reset()

    def reset(self, seed=None, options=None):
        # Optionally set the seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reset pellet grid (deep copy the original)
        self.pellet_grid = [row[:] for row in self.original_pellet_grid]
        total_pellets = sum(sum(row) for row in self.pellet_grid)

        # Get pacman spawn position based on current maze
        pacman_spawn_pos = constants.get_pacman_spawn()
        self.pacman_spawn = grid_to_pixel(*pacman_spawn_pos)

        # Initialise Pac-Man and ghosts
        self.pacman = Pacman(*self.pacman_spawn)
        self.active_ghosts = []
        self.ghosts_to_release = []

        if constants.CURRENT_MAZE_TYPE == "SIMPLE":
            # For simple maze, place ghosts directly in the maze at fixed positions
            ghost_count = min(constants.SIMPLE_MAZE_GHOST_COUNT, len(self.ghost_colors))

            if ghost_count > 0:
                # First ghost in top right
                top_right_pos = grid_to_pixel(constants.COLS - 3, 3)
                self.active_ghosts.append(
                    Ghost(top_right_pos[0], top_right_pos[1], self.ghost_colors[0],
                         scatter_target=self.ghost_targets[self.ghost_colors[0]])
                )

            if ghost_count > 1:
                # Second ghost in bottom left
                bottom_left_pos = grid_to_pixel(3, constants.ROWS - 3)
                self.active_ghosts.append(
                    Ghost(bottom_left_pos[0], bottom_left_pos[1], self.ghost_colors[1],
                         scatter_target=self.ghost_targets[self.ghost_colors[1]])
                )

            # Mark these ghosts as already exited from ghost house
            for ghost in self.active_ghosts:
                ghost.exited = True


        else:
            # For complex maze, use ghost house and release system
            # Create first ghost in ghost house
            first_ghost = Ghost(self.ghost_spawn[0], self.ghost_spawn[1],
                               self.ghost_colors[0],
                               scatter_target=self.ghost_targets[self.ghost_colors[0]])
            self.active_ghosts.append(first_ghost)

            # Queue remaining ghosts
            for color in self.ghost_colors[1:]:
                ghost = Ghost(self.ghost_spawn[0], self.ghost_spawn[1],
                             color,
                             scatter_target=self.ghost_targets[color])
                self.ghosts_to_release.append(ghost)


        # Reset score and step counters
        self.score = 0
        self.last_score = 0
        self.steps_taken = 0
        self.last_pellet_steps = 0
        self.cumulative_reward = 0

        # Reset exploration tracking
        self.position_visit_counts = defaultdict(int)
        self.recent_positions = deque(maxlen=20)
        self.pellets_eaten = 0

        # Reset reward history
        self.reward_history = []

        # Reset tracking variables for improved reward function
        self.last_distance_to_nearest_pellet = None
        self.consecutive_moves_without_progress = 0
        self.previous_ghost_distances = None

        # Get initial observation
        observation = self._get_observation()
        return observation, {}  # Return observation and empty info dict


    def step(self, action):
        # 接收并应用模型做出的动作
        action = int(action.item()) if hasattr(action, 'item') else int(action)
        direction_map = {
            0: pygame.Vector2(1, 0),  # Right
            1: pygame.Vector2(-1, 0),  # Left
            2: pygame.Vector2(0, 1),  # Down
            3: pygame.Vector2(0, -1)  # Up
        }
        self.pacman.desired_direction = direction_map.get(action, pygame.Vector2(0, 0))
        self.pacman.last_action = action

        # 保存用于计算奖励的旧状态
        prev_score = self.score
        prev_pacman_pos = pixel_to_grid(self.pacman.x, self.pacman.y)

        # ==================== 核心逻辑修正 ====================
        # 1. 先检查吃豆和更新分数
        #    注意：这里是在 Pacman 移动 *之前* 检查他当前格子有没有豆
        pacman_col, pacman_row = pixel_to_grid(self.pacman.x, self.pacman.y)
        pellet_eaten = False
        if 0 <= pacman_row < constants.ROWS and 0 <= pacman_col < constants.COLS:
            if self.pellet_grid[pacman_row][pacman_col]:
                self.pellet_grid[pacman_row][pacman_col] = False
                self.score += 10
                self.pellets_eaten += 1
                pellet_eaten = True
                self.last_pellet_steps = 0

        # 2. 用最新的分数，调用一次完整的 update 函数
        #    这个函数会处理移动、墙壁碰撞、技能触发等所有逻辑
        self.pacman.update(self.maze, self.active_ghosts, self.pellet_grid, score=self.score)

        # 3. 更新幽灵
        for ghost in self.active_ghosts:
            ghost.update(self.maze, self.pacman)
        # =======================================================

        # 更新后的新位置
        current_pos = pixel_to_grid(self.pacman.x, self.pacman.y)

        # 检查游戏是否结束
        done = False
        game_won = all(not pellet for row in self.pellet_grid for pellet in row)

        if game_won:
            done = True
        elif not self.pacman.invisible_mode:  # 只有在非隐身时才检查与幽灵的碰撞
            for ghost in self.active_ghosts:
                if not ghost.frozen and check_collision_pacman_ghost(self.pacman, ghost):
                    done = True
                    break

        # 其他结束条件
        self.steps_taken += 1
        self.last_pellet_steps += 1
        if self.steps_taken >= self.max_steps or self.last_pellet_steps >= self.stuck_threshold:
            done = True

        # 更新探索信息
        self.position_visit_counts[current_pos] += 1
        self.recent_positions.append(current_pos)

        # 为计算奖励寻找最近的豆子
        nearest_pellet = None
        nearest_dist = float('inf')
        for r in range(constants.ROWS):
            for c in range(constants.COLS):
                if self.pellet_grid[r][c]:
                    dist = abs(current_pos[0] - c) + abs(current_pos[1] - r)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_pellet = (c, r)

        # 计算奖励
        reward = self.calculate_improved_reward(
            pellet_eaten, done, game_won, prev_pacman_pos,
            current_pos, nearest_pellet, prev_score
        )

        # 获取给模型的下一个观察
        observation = self._get_observation()

        # 更新和返回信息
        self.last_score = self.score
        self.cumulative_reward += reward
        info = {
            'score': self.score, 'steps': self.steps_taken,
            'pellets_remaining': self.total_pellets - self.pellets_eaten,
            'pellets_eaten': self.pellets_eaten, 'won': game_won,
            'pellet_eaten': pellet_eaten,
            'nearest_pellet_dist': nearest_dist if nearest_pellet else None,
            'cumulative_reward': self.cumulative_reward,
            'revisit_rate': self._calculate_revisit_rate()
        }
        return observation, reward, done, False, info

    def _calculate_revisit_rate(self):
        """Calculate how often the agent revisits the same positions"""
        if not self.recent_positions:
            return 0.0

        # Count frequency of each position in recent history
        position_counts = {}
        for pos in self.recent_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1

        # Calculate average revisit count (avg number of times each unique position appears)
        if not position_counts:
            return 0.0

        unique_positions = len(position_counts)
        total_visits = len(self.recent_positions)

        # Revisit rate: 0 means all different positions, higher means more revisits
        if unique_positions == 0:
            return 0.0

        return (total_visits - unique_positions) / total_visits

    def calculate_improved_reward(self, pellet_eaten, done, game_won, prev_pos, current_pos, nearest_pellet, prev_score):
        """
        Optimized reward function with enhanced exploration and anti-stuck mechanisms
        """
        reward = 0
        # ================== 新增代码块：开始 ==================
        # 奖励技能的触发
        score_delta = self.score - prev_score
        if score_delta > 0:
            # 如果分数刚刚跨过200的倍数，说明很可能触发了技能
            if (self.score // 200) > (prev_score // 200):
                # 给予一个较大的奖励来鼓励冰冻幽灵和加速
                reward += 3.0
        # Scale factor for rewards based on maze type (smaller maze = higher rewards)
        scale_factor = 1.0
        if constants.CURRENT_MAZE_TYPE == "SIMPLE":
            scale_factor = 1.5  # Increase rewards for simple maze

        # --- PRIMARY OBJECTIVE: PELLET COLLECTION ---
        # Enhanced reward for eating pellets (primary goal)
        if pellet_eaten:
            # Base reward for eating any pellet
            reward += 2.5 * scale_factor

            # ================== 新增代码块：开始 ==================
            # 如果在加速状态下吃到豆，给予额外奖励
            if self.pacman.boost_mode:
                reward += 0.5 * scale_factor

            # 如果在隐身状态下吃到豆，也给予奖励
            if self.pacman.invisible_mode:
                reward += 0.5 * scale_factor
            # ================== 新增代码块：结束 ==================
            # Additional bonus for eating pellets that are part of a cluster
            nearby_pellets = self._count_nearby_pellets(current_pos)
            if nearby_pellets > 1:
                reward += 0.5 * min(nearby_pellets, 4) * scale_factor

            # Extra reward if collecting pellets rapidly
            if self.last_pellet_steps < 10:  # Collected another pellet quickly
                reward += 0.8 * scale_factor

            # Bonus for eating pellets that were further away (encourages going after distant pellets)
            if self.last_distance_to_nearest_pellet is not None and self.last_distance_to_nearest_pellet > 5:
                reward += 0.3 * min(1.0, self.last_distance_to_nearest_pellet / 10.0) * scale_factor

            # Progressive bonus based on percentage of pellets collected
            collection_rate = self.pellets_eaten / self.total_pellets if self.total_pellets > 0 else 0
            if collection_rate > 0.25:
                collection_milestone_bonus = min(2.0, collection_rate * 2) * scale_factor
                reward += collection_milestone_bonus

            # Reset counter for moves without progress
            self.consecutive_moves_without_progress = 0
            self.last_pellet_steps = 0  # Reset counter

        # --- TERMINAL STATES ---
        # Major penalty for dying
        if done and not game_won:
            reward -= 3.0 * scale_factor

        # Huge reward for winning - scaled based on maze type
        if game_won:
            if constants.CURRENT_MAZE_TYPE == "SIMPLE":
                reward += 20.0  # Higher reward for winning in simple maze
            else:
                reward += 15.0  # Original reward for complex maze

        # --- STRATEGIC NAVIGATION ---
        # Current position and tracking
        current_pos_tuple = (current_pos[0], current_pos[1])

        # --- ANTI-REVISIT MECHANISM ---
        # Apply increasing penalty for revisiting the same positions frequently
        position_revisit_count = self.position_visit_counts.get(current_pos_tuple, 0)
        if position_revisit_count > 3:
            # Scale penalty by how long we've been stuck
            stuck_factor = min(1.0, self.last_pellet_steps / self.stuck_threshold)
            revisit_penalty = min(0.7, 0.1 * (position_revisit_count - 3)) * stuck_factor * scale_factor
            reward -= revisit_penalty

        # Conversely, reward for exploring new areas, especially when stuck
        if position_revisit_count <= 1:
            # Higher exploration bonus when stuck
            stuck_factor = min(1.0, self.last_pellet_steps / (self.stuck_threshold / 2))
            exploration_bonus = 0.3 * stuck_factor * scale_factor
            reward += exploration_bonus

        # Adjust exploration bonus based on game progress
        remaining_pellets = self.total_pellets - self.pellets_eaten
        progress = 1.0 - (remaining_pellets / self.total_pellets) if self.total_pellets > 0 else 0

        # Higher exploration early, more focus on exploitation later
        exploration_bonus = 0.3 * (1.0 - progress) if progress < 0.7 else 0.1
        if constants.CURRENT_MAZE_TYPE == "SIMPLE":
            exploration_bonus *= 1.2  # Higher exploration bonus in simple maze

        # --- PROGRESS TOWARD NEAREST PELLET ---
        # Calculate current distance to nearest pellet
        current_distance = float('inf')
        if nearest_pellet:
            current_distance = abs(current_pos[0] - nearest_pellet[0]) + abs(current_pos[1] - nearest_pellet[1])

            # If we have a previous distance, compare to see if we're getting closer or farther
            if self.last_distance_to_nearest_pellet is not None:
                # Reward for getting closer to a pellet
                if current_distance < self.last_distance_to_nearest_pellet:
                    # Progressive reward - more reward for bigger improvements
                    improvement = self.last_distance_to_nearest_pellet - current_distance
                    reward += (0.1 + (0.05 * improvement)) * scale_factor
                    self.consecutive_moves_without_progress = 0
                # Penalty for moving away from the nearest pellet - only apply if not avoiding a ghost
                elif current_distance > self.last_distance_to_nearest_pellet:
                    # Check if any ghost is nearby before penalizing
                    ghost_nearby = False
                    for ghost in self.active_ghosts:
                        ghost_distance = math.hypot(ghost.x - self.pacman.x, ghost.y - self.pacman.y) / constants.TILE_SIZE
                        if ghost_distance < 4.0:  # Only consider "nearby" ghosts
                            ghost_nearby = True
                            break

                    # Check if potentially heading toward a pellet cluster
                    heading_to_cluster = self._is_heading_to_pellet_cluster(current_pos, self.pacman.desired_direction)

                    if not ghost_nearby and not heading_to_cluster:
                        reward -= 0.1 * scale_factor
                        self.consecutive_moves_without_progress += 1
                else:
                    # No change in distance - might be following a wall
                    self.consecutive_moves_without_progress += 1

            # Update last distance for next step
            self.last_distance_to_nearest_pellet = current_distance

        # --- GHOST AVOIDANCE ---
        # Calculate distances to ghosts
        normal_ghosts = [g for g in self.active_ghosts if not g.frozen]
        ghost_distances = [math.hypot(ghost.x - self.pacman.x, ghost.y - self.pacman.y) / constants.TILE_SIZE
                           for ghost in normal_ghosts]

        # Smart ghost avoidance rewards/penalties based on distance
        if ghost_distances:
            min_ghost_distance = min(ghost_distances)

            # Adjust thresholds based on maze size
            severe_danger_threshold = 2.0
            medium_danger_threshold = 3.0
            light_danger_threshold = 4.0
            if constants.CURRENT_MAZE_TYPE == "SIMPLE":
                severe_danger_threshold = 1.5
                medium_danger_threshold = 2.5
                light_danger_threshold = 3.5

            # Severe danger - very close to ghost
            if min_ghost_distance < severe_danger_threshold:
                reward -= 2.0 * scale_factor
            # Medium danger
            elif min_ghost_distance < medium_danger_threshold:
                reward -= 0.8 * scale_factor
            # Light danger
            elif min_ghost_distance < light_danger_threshold:
                reward -= 0.3 * scale_factor
            # Reward for keeping safe distance
            elif min_ghost_distance > 8.0:
                reward += 0.1 * scale_factor

        # Compare with previous distances - reward for active ghost avoidance
        if self.previous_ghost_distances is not None and ghost_distances:
            for i, (prev_dist, curr_dist) in enumerate(zip(self.previous_ghost_distances, ghost_distances)):
                if prev_dist < 4.0:  # Only consider ghosts that were close
                    if curr_dist > prev_dist:
                        # More significant reward for moving away from a nearby ghost
                        escape_reward = 0.5 * (1.0 / (prev_dist + 0.1)) * scale_factor
                        reward += escape_reward

        # Update ghost distances for next comparison
        self.previous_ghost_distances = ghost_distances

        # --- JUNCTIONS AND DECISION POINTS ---
        # Reward for making good decisions at junctions
        if hasattr(self.pacman, '_get_valid_actions') and self.maze is not None:
            valid_actions = self.pacman._get_valid_actions(self.maze)
            if len(valid_actions) > 2:  # We're at a junction
                # If we were previously at this junction and now moving in a good direction
                # (getting closer to pellet or away from ghost), give bonus
                if hasattr(self, 'last_junction_pos') and self.last_junction_pos == current_pos_tuple:
                    if (nearest_pellet and current_distance < self.last_junction_pellet_dist) or \
                    (ghost_distances and min(ghost_distances) > self.last_junction_ghost_dist):
                        reward += 0.3 * scale_factor  # Good decision at junction

                # Remember this junction
                self.last_junction_pos = current_pos_tuple
                self.last_junction_pellet_dist = current_distance if nearest_pellet else float('inf')
                self.last_junction_ghost_dist = min(ghost_distances) if ghost_distances else 0

        # --- ANTI-OSCILLATION MECHANISM ---
        # Keep the existing oscillation prevention logic
        if hasattr(self.pacman, 'last_positions'):
            # Check if we've been at the same position for multiple steps
            if len(self.pacman.last_positions) >= 3:
                # If we've been in the same position for 3+ steps
                if all(pos == current_pos_tuple for pos in self.pacman.last_positions[-3:]):
                    reward -= 0.5 * scale_factor  # Significant penalty for being stuck

            # Check for back-and-forth oscillation (alternating between two positions)
            if len(self.pacman.last_positions) >= 4:
                if (self.pacman.last_positions[-1] == self.pacman.last_positions[-3] and
                    self.pacman.last_positions[-2] == self.pacman.last_positions[-4]):
                    reward -= 0.7 * scale_factor  # Stronger penalty for oscillation

        # Check for action oscillation
        if hasattr(self.pacman, 'action_history') and len(self.pacman.action_history) >= 4:
            # Check if the agent is just alternating between opposite actions
            # 0/1 (right/left) are opposites, 2/3 (down/up) are opposites
            last_4_actions = self.pacman.action_history[-4:]
            pattern_is_oscillating = ((last_4_actions[0] == 0 and last_4_actions[1] == 1 and last_4_actions[2] == 0 and last_4_actions[3] == 1) or
                (last_4_actions[0] == 1 and last_4_actions[1] == 0 and last_4_actions[2] == 1 and last_4_actions[3] == 0) or
                (last_4_actions[0] == 2 and last_4_actions[1] == 3 and last_4_actions[2] == 2 and last_4_actions[3] == 3) or
                (last_4_actions[0] == 3 and last_4_actions[1] == 2 and last_4_actions[2] == 3 and last_4_actions[3] == 2))

            # Only penalize if no pellets eaten recently to avoid penalizing useful oscillation
            if pattern_is_oscillating and self.last_pellet_steps > 5:
                reward -= 0.8 * scale_factor  # Heavy penalty for direction thrashing

        # --- CONSECUTIVE MOVES WITHOUT PROGRESS ---
        if self.consecutive_moves_without_progress > 5:
            # Apply an increasing penalty for lack of progress
            reward -= 0.1 * (self.consecutive_moves_without_progress - 5) * scale_factor

        # --- MOVEMENT PENALTY ---
        # Smaller movement penalty to encourage more exploration
        movement_penalty = 0.005
        if constants.CURRENT_MAZE_TYPE == "SIMPLE":
            movement_penalty = 0.003  # Even smaller penalty in simple maze
        reward -= movement_penalty * scale_factor

        # --- WALL COLLISION PENALTY ---
        if hasattr(self.pacman, 'wall_collision_count') and self.pacman.wall_collision_count > 0:
            # Exponentially increasing penalty for repeated wall collisions
            wall_penalty = 0.2 * min(4, self.pacman.wall_collision_count) * scale_factor
            reward -= wall_penalty

            # Extra penalty for collisions at the same spot
            if self.pacman.wall_collision_count > 2:
                reward -= 0.5 * scale_factor

        # --- DYNAMIC REWARD SCALING ---
        # Scale rewards based on progress to encourage completing the game
        collection_rate = self.pellets_eaten / self.total_pellets if self.total_pellets > 0 else 0
        if collection_rate > 0.5:
            # Boost rewards in latter half of game
            reward *= (1.0 + 0.5 * collection_rate)

        return reward


    def _count_nearby_pellets(self, position):
        """Count pellets in adjacent cells with expanded search radius"""
        count = 0
        x, y = position

        # Check immediate neighbors (Manhattan distance 1)
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if (0 <= ny < constants.ROWS and 0 <= nx < constants.COLS and
                self.pellet_grid[ny][nx]):
                count += 1

        # Check diagonal neighbors (lower weight)
        for dx, dy in [(1,1), (-1,1), (1,-1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            if (0 <= ny < constants.ROWS and 0 <= nx < constants.COLS and
                self.pellet_grid[ny][nx]):
                count += 0.5  # Diagonal neighbors count as half

        return count

    def _is_heading_to_pellet_cluster(self, position, direction):
        """Enhanced check if current direction leads toward a cluster of pellets"""
        if direction.length_squared() == 0:
            return False

        x, y = position
        dx, dy = direction.x, direction.y

        # Check next few cells in this direction
        pellet_count = 0
        pellet_value = 0  # Weighted value of pellets (closer = more valuable)

        for i in range(1, 7):  # Look up to 7 cells ahead (extended range)
            nx, ny = x + int(dx * i), y + int(dy * i)
            if (0 <= ny < constants.ROWS and 0 <= nx < constants.COLS):
                if self.maze[ny][nx] == 1:  # Hit a wall
                    break

                if self.pellet_grid[ny][nx]:
                    pellet_count += 1
                    # Closer pellets are worth more
                    pellet_value += 1.0 / i

                # Also check adjacent cells to the path
                for side_dx, side_dy in [(-dy, dx), (dy, -dx)]:  # Perpendicular directions
                    # Convert to integers to avoid the float index error
                    side_x, side_y = int(nx + side_dx), int(ny + side_dy)
                    # Make sure indices are valid
                    if (0 <= side_y < constants.ROWS and 0 <= side_x < constants.COLS and
                        self.maze[side_y][side_x] == 0 and self.pellet_grid[side_y][side_x]):
                        pellet_count += 0.5  # Adjacent pellets count as half
                        pellet_value += 0.3 / i  # Less valuable but still count

        # Consider both count and weighted value
        return pellet_count >= 2 or pellet_value >= 0.75

    def _get_observation(self):
        """Create observation for DQN agent with limited visibility"""
        # Always pass the maze and pellet grid to the observation creation method
        # full_obs = self.pacman._create_observation(
        #     self.active_ghosts,
        #     self.pellet_grid,
        #     self.maze,
        #     visibility_radius=self.visibility_radius
        # )
        full_obs = self.pacman._create_observation(
            self.active_ghosts,
            self.pellet_grid,
            self.maze
        )

        # Return either the full observation or just the original features
        if self.use_enhanced_features:
            return full_obs
        else:
            return full_obs[:18]

    def render(self, mode='human'):
        """This is handled by the main game loop"""
        pass

    def close(self):
        pass



