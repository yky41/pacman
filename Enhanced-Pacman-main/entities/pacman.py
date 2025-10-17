import pygame
import math
import numpy as np
from collections import defaultdict, deque  # <--- 修正: 添加了必要的 import
from utils.helpers import grid_to_pixel, pixel_to_grid, collides_with_wall
from utils.a_star import a_star
import constants
import random


class Pacman:
    def __init__(self, x, y):
        """Initialise Pac-Man with position and movement parameters."""
        self.x = x
        self.y = y
        self.radius = constants.TILE_SIZE // 2 - 2
        self.speed = 3
        self.direction = pygame.Vector2(0, 0)
        self.desired_direction = pygame.Vector2(0, 0)

        # A* pathfinding attributes
        self.path = []
        self.target_pellet = None
        self.path_finding_cooldown = 0

        # DQN attributes
        self.last_action = None
        self.dqn_model = None
        self.action_history = []
        self.last_positions = []

        # 加速状态
        self.boost_mode = False
        self.boost_timer = 0

        # 隐身状态
        self.invisible_mode = False
        self.invisible_timer = 0

        # 统一的技能触发分数记录器
        self.last_skill_trigger_score = 0

    # <--- 重构: 使用统一、清晰的逻辑来触发和管理所有技能
    def update(self, maze, active_ghosts=None, pellet_grid=None, score=None):
        """Update Pac-Man's position and state."""
        prev_x, prev_y = self.x, self.y

        # 1. 管理已激活的技能状态（倒计时）
        if self.boost_mode:
            self.boost_timer -= 1
            if self.boost_timer <= 0:
                self.boost_mode = False
                self.speed = 3  # 恢复默认速度

        if self.invisible_mode:
            self.invisible_timer -= 1
            if self.invisible_timer <= 0:
                self.invisible_mode = False

        # 2. 统一的技能触发检查 (核心修正)
        if score is not None:
            # 检查当前分数是否比上次触发点高出200分
            if score - self.last_skill_trigger_score >= 200:

                print(f"DEBUG: 在分数 {score} 时，成功触发所有技能！")

                # a. 触发加速
                self.boost_mode = True
                self.boost_timer = 90  # 持续1.5秒 (1.5 * 60 FPS)
                self.speed = 5

                # b. 触发隐身
                self.invisible_mode = True
                self.invisible_timer = 90  # 持续1.5秒

                # c. 触发冰冻
                if active_ghosts:
                    for ghost in active_ghosts:
                        ghost.frozen = True
                        ghost.frozen_timer = 120  # 冻结2秒

                # 更新触发点，为下一次触发做准备
                self.last_skill_trigger_score += 200

        # 3. 移动和决策逻辑 (保持不变)
        if constants.GAME_MODE == "DQN" and self.dqn_model:
            current_cell = pixel_to_grid(self.x, self.y)
            # 确保 observation space 维度正确
            model_obs_dim = self.dqn_model.policy.observation_space.shape[0]
            observation = self._create_observation(active_ghosts, pellet_grid, maze)

            if len(observation) != model_obs_dim:
                # 如果维度不匹配，则可能需要截断或填充，但理想情况下应该匹配
                # 这里我们假设它匹配，因为 environment.py 中已修正
                pass

            action, _ = self.dqn_model.predict(observation, deterministic=True)
            action = int(action.item()) if hasattr(action, 'item') else int(action)
            self.last_action = action

            direction_map = {
                0: pygame.Vector2(1, 0),  # Right
                1: pygame.Vector2(-1, 0),  # Left
                2: pygame.Vector2(0, 1),  # Down
                3: pygame.Vector2(0, -1)  # Up
            }
            self.desired_direction = direction_map.get(action, pygame.Vector2(0, 0))  # 使用 .get() 增加安全性

            self.action_history.append(action)
            if len(self.action_history) > 10:
                self.action_history.pop(0)

            current_pos = pixel_to_grid(self.x, self.y)
            self.last_positions.append(current_pos)
            if len(self.last_positions) > 5:
                self.last_positions.pop(0)

        elif constants.GAME_MODE == "A_STAR":
            current_cell = pixel_to_grid(self.x, self.y)

            if self.path_finding_cooldown > 0:
                self.path_finding_cooldown -= 1

            if (not self.path or len(self.path) <= 1 or self.target_pellet is None or
                not pellet_grid[self.target_pellet[1]][self.target_pellet[0]]) and self.path_finding_cooldown == 0:

                best_goal = None
                best_distance = float('inf')

                for r in range(constants.ROWS):
                    for c in range(constants.COLS):
                        if pellet_grid[r][c]:
                            d = abs(current_cell[0] - c) + abs(current_cell[1] - r)
                            if d < best_distance:
                                best_distance = d
                                best_goal = (c, r)

                if best_goal is not None:
                    self.target_pellet = best_goal
                    self.path = a_star(current_cell, best_goal, maze)

                    if not self.path:
                        self.path_finding_cooldown = 10
                        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        random.shuffle(directions)
                        for dx, dy in directions:
                            test_x = self.x + dx * self.speed
                            test_y = self.y + dy * self.speed
                            if not collides_with_wall(test_x, test_y, self.radius, maze):
                                self.desired_direction = pygame.Vector2(dx, dy)
                                break

            if self.path and len(self.path) > 1:
                next_cell = self.path[1]
                next_pixel_x, next_pixel_y = grid_to_pixel(next_cell[0], next_cell[1])
                dx = next_pixel_x - self.x
                dy = next_pixel_y - self.y

                length = math.sqrt(dx * dx + dy * dy)
                if length > 0:
                    dx /= length
                    dy /= length

                if abs(dx) > abs(dy):
                    self.desired_direction = pygame.Vector2(1 if dx > 0 else -1, 0)
                else:
                    self.desired_direction = pygame.Vector2(0, 1 if dy > 0 else -1)

                if length < self.speed * 2:
                    self.path.pop(0)

        door_rect = None
        if constants.CURRENT_MAZE_TYPE == "COMPLEX":
            ghost_info = constants.get_ghost_house_info()
            door_rect = pygame.Rect(
                ghost_info["door_col_start"] * constants.TILE_SIZE,
                ghost_info["door_row"] * constants.TILE_SIZE,
                (ghost_info["door_col_end"] - ghost_info["door_col_start"] + 1) * constants.TILE_SIZE,
                constants.TILE_SIZE
            )

        if self.desired_direction.length_squared() > 0:
            test_x = self.x + self.desired_direction.x * self.speed
            test_y = self.y + self.desired_direction.y * self.speed
            test_rect = pygame.Rect(test_x - self.radius, test_y - self.radius, 2 * self.radius, 2 * self.radius)

            if not (constants.CURRENT_MAZE_TYPE == "COMPLEX" and door_rect and test_rect.colliderect(door_rect)):
                if not collides_with_wall(test_x, test_y, self.radius, maze):
                    self.direction = self.desired_direction

        if self.direction.length_squared() > 0:
            new_x = self.x + self.direction.x * self.speed
            new_y = self.y + self.direction.y * self.speed
            new_rect = pygame.Rect(new_x - self.radius, new_y - self.radius, 2 * self.radius, 2 * self.radius)

            if not (constants.CURRENT_MAZE_TYPE == "COMPLEX" and door_rect and new_rect.colliderect(door_rect)):
                if not collides_with_wall(new_x, new_y, self.radius, maze):
                    self.x = new_x
                    self.y = new_y

        if self.x < 0:
            self.x = constants.WIDTH
        elif self.x > constants.WIDTH:
            self.x = 0

        if abs(self.x - prev_x) < 0.1 and abs(self.y - prev_y) < 0.1 and self.direction.length_squared() > 0:
            self.wall_collision_count = getattr(self, 'wall_collision_count', 0) + 1
        else:
            self.wall_collision_count = 0

    # <--- 修正: 确保生成43维的特征向量
    def _create_observation(self, ghosts, pellet_grid=None, maze=None):
        """Create observation vector for DQN agent."""
        pacman_x, pacman_y = pixel_to_grid(self.x, self.y)
        norm_x = pacman_x / constants.COLS
        norm_y = pacman_y / constants.ROWS

        is_boosted = 1.0 if self.boost_mode else 0.0
        boost_timer_normalized = self.boost_timer / 120.0 if self.boost_mode else 0.0
        is_invisible = 1.0 if self.invisible_mode else 0.0
        invisible_timer_normalized = self.invisible_timer / 180.0 if self.invisible_mode else 0.0

        normal_ghosts = [g for g in ghosts if not g.frozen] if ghosts else []
        frozen_ghosts = [g for g in ghosts if g.frozen] if ghosts else []

        direction_vec = [0, 0, 0, 0]
        if self.direction.x > 0:
            direction_vec[0] = 1
        elif self.direction.x < 0:
            direction_vec[1] = 1
        elif self.direction.y > 0:
            direction_vec[2] = 1
        elif self.direction.y < 0:
            direction_vec[3] = 1

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        wall_sensors = [0, 0, 0, 0]
        if maze is not None:
            for i, (dx, dy) in enumerate(directions):
                for dist in range(1, 8):
                    check_x, check_y = pacman_x + (dx * dist), pacman_y + (dy * dist)
                    if not (0 <= check_y < constants.ROWS and 0 <= check_x < constants.COLS) or maze[check_y][
                        check_x] == 1:
                        wall_sensors[i] = 1.0 - ((dist - 1) / 7.0)
                        break

        pellet_sensors = [0, 0, 0, 0]
        if pellet_grid is not None and maze is not None:
            for i, (dx, dy) in enumerate(directions):
                for dist in range(1, 10):
                    check_x, check_y = pacman_x + (dx * dist), pacman_y + (dy * dist)
                    if 0 <= check_y < constants.ROWS and 0 <= check_x < constants.COLS:
                        if maze[check_y][check_x] == 1: break
                        if pellet_grid[check_y][check_x]:
                            if dist <= 3:
                                pellet_sensors[i] = 1.0
                            elif dist <= 6:
                                pellet_sensors[i] = 0.7
                            else:
                                pellet_sensors[i] = 0.4
                            break
                    else:
                        break

        normal_ghost_sensors = [0, 0, 0, 0]
        frozen_ghost_sensors = [0, 0, 0, 0]
        closest_normal_ghost_dist = 1.0
        closest_frozen_ghost_dist = 1.0

        if normal_ghosts:
            ghost_distances = sorted(
                [(g, abs(pixel_to_grid(g.x, g.y)[0] - pacman_x) + abs(pixel_to_grid(g.x, g.y)[1] - pacman_y)) for g in
                 normal_ghosts], key=lambda x: x[1])
            if ghost_distances:
                closest_normal_ghost_dist = min(1.0, ghost_distances[0][1] / 15.0)
                for ghost, dist in ghost_distances:
                    if dist > 10: continue
                    gx, gy = pixel_to_grid(ghost.x, ghost.y)
                    dx, dy = gx - pacman_x, gy - pacman_y
                    idx = (0 if dx > 0 else 1) if abs(dx) > abs(dy) else (2 if dy > 0 else 3)
                    normal_ghost_sensors[idx] = max(normal_ghost_sensors[idx], 1.0 - (dist / 10.0))

        if frozen_ghosts:
            ghost_distances = sorted(
                [(g, abs(pixel_to_grid(g.x, g.y)[0] - pacman_x) + abs(pixel_to_grid(g.x, g.y)[1] - pacman_y)) for g in
                 frozen_ghosts], key=lambda x: x[1])
            if ghost_distances:
                closest_frozen_ghost_dist = min(1.0, ghost_distances[0][1] / 15.0)
                for ghost, dist in ghost_distances:
                    if dist > 10: continue
                    gx, gy = pixel_to_grid(ghost.x, ghost.y)
                    dx, dy = gx - pacman_x, gy - pacman_y
                    idx = (0 if dx > 0 else 1) if abs(dx) > abs(dy) else (2 if dy > 0 else 3)
                    frozen_ghost_sensors[idx] = max(frozen_ghost_sensors[idx], 1.0 - (dist / 10.0))

        quadrant_pellets = [0, 0, 0, 0]
        if pellet_grid is not None and maze is not None:
            mid_col, mid_row = constants.COLS // 2, constants.ROWS // 2
            quad_counts = defaultdict(int)
            quad_totals = defaultdict(int)
            for r in range(constants.ROWS):
                for c in range(constants.COLS):
                    if maze[r][c] == 1: continue
                    quad = (1 if c < mid_col else 0) + (2 if r >= mid_row else 0)
                    quad_totals[quad] += 1
                    if pellet_grid[r][c]: quad_counts[quad] += 1
            for i in range(4):
                if quad_totals[i] > 0: quadrant_pellets[i] = quad_counts[i] / quad_totals[i]

        valid_moves = [0, 0, 0, 0]
        valid_move_count = 0
        if maze is not None:
            for i, (dx, dy) in enumerate(directions):
                next_x, next_y = pacman_x + dx, pacman_y + dy
                if 0 <= next_y < constants.ROWS and 0 <= next_x < constants.COLS and maze[next_y][next_x] == 0:
                    valid_move_count += 1
                    valid_moves[i] = 1.0
        is_junction = 1.0 if valid_move_count >= 3 else 0.0

        is_stuck = 1.0 if hasattr(self, 'last_positions') and len(self.last_positions) >= 3 and all(
            p == self.last_positions[-1] for p in self.last_positions[-3:]) else 0.0
        is_oscillating = 1.0 if hasattr(self, 'last_positions') and len(self.last_positions) >= 4 and \
                                self.last_positions[-1] == self.last_positions[-3] and self.last_positions[-2] == \
                                self.last_positions[-4] else 0.0

        movement_history = [0, 0, 0, 0]
        if hasattr(self, 'action_history') and self.action_history:
            counts = defaultdict(int)
            for action in self.action_history[-8:]:
                if action is not None: counts[action] += 1
            total = sum(counts.values())
            if total > 0: movement_history = [counts[i] / total for i in range(4)]

        # --- ASSEMBLE OBSERVATION ---
        # 4 + 22 + 17 = 43 total features
        pacman_status_features = np.array([
            is_boosted, boost_timer_normalized, is_invisible, invisible_timer_normalized
        ], dtype=np.float32)

        base_observation = np.array([
            norm_x, norm_y,
            *direction_vec,
            *wall_sensors,
            *pellet_sensors,
            *normal_ghost_sensors,
            *frozen_ghost_sensors
        ], dtype=np.float32)

        enhanced_features = np.array([
            *quadrant_pellets,
            is_junction,
            closest_normal_ghost_dist,
            closest_frozen_ghost_dist,
            is_stuck,
            is_oscillating,
            *valid_moves,
            *movement_history,
        ], dtype=np.float32)

        full_observation = np.concatenate([
            pacman_status_features,
            base_observation,
            enhanced_features
        ])

        return full_observation

    def draw(self, screen, score_font):
        # --------------------------
        # 设置颜色 + 透明度
        # --------------------------
        if self.invisible_mode:
            if self.boost_mode:
                color = (255, 140, 0, 128)  # 半透明橘色（隐身+加速）
            else:
                color = (255, 255, 0, 128)  # 半透明黄色（隐身）
        elif self.boost_mode:
            color = (255, 140, 0, 255)  # 不透明橘色（加速）
        else:
            color = (255, 255, 0, 255)  # 不透明黄色（正常）

        # 使用带透明度的 Surface 绘制 PacMan 身体
        pacman_surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(pacman_surface, color, (self.radius, self.radius), self.radius)
        screen.blit(pacman_surface, (self.x - self.radius, self.y - self.radius))

        # 嘴巴角度
        if self.direction.length_squared() > 0:
            angle = math.atan2(self.direction.y, self.direction.x)
        else:
            angle = 0

        mouth_angle_deg = 30
        half_mouth_rad = math.radians(mouth_angle_deg / 2)

        point1 = (self.x + self.radius * math.cos(angle + half_mouth_rad),
                  self.y + self.radius * math.sin(angle + half_mouth_rad))
        point2 = (self.x + self.radius * math.cos(angle - half_mouth_rad),
                  self.y + self.radius * math.sin(angle - half_mouth_rad))

        pygame.draw.polygon(screen, (0, 0, 0), [(self.x, self.y), point1, point2])

        if constants.GAME_MODE == "A_STAR" and self.path:
            for i in range(len(self.path) - 1):
                start_x, start_y = grid_to_pixel(self.path[i][0], self.path[i][1])
                end_x, end_y = grid_to_pixel(self.path[i + 1][0], self.path[i + 1][1])
                pygame.draw.line(screen, (255, 0, 255), (start_x, start_y), (end_x, end_y), 2)

        if constants.GAME_MODE == "A_STAR" and self.target_pellet:
            target_x, target_y = grid_to_pixel(self.target_pellet[0], self.target_pellet[1])
            pygame.draw.circle(screen, (255, 255, 0), (target_x, target_y), 8, 2)

        if constants.GAME_MODE == "DQN" and self.last_action is not None:
            action_names = ["→", "←", "↓", "↑"]
            if 0 <= self.last_action < len(action_names):
                text = score_font.render(action_names[self.last_action], True, (0, 255, 0))
                screen.blit(text, (self.x - 10, self.y - 30))

    def _get_valid_actions(self, maze):
        """Get list of valid actions from current position."""
        valid_actions = []
        directions = [
            (0, pygame.Vector2(1, 0)),  # Right
            (1, pygame.Vector2(-1, 0)),  # Left
            (2, pygame.Vector2(0, 1)),  # Down
            (3, pygame.Vector2(0, -1))  # Up
        ]

        door_rect = None
        if constants.CURRENT_MAZE_TYPE == "COMPLEX":
            ghost_info = constants.get_ghost_house_info()
            door_rect = pygame.Rect(
                ghost_info["door_col_start"] * constants.TILE_SIZE,
                ghost_info["door_row"] * constants.TILE_SIZE,
                (ghost_info["door_col_end"] - ghost_info["door_col_start"] + 1) * constants.TILE_SIZE,
                constants.TILE_SIZE
            )

        for action, direction in directions:
            test_x = self.x + direction.x * self.speed
            test_y = self.y + direction.y * self.speed
            test_rect = pygame.Rect(test_x - self.radius, test_y - self.radius, 2 * self.radius, 2 * self.radius)

            if not (constants.CURRENT_MAZE_TYPE == "COMPLEX" and door_rect and test_rect.colliderect(door_rect)):
                if not collides_with_wall(test_x, test_y, self.radius, maze):
                    valid_actions.append(action)

        return valid_actions

    def _get_nearby_ghosts(self, active_ghosts):
        """Get a list of nearby ghosts, sorted by distance"""
        if not active_ghosts:
            return []

        ghost_distances = []
        for ghost in active_ghosts:
            dist = math.hypot(ghost.x - self.x, ghost.y - self.y) / constants.TILE_SIZE
            ghost_distances.append((ghost, dist))

        ghost_distances.sort(key=lambda x: x[1])

        return [g for g, _ in ghost_distances]