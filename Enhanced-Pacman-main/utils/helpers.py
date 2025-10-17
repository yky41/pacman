# utils/helpers.py
import pygame
import constants

def draw_maze(screen, maze):
    """Draw maze walls on the screen."""
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            if maze[row][col] == 1:
                rect = pygame.Rect(col * constants.TILE_SIZE, row * constants.TILE_SIZE, constants.TILE_SIZE, constants.TILE_SIZE)
                pygame.draw.rect(screen, (0, 0, 255), rect)

def collides_with_wall(x, y, radius, maze):
    """Check if a circular object at (x, y) collides with any wall cell."""
    obj_rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            if maze[row][col] == 1:
                wall_rect = pygame.Rect(col * constants.TILE_SIZE, row * constants.TILE_SIZE, constants.TILE_SIZE, constants.TILE_SIZE)
                if obj_rect.colliderect(wall_rect):
                    return True
    return False

def grid_to_pixel(col, row):
    """Convert grid coordinates to pixel coordinates (centre of cell)."""
    return col * constants.TILE_SIZE + constants.TILE_SIZE // 2, row * constants.TILE_SIZE + constants.TILE_SIZE // 2

def pixel_to_grid(x, y):
    """Convert pixel coordinates to grid coordinates."""
    return int(x // constants.TILE_SIZE), int(y // constants.TILE_SIZE)

def draw_pellets(screen, pellet_grid):
    """Draw pellets on the maze."""
    pellet_radius = 4
    for row in range(len(pellet_grid)):
        for col in range(len(pellet_grid[row])):
            if pellet_grid[row][col]:
                centre_x = col * constants.TILE_SIZE + constants.TILE_SIZE // 2
                centre_y = row * constants.TILE_SIZE + constants.TILE_SIZE // 2
                pygame.draw.circle(screen, (255, 255, 255), (centre_x, centre_y), pellet_radius)

def is_in_ghost_house(tile_x, tile_y):
    """Check if the tile is within the ghost house region."""
    ghost_info = constants.get_ghost_house_info()
    return (ghost_info["house_col_start"] <= tile_x <= ghost_info["house_col_end"] and 
            ghost_info["house_row_start"] <= tile_y <= ghost_info["house_row_end"])

def is_ghost_house_door(tile_x, tile_y):
    """Check if the tile is in the door region."""
    ghost_info = constants.get_ghost_house_info()
    return (tile_y == ghost_info["door_row"] and 
            ghost_info["door_col_start"] <= tile_x <= ghost_info["door_col_end"])

def check_collision_pacman_ghost(pacman, ghost):
    """Check if the ghost touches Pac-Man."""
    distance = pygame.Vector2(pacman.x - ghost.x, pacman.y - ghost.y).length()
    return distance < (pacman.radius + ghost.radius)