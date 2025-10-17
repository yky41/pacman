import pygame
import constants

def draw_maze(screen, maze):
    """Draw maze on the screen."""
    wall_colour = (0, 0, 255)  # Blue walls
    
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            if maze[row][col] == 1:  # Wall
                rect = pygame.Rect(
                    col * constants.TILE_SIZE,
                    row * constants.TILE_SIZE,
                    constants.TILE_SIZE,
                    constants.TILE_SIZE
                )
                pygame.draw.rect(screen, wall_colour, rect)


def draw_pellets(screen, pellet_grid):
    """Draw pellets on the maze."""
    pellet_radius = 4
    for row in range(len(pellet_grid)):
        for col in range(len(pellet_grid[row])):
            if pellet_grid[row][col]:
                centre_x = col * constants.TILE_SIZE + constants.TILE_SIZE // 2
                centre_y = row * constants.TILE_SIZE + constants.TILE_SIZE // 2
                pygame.draw.circle(screen, (255, 255, 255), (centre_x, centre_y), pellet_radius)


def draw_ghost_house_barrier(screen):
    """Draw ghost house entrance barrier for complex maze."""
    if constants.CURRENT_MAZE_TYPE == "COMPLEX":
        ghost_info = constants.get_ghost_house_info()
        
        door_left = ghost_info["door_col_start"] * constants.TILE_SIZE
        door_top = ghost_info["door_row"] * constants.TILE_SIZE
        door_width = (ghost_info["door_col_end"] - ghost_info["door_col_start"] + 1) * constants.TILE_SIZE
        door_height = 3
        
        pygame.draw.rect(screen, (255, 255, 255), (door_left, door_top, door_width, door_height))


def draw_game_ui(screen, score, game_mode, maze_type, font, using_imitation=False):
    """Draw game UI including score and mode indicators."""
    screen_width = constants.WIDTH
    screen_height = constants.HEIGHT
    
    if screen_width < 600:
        small_font = pygame.font.SysFont(None, 24)
    else:
        small_font = font
    
    score_text = font.render("Score: " + str(score), True, (255, 255, 255))
    screen.blit(score_text, (10, 10))
    
    if game_mode == "HUMAN":
        mode_text = small_font.render("Human Mode", True, (255, 255, 255))
    elif game_mode == "A_STAR":
        mode_text = small_font.render("A* Mode", True, (255, 255, 0))
    elif game_mode == "DQN":
        mode_text = small_font.render("DQN Mode", True, (0, 255, 0))
    elif game_mode == "IMITATION":
        mode_text = small_font.render("Imitation Mode", True, (0, 255, 255))
    else:
        mode_text = small_font.render(f"{game_mode} Mode", True, (255, 150, 150))
    
    mode_rect = mode_text.get_rect()
    mode_rect.topright = (screen_width - 10, 10)
    screen.blit(mode_text, mode_rect)
    
    maze_text = small_font.render(f"Maze: {maze_type}", True, (255, 150, 0))
    maze_rect = maze_text.get_rect()
    maze_rect.topright = (screen_width - 10, mode_rect.bottom + 5)
    screen.blit(maze_text, maze_rect)
    
    controls_text = small_font.render("Keys: 1=Human 2=A* 3=DQN 4=Imitation M=SwitchMaze", True, (200, 200, 200))
    controls_rect = controls_text.get_rect()
    controls_rect.midbottom = (screen_width // 2, screen_height - 10)
    screen.blit(controls_text, controls_rect)


def draw_game_over(screen, game_won):
    """Draw game over or victory screen with restart button."""
    large_font = pygame.font.SysFont(None, 72)
    button_font = pygame.font.SysFont(None, 48)
    
    overlay = pygame.Surface((constants.WIDTH, constants.HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))
    
    if game_won:
        text = large_font.render("You Win!", True, (255, 255, 255))
    else:
        text = large_font.render("Game Over", True, (255, 255, 255))
        
    text_rect = text.get_rect(center=(constants.WIDTH // 2, constants.HEIGHT // 2 - 50))
    screen.blit(text, text_rect)
    
    button_text = button_font.render("Restart", True, (255, 255, 255))
    button_width = 200
    button_height = 60
    button_x = constants.WIDTH // 2 - button_width // 2
    button_y = constants.HEIGHT // 2 + 30
    
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    pygame.draw.rect(screen, (0, 100, 255), button_rect)
    pygame.draw.rect(screen, (0, 150, 255), button_rect, 4)
    
    text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, text_rect)
    
    instruction_font = pygame.font.SysFont(None, 32)
    instruction_text = instruction_font.render("Press SPACE or click to restart", True, (200, 200, 200))
    instruction_rect = instruction_text.get_rect(center=(constants.WIDTH // 2, button_y + button_height + 30))
    screen.blit(instruction_text, instruction_rect)
    
    return button_rect


def render_game(screen, maze, pellet_grid, pacman, active_ghosts, score, game_mode, game_over, game_won, font, maze_type="COMPLEX", using_imitation=False):
    """Render all game elements and return restart button if game is over."""
    screen.fill((0, 0, 0))
    
    draw_maze(screen, maze)
    draw_pellets(screen, pellet_grid)
    
    if maze_type == "COMPLEX":
        draw_ghost_house_barrier(screen)
    
    pacman.draw(screen, font)
    for ghost in active_ghosts:
        ghost.draw(screen)
    
    draw_game_ui(screen, score, game_mode, maze_type, font, using_imitation)
    
    if game_over or game_won:
        return draw_game_over(screen, game_won)
    
    return None