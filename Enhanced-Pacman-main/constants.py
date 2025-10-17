# Game constants
TILE_SIZE = 30
CHASE_THRESHOLD = TILE_SIZE * 4
GAME_MODE = "DQN"  # Default game mode
# GAME_MODE = "HUMAN"  # Default game mode
# GAME_MODE = "A_STAR"  # Default game mode

# Complex maze layout
COMPLEX_MAZE_LAYOUT = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "     #.##### ## #####.#     ",
    "     #.##          ##.#     ",
    "     #.## ###  ### ##.#     ",
    "######.## #      # ##.######",
    "      .   #      #   .      ",
    "######.## ######## ##.######",
    "     #.##          ##.#     ",
    "     #.## ######## ##.#     ",
    "     #.## ######## ##.#     ",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#o..##.......  .......##..o#",
    "###.##.##.########.##.##.###",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#.##########.##.##########.#",
    "#............##............#",
    "############################"
]

# Simple maze layout
SIMPLE_MAZE_LAYOUT = [
    "##############",
    "#............#",
    "#.#####.####.#",
    "#.#........#.#",
    "#.#.##.###.#.#",
    "#.#.#....#.#.#",
    "#...#.##.#...#",
    "#.###.##.###.#",
    "#.#........#.#",
    "#...######...#",
    "#.#........#.#",
    "#.#####.####.#",
    "#............#",
    "##############"
]

# Ghost configuration
SIMPLE_MAZE_GHOST_COUNT = 2  # Number of ghosts in simple maze

# Current maze settings
CURRENT_MAZE_TYPE = "SIMPLE"  # "COMPLEX" or "SIMPLE"
MAZE_LAYOUT = COMPLEX_MAZE_LAYOUT  # Default maze layout

def set_maze_type(maze_type):
    """Update the current maze layout based on type"""
    global CURRENT_MAZE_TYPE, MAZE_LAYOUT, ROWS, COLS, WIDTH, HEIGHT
    
    if maze_type == "COMPLEX":
        MAZE_LAYOUT = COMPLEX_MAZE_LAYOUT
        CURRENT_MAZE_TYPE = "COMPLEX"
    elif maze_type == "SIMPLE":
        MAZE_LAYOUT = SIMPLE_MAZE_LAYOUT
        CURRENT_MAZE_TYPE = "SIMPLE"
    
    # Update dimensions based on the new maze
    ROWS = len(MAZE_LAYOUT)
    COLS = len(MAZE_LAYOUT[0])
    WIDTH = COLS * TILE_SIZE
    HEIGHT = ROWS * TILE_SIZE

# Initialise dimensions
ROWS = len(MAZE_LAYOUT)
COLS = len(MAZE_LAYOUT[0])
WIDTH = COLS * TILE_SIZE
HEIGHT = ROWS * TILE_SIZE

def get_ghost_house_info():
    """Return ghost house information based on current maze"""
    if CURRENT_MAZE_TYPE == "COMPLEX":
        return {
            "spawn": (13, 14),      # Ghost spawn position (col, row)
            "door_col_start": 13,   # Door start column
            "door_col_end": 15,     # Door end column
            "door_row": 12,         # Door row
            "house_col_start": 10,  # Ghost house start column
            "house_col_end": 16,    # Ghost house end column
            "house_row_start": 11,  # Ghost house start row
            "house_row_end": 18     # Ghost house end row
        }
    else:  # SIMPLE maze
        return {
            "spawn": (7, 6),        # Ghost spawn position (col, row)
            "door_col_start": 7,    # Door start column 
            "door_col_end": 7,      # Door end column
            "door_row": 7,          # Door row
            "house_col_start": 6,   # Ghost house start column
            "house_col_end": 8,     # Ghost house end column
            "house_row_start": 5,   # Ghost house start row
            "house_row_end": 7      # Ghost house end row
        }
    
def get_pacman_spawn():
    """Return Pac-Man spawn position based on current maze"""
    if CURRENT_MAZE_TYPE == "COMPLEX":
        return (13, 23)  # Complex maze spawn (col, row)
    else:
        return (7, 12)   # Simple maze spawn (col, row)