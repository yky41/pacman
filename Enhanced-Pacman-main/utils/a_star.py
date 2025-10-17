# utils/astar.py
import heapq

def heuristic(cell, goal):
    """Calculate Manhattan distance between two cells."""
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

def get_neighbours(cell, maze):
    """Return navigable neighbouring cells of the given cell."""
    neighbours = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    col, row = cell
    for d in directions:
        ncol = col + d[0]
        nrow = row + d[1]
        if 0 <= nrow < len(maze) and 0 <= ncol < len(maze[0]) and maze[nrow][ncol] == 0:
            neighbours.append((ncol, nrow))
    return neighbours

def a_star(start, goal, maze):
    """Find optimal path from start to goal using A* algorithm."""
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
        
        if current in closed_set:
            continue
        closed_set.add(current)
        
        for neighbour in get_neighbours(current, maze):
            tentative_g_score = g_score[current] + 1
            if neighbour in closed_set and tentative_g_score >= g_score.get(neighbour, float('inf')):
                continue
            if tentative_g_score < g_score.get(neighbour, float('inf')):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g_score
                f_score[neighbour] = tentative_g_score + heuristic(neighbour, goal)
                heapq.heappush(open_set, (f_score[neighbour], neighbour))
    return []  # No path found
