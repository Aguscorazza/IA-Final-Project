import heapq


class AStarSolver:
    def __init__(self, maze):
        self.maze = maze
        self.open_set = []
        self.came_from = {}
        self.g_score = {}
        self.f_score = {}

    def heuristic(self, a, b):
        (x1, y1), (x2, y2) = a, b
        return abs(x1 - x2) + abs(y1 - y2)

    def get_neighbors(self, node):
        neighbors = []
        x, y = node
        if self.maze.isWithinBounds(x+1, y) and self.maze.isWalkable(x+1, y):
            neighbors.append((x+1, y))
        if self.maze.isWithinBounds(x-1, y) and self.maze.isWalkable(x-1, y):
            neighbors.append((x-1, y))
        if self.maze.isWithinBounds(x, y+1) and self.maze.isWalkable(x, y+1):
            neighbors.append((x, y+1))
        if self.maze.isWithinBounds(x, y-1) and self.maze.isWalkable(x, y-1):
            neighbors.append((x, y-1))
        return neighbors

    def reconstruct_path(self, current):
        path = []
        while current in self.came_from:
            path.append(current)
            current = self.came_from[current]
        path.reverse()
        return path

    def solve(self):
        start = self.maze.startCell
        end = self.maze.endCell

        self.open_set = []
        heapq.heappush(self.open_set, (0, start))
        self.came_from = {}
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start, end)}

        while self.open_set:
            _, current = heapq.heappop(self.open_set)

            if current == end:
                return self.reconstruct_path(current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = self.g_score.get(current, float('inf')) + 1

                if tentative_g_score < self.g_score.get(neighbor, float('inf')):
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
                    heapq.heappush(self.open_set, (self.f_score[neighbor], neighbor))

        return None  # No path found
