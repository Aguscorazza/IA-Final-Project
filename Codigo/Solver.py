import heapq
from PyQt5.QtCore import QTimer


class AStarSolver:
    def __init__(self, maze):
        self.maze = maze
        self.openSet = []
        self.cameFrom = {}
        self.g_score = {}
        self.f_score = {}
        self.controller = None  # To be set by the controller
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)
        self.endCell = None

    def heuristic(self, a, b):
        (x1, y1), (x2, y2) = a, b
        return abs(x1 - x2) + abs(y1 - y2)

    def get_neighbors(self, node):
        neighbors = []
        rowID, columnID = node
        if self.maze.isWithinBounds(rowID + 1, columnID) and self.maze.isWalkable(rowID + 1, columnID):
            neighbors.append((rowID + 1, columnID))
        if self.maze.isWithinBounds(rowID - 1, columnID) and self.maze.isWalkable(rowID - 1, columnID):
            neighbors.append((rowID - 1, columnID))
        if self.maze.isWithinBounds(rowID, columnID + 1) and self.maze.isWalkable(rowID, columnID + 1):
            neighbors.append((rowID, columnID + 1))
        if self.maze.isWithinBounds(rowID, columnID - 1) and self.maze.isWalkable(rowID, columnID - 1):
            neighbors.append((rowID, columnID - 1))
        return neighbors

    def reconstruct_path(self, current):
        path = []
        while current in self.cameFrom:
            path.append(current)
            current = self.cameFrom[current]
        path.reverse()
        return path

    def solve(self):
        start = self.maze.startCell
        self.endCell = self.maze.endCell

        if not start or not self.endCell:
            return None

        self.openSet = []
        # Push start node into the priority queue
        heapq.heappush(self.openSet, (0, start))
        self.cameFrom = {}

        self.g_score = {start: 0}  # Traveled distance from starting node to the starting node is zero
        self.f_score = {start: self.heuristic(start, self.endCell)}

        self.maze.clearSetNodes()
        self.timer.start(10)  # Adjust the interval as needed

    def step(self):
        if not self.openSet:
            self.timer.stop()
            return
        # Pop first element of the queue (node with minimum value of f_score)
        _, current = heapq.heappop(self.openSet)

        self.maze.addClosedSetNode(*current)
        self.controller.maze_widget.update()

        for neighbor in self.get_neighbors(current):
            if neighbor == self.endCell:  # Skip further exploration if neighbor is the destination
                self.timer.stop()
                self.cameFrom[neighbor] = current
                path = self.reconstruct_path(current)
                self.controller.maze_widget.path = path
                self.controller.maze_widget.update()
                return

            # Get cost from the start node to the current node (defaulting to infinity if it's not found),
            # adds the cost of moving from the current node to the neighbor which is assumed to be 1
            tentative_g_score = self.g_score.get(current, float('inf')) + 1

            if tentative_g_score < self.g_score.get(neighbor, float('inf')):
                self.cameFrom[neighbor] = current
                self.g_score[neighbor] = tentative_g_score
                self.f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, self.endCell)
                # Adds the neighbor to the priority queue (open set) with its f_score as the priority.
                # The priority queue ensures nodes with the lowest f_score are explored first.
                heapq.heappush(self.openSet, (self.f_score[neighbor], neighbor))
                self.maze.addOpenSetNode(*neighbor)
                self.controller.maze_widget.update()
