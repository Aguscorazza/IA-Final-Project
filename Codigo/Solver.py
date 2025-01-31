import heapq
import random
import numpy as np
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from datetime import datetime

class Solver(QObject):
    solve_completed = pyqtSignal()

    def __init__(self, maze):
        super().__init__()
        self.maze = maze
        self.openSet = []
        self.cameFrom = {}
        self.g_score = {}
        self.f_score = {}
        self.controller = None  # To be set by the controller
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)
        self.endCell = None
        self.visited_count = 0
        self.start_time = 0
        self.end_time = 0

    def heuristic(self, current, target):
        raise NotImplementedError("Subclasses should implement heuristic.")

    def eval_function(self, values):
        raise NotImplementedError("Subclasses should implement eval_function.")

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
        raise NotImplementedError("Subclasses should implement solve.")

    def step(self):
        raise NotImplementedError("Subclasses should implement step.")


class AStarSolver(Solver):
    def __init__(self, maze):
        super().__init__(maze)

    def heuristic(self, current, target):
        (x1, y1), (x2, y2) = current, target
        #return np.sqrt((x1-x2)**2 + (y1-y2)**2) # Euclidean Distance
        return abs(x1 - x2) + abs(y1 - y2)      # Manhattan Distance

    def eval_function(self, values):
        if isinstance(values, list):
            return sum(values)
        elif isinstance(values, int):
            return values
        else:
            raise TypeError("Score values should be either a list or an integer")

    def solve(self):
        self.visited_count = 0
        self.start_time = datetime.now()
        start = self.maze.startCell
        self.endCell = self.maze.endCell

        if not start or not self.endCell:
            return None

        self.openSet = []
        # Push start node into the priority queue
        heapq.heappush(self.openSet, (0, start))
        self.cameFrom = {}

        self.g_score = {start: 0}  # Traveled distance from starting node to the starting node is zero
        self.f_score = {start: self.eval_function([self.g_score[start], self.heuristic(start, self.endCell)])}

        self.maze.clearSetNodes()
        self.timer.start(10)  # Adjust the interval as needed

    def step(self):
        if not self.openSet:
            self.timer.stop()
            return
        # Pop first element of the queue (node with minimum value of f_score)
        _, current = heapq.heappop(self.openSet)

        # Check if we've reached the goal
        if current == self.endCell:
            self.timer.stop()
            self.end_time = datetime.now()
            path = self.reconstruct_path(current)
            self.controller.maze_widget.path = path
            self.controller.maze_widget.path_stats["N_Visited"] = self.visited_count
            self.controller.maze_widget.path_stats["Time_Spent"] = self.end_time - self.start_time
            self.controller.maze_widget.path_stats["Route_Cost"] = self.g_score.get(current, float('inf'))
            self.solve_completed.emit()
            self.controller.maze_widget.update()
            return

        self.visited_count += 1
        self.maze.addClosedSetNode(*current)
        self.controller.maze_widget.update()
        for neighbor in self.get_neighbors(current):
            if neighbor in self.maze.closedSetNodes:
                continue
            # Get cost from the start node to the current node (defaulting to infinity if it's not found),
            # adds the cost of moving from the current node to the neighbor which is assumed to be 1
            tentative_g_score = self.g_score.get(current, float('inf')) + 1

            if tentative_g_score < self.g_score.get(neighbor, float('inf')):
                self.cameFrom[neighbor] = current
                self.g_score[neighbor] = tentative_g_score
                self.f_score[neighbor] = self.eval_function(
                    [self.g_score[neighbor], self.heuristic(neighbor, self.endCell)])
                # Adds the neighbor to the priority queue (open set) with its f_score as the priority.
                # The priority queue ensures nodes with the lowest f_score are explored first.
                heapq.heappush(self.openSet, (self.f_score[neighbor], neighbor))
                self.maze.addOpenSetNode(*neighbor)
                self.controller.maze_widget.update()


class ModifiedAStarSolver(Solver):
    def __init__(self, maze):
        super().__init__(maze)

    def heuristic(self, current, target):
        (x1, y1), (x2, y2) = current, target
        return abs(x1 - x2) + abs(y1 - y2)

    def eval_function(self, values):
        if isinstance(values, list):
            return sum(values)
        elif isinstance(values, int):
            return values
        else:
            raise TypeError("Score values should be either a list or an integer")

    def solve(self):
        self.visited_count = 0
        self.start_time = datetime.now()
        start = self.maze.startCell
        self.endCell = self.maze.endCell

        if not start or not self.endCell:
            return None

        self.openSet = []
        # Push start node into the priority queue
        heapq.heappush(self.openSet, (0, start))
        self.cameFrom = {}

        self.g_score = {start: 0}  # Traveled distance from starting node to the starting node is zero
        self.f_score = {start: self.eval_function([self.g_score[start], self.heuristic(start, self.endCell)])}

        self.maze.clearSetNodes()
        self.timer.start(10)  # Adjust the interval as needed

    def step(self):
        if not self.openSet:
            self.timer.stop()
            return
        # Pop first element of the queue (node with minimum value of f_score)
        _, current = heapq.heappop(self.openSet)

        if current == self.endCell:
            self.timer.stop()
            self.end_time = datetime.now()
            path = self.reconstruct_path(current)
            self.controller.maze_widget.path = path
            self.controller.maze_widget.path_stats["N_Visited"] = self.visited_count
            self.controller.maze_widget.path_stats["Time_Spent"] = self.end_time - self.start_time
            self.controller.maze_widget.path_stats["Route_Cost"] = self.g_score.get(current, float('inf'))
            self.solve_completed.emit()
            self.controller.maze_widget.update()
            return

        self.visited_count += 1
        self.maze.addClosedSetNode(*current)
        self.controller.maze_widget.update()

        for neighbor in self.get_neighbors(current):
            if neighbor in self.maze.closedSetNodes:
                continue
            # Get cost from the start node to the current node (defaulting to infinity if it's not found),
            # adds the cost of moving from the current node to the neighbor which is assumed to be 1
            tentative_g_score = self.g_score.get(current, float('inf')) + 1

            if tentative_g_score < self.g_score.get(neighbor, float('inf')):
                self.cameFrom[neighbor] = current
                self.g_score[neighbor] = tentative_g_score
                self.f_score[neighbor] = self.eval_function(
                    [self.g_score[neighbor], self.heuristic(neighbor, self.endCell),
                     self.heuristic(current, self.endCell)])
                # Adds the neighbor to the priority queue (open set) with its f_score as the priority.
                # The priority queue ensures nodes with the lowest f_score are explored first.
                heapq.heappush(self.openSet, (self.f_score[neighbor], neighbor))
                self.maze.addOpenSetNode(*neighbor)
                self.controller.maze_widget.update()


class DijkstraSolver(Solver):
    def __init__(self, maze):
        super().__init__(maze)

    def heuristic(self, current, target):
        # Dijkstra's algorithm does not use a heuristic
        return 0

    def eval_function(self, values):
        # Dijkstra's algorithm uses only the g_score
        if isinstance(values, list):
            return values[0]
        elif isinstance(values, int):
            return values
        else:
            raise TypeError("Score values should be either a list or an integer")

    def solve(self):
        self.visited_count = 0
        self.start_time = datetime.now()
        start = self.maze.startCell
        self.endCell = self.maze.endCell

        if not start or not self.endCell:
            return None

        self.openSet = []
        # Push start node into the priority queue
        heapq.heappush(self.openSet, (0, start))
        self.cameFrom = {}

        self.g_score = {start: 0}  # Traveled distance from starting node to the starting node is zero
        self.f_score = {start: self.eval_function([self.g_score[start]])}

        self.maze.clearSetNodes()
        self.timer.start(10)  # Adjust the interval as needed

    def step(self):
        if not self.openSet:
            self.timer.stop()
            return
        # Pop first element of the queue (node with minimum value of f_score)
        _, current = heapq.heappop(self.openSet)

        if current == self.endCell:
            self.timer.stop()
            self.end_time = datetime.now()
            path = self.reconstruct_path(current)
            self.controller.maze_widget.path = path
            self.controller.maze_widget.path_stats["N_Visited"] = self.visited_count
            self.controller.maze_widget.path_stats["Time_Spent"] = self.end_time - self.start_time
            self.controller.maze_widget.path_stats["Route_Cost"] = self.g_score.get(current, float('inf'))
            self.solve_completed.emit()
            self.controller.maze_widget.update()
            return

        self.visited_count += 1
        self.maze.addClosedSetNode(*current)
        self.controller.maze_widget.update()

        for neighbor in self.get_neighbors(current):
            if neighbor in self.maze.closedSetNodes:
                continue
            # Get cost from the start node to the current node (defaulting to infinity if it's not found),
            # adds the cost of moving from the current node to the neighbor which is assumed to be 1
            tentative_g_score = self.g_score.get(current, float('inf')) + 1

            if tentative_g_score < self.g_score.get(neighbor, float('inf')):
                self.cameFrom[neighbor] = current
                self.g_score[neighbor] = tentative_g_score
                self.f_score[neighbor] = self.eval_function([self.g_score[neighbor]])
                # Adds the neighbor to the priority queue (open set) with its f_score as the priority.
                # The priority queue ensures nodes with the lowest f_score are explored first.
                heapq.heappush(self.openSet, (self.f_score[neighbor], neighbor))
                self.maze.addOpenSetNode(*neighbor)
                self.controller.maze_widget.update()
