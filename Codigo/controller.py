class MazeController:
    def __init__(self, maze):
        self.maze = maze
        self.solver = None
        self.maze_widget = None  # Initialize it later

    def set_solver(self, solver):
        self.solver = solver
        self.solver.controller = self  # Link solver to controller

    def solve_maze(self):
        if self.solver:
            self.solver.solve()
