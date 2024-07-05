class MazeController:
    def __init__(self, maze, solver):
        self.maze = maze
        self.solver = solver
        self.maze_widget = None  # Initialize it later

    def solve_maze(self):
        print("Solving maze...")
        self.solver.maze = self.maze
        path = self.solver.solve()
        print(f"Path found: {path}")
        self.maze_widget.path = path or []
        self.maze_widget.update()
