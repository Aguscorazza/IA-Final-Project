import sys
from PyQt5.QtWidgets import QApplication
from Maze import Maze
from Solver import AStarSolver
from main_window import MainWindow
from controller import MazeController

def main():
    print("Starting application...")
    app = QApplication(sys.argv)

    maze = Maze(20, 20)
    solver = AStarSolver(maze)
    controller = MazeController(maze, solver)

    window = MainWindow(controller)
    controller.maze_widget = window.maze_widget  # Link the view to the controller
    window.show()

    print("Application started.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
