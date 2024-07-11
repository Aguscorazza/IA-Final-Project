from PyQt5.QtWidgets import QWidget, QMainWindow, QVBoxLayout, QPushButton, QRadioButton, QHBoxLayout, QButtonGroup
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt
from Solver import AStarSolver, DijkstraSolver, ModifiedAStarSolver


class MazeWidget(QWidget):
    def __init__(self, maze, controller):
        super().__init__()
        self.maze = maze
        self.controller = controller
        self.path = []
        self.start_set = False
        self.end_set = False
        self.cell_size = 20
        self.drawing_wall = False
        self.closedSet = None

    def paintEvent(self, event):
        painter = QPainter(self)
        font = QFont("Arial", 8)
        painter.setFont(font)

        for rowID in range(self.maze.nbRows):
            for columnID in range(self.maze.nbColumns):
                if self.maze.grid[rowID][columnID] == 1:
                    painter.setBrush(QColor(0, 0, 0))
                elif self.maze.grid[rowID][columnID] == 2:
                    painter.setBrush(QColor(0, 255, 0))
                elif self.maze.grid[rowID][columnID] == 3:
                    painter.setBrush(QColor(255, 0, 0))
                elif (rowID, columnID) in self.maze.closedSetNodes:
                    painter.setBrush(QColor(100, 100, 100))  # Dark red for closed set nodes
                elif (rowID, columnID) in self.maze.openSetNodes:
                    painter.setBrush(QColor(0, 0, 200))  # Dark blue for open set nodes
                else:
                    painter.setBrush(QColor(255, 255, 255))
                painter.drawRect(columnID * self.cell_size, rowID * self.cell_size, self.cell_size, self.cell_size)

        painter.setBrush(QColor(255, 255, 0))
        for rowID, columnID in self.path:
            painter.drawRect(columnID * self.cell_size, rowID * self.cell_size, self.cell_size, self.cell_size)

    def mousePressEvent(self, event):
        widget_pos = self.mapFromGlobal(event.globalPos())
        columnID = widget_pos.x() // self.cell_size
        rowID = widget_pos.y() // self.cell_size

        if rowID < 0 or rowID > self.maze.nbRows or columnID < 0 or columnID > self.maze.nbColumns:
            return  # Ignore clicks outside the maze grid

        if not self.start_set:
            self.maze.setStartCell(rowID, columnID)
            self.start_set = True
        elif not self.end_set:
            self.maze.setEndCell(rowID, columnID)
            self.end_set = True
        else:
            self.drawing_wall = True
            self.maze.setWallCell(rowID, columnID)

        self.update()

    def mouseMoveEvent(self, event):
        if self.drawing_wall:
            widget_pos = self.mapFromGlobal(event.globalPos())
            columnID = widget_pos.x() // self.cell_size
            rowID = widget_pos.y() // self.cell_size

            if 0 <= rowID < self.maze.nbRows and  0 <= columnID < self.maze.nbColumns:
                self.maze.setWallCell(rowID, columnID)
                self.update()

    def mouseReleaseEvent(self, event):
        self.drawing_wall = False

class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.setWindowTitle("Maze Solver")
        self.setGeometry(100, 100, 800, 600)

        self.maze_widget = MazeWidget(self.controller.maze, self.controller)
        self.solve_button = QPushButton("Solve Maze")
        self.solve_button.clicked.connect(self.controller.solve_maze)

        self.astar_radio = QRadioButton("A* Solver")
        self.dijkstra_radio = QRadioButton("Dijkstra Solver")
        self.modified_astar_radio = QRadioButton("Modified A* Solver")
        self.astar_radio.setChecked(True)

        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.astar_radio)
        self.radio_group.addButton(self.dijkstra_radio)
        self.radio_group.addButton(self.modified_astar_radio)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.astar_radio)
        radio_layout.addWidget(self.dijkstra_radio)
        radio_layout.addWidget(self.modified_astar_radio)

        layout = QVBoxLayout()
        layout.addWidget(self.maze_widget)
        layout.addLayout(radio_layout)
        layout.addWidget(self.solve_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.controller.set_solver(self.get_selected_solver())

        self.astar_radio.toggled.connect(self.on_solver_changed)
        self.dijkstra_radio.toggled.connect(self.on_solver_changed)
        self.modified_astar_radio.toggled.connect(self.on_solver_changed)

    def get_selected_solver(self):
        if self.astar_radio.isChecked():
            return AStarSolver(self.controller.maze)
        elif self.dijkstra_radio.isChecked():
            return DijkstraSolver(self.controller.maze)
        elif self.modified_astar_radio.isChecked():
            return ModifiedAStarSolver(self.controller.maze)

    def on_solver_changed(self):
        self.controller.set_solver(self.get_selected_solver())

    def solve_maze(self):
        self.controller.solve_maze()
