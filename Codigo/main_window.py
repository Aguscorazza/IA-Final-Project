from PyQt5.QtWidgets import QWidget, QMainWindow, QVBoxLayout, QPushButton, QRadioButton, QHBoxLayout, QButtonGroup, \
    QDesktopWidget, QLabel
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
        self.path_stats = {}
        self.setMinimumSize(800, 800)  # Set a minimum size for the MazeWidget

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
            if self.maze.grid[rowID][columnID] == 0:
                self.maze.setStartCell(rowID, columnID)
                self.start_set = True
        elif not self.end_set:
            if self.maze.grid[rowID][columnID] == 0:
                self.maze.setEndCell(rowID, columnID)
                self.end_set = True
        else:
            self.drawing_wall = True
            if self.maze.grid[rowID][columnID] == 0:
                self.maze.setWallCell(rowID, columnID)

        self.update()

    def mouseMoveEvent(self, event):
        if self.drawing_wall:
            widget_pos = self.mapFromGlobal(event.globalPos())
            columnID = widget_pos.x() // self.cell_size
            rowID = widget_pos.y() // self.cell_size

            if 0 <= rowID < self.maze.nbRows and 0 <= columnID < self.maze.nbColumns:
                if self.maze.grid[rowID][columnID] == 0:
                    self.maze.setWallCell(rowID, columnID)
                    self.update()

    def mouseReleaseEvent(self, event):
        self.drawing_wall = False

    def clearMaze(self):
        self.start_set = False
        self.end_set = False
        self.path = []
        self.update()


class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.setWindowTitle("Maze Generator & Solver")
        # Get screen resolution
        screen_resolution = QDesktopWidget().screenGeometry()
        width, height = int(screen_resolution.width() * 0.80), int(screen_resolution.height() * 0.85)

        title_font = QFont("Arial", 15)
        normal_font = QFont("Arial", 10)

        self.setGeometry(150, 100, width, height)

        self.maze_widget = MazeWidget(self.controller.maze, self.controller)
        self.solve_button = QPushButton("Solve Maze")
        self.solve_button.setFont(normal_font)
        self.solve_button.setStyleSheet("background-color: #CCD2CD;")
        self.solve_button.clicked.connect(self.controller.solve_maze)

        self.generate_button = QPushButton("Generate Maze")
        self.generate_button.setFont(normal_font)
        self.generate_button.setStyleSheet("background-color: #CCD2CD;")
        self.generate_button.clicked.connect(self.generate_maze)

        self.clear_button = QPushButton("Clear Maze")
        self.clear_button.setFont(normal_font)
        self.clear_button.setStyleSheet("background-color: #CCD2CD;")
        self.clear_button.clicked.connect(self.clear_maze)

        self.title_label = QLabel("Maze Generator & Solver")
        self.title_label.setFont(title_font)
        self.title_label.setContentsMargins(0, 0, 0, 10)

        self.solver_label = QLabel("Choose Maze Solver: ")
        self.solver_label.setFont(normal_font)
        self.solver_label.setContentsMargins(5, 5, 0, 0)

        self.astar_radio = QRadioButton("A* Solver")
        self.astar_radio.setFont(normal_font)
        self.dijkstra_radio = QRadioButton("Dijkstra Solver")
        self.dijkstra_radio.setFont(normal_font)
        self.modified_astar_radio = QRadioButton("Modified A* Solver")
        self.modified_astar_radio.setFont(normal_font)

        self.astar_radio.setChecked(True)

        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.astar_radio)
        self.radio_group.addButton(self.dijkstra_radio)
        self.radio_group.addButton(self.modified_astar_radio)

        radio_layout = QVBoxLayout()
        radio_layout.addWidget(self.astar_radio)
        radio_layout.addWidget(self.dijkstra_radio)
        radio_layout.addWidget(self.modified_astar_radio)
        radio_layout.setContentsMargins(20, 10, 10, 30)

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(30, 0, 30, 0)
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.addWidget(self.title_label, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.solver_label)
        left_layout.addLayout(radio_layout)
        left_layout.addWidget(self.solve_button)
        left_layout.addWidget(self.generate_button)
        left_layout.addWidget(self.clear_button)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=2)

        # Create a centered layout for the maze widget
        centered_layout = QVBoxLayout()
        centered_layout.addWidget(self.maze_widget, alignment=Qt.AlignCenter)
        main_layout.addLayout(centered_layout, stretch=4)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.controller.set_solver(self.get_selected_solver())

        self.astar_radio.toggled.connect(self.on_solver_changed)
        self.dijkstra_radio.toggled.connect(self.on_solver_changed)
        self.modified_astar_radio.toggled.connect(self.on_solver_changed)

        self.setStyleSheet("background-color: lightblue;")

        # Add label to show path_stats
        self.path_stats_label = QLabel()
        self.path_stats_label.setContentsMargins(15, 10, 0, 0)
        self.path_stats_label.setFont(normal_font)
        left_layout.addWidget(self.path_stats_label)

        self.controller.solver.solve_completed.connect(self.update_path_stats_label)

    def update_path_stats_label(self):
        path_stats = self.maze_widget.path_stats
        text = f"""Solver Information: 
        \tNumber of visited cells: {path_stats["N_Visited"]}
        \tTime Spent: {path_stats["Time_Spent"]}"""
        self.path_stats_label.setText(text)

    def get_selected_solver(self):
        if self.astar_radio.isChecked():
            return AStarSolver(self.controller.maze)
        elif self.dijkstra_radio.isChecked():
            return DijkstraSolver(self.controller.maze)
        elif self.modified_astar_radio.isChecked():
            return ModifiedAStarSolver(self.controller.maze)

    def on_solver_changed(self):
        self.controller.set_solver(self.get_selected_solver())
        self.controller.solver.solve_completed.connect(self.update_path_stats_label)

    def solve_maze(self):
        self.controller.solve_maze()

    def clear_maze(self):
        self.controller.clear_maze()
        self.maze_widget.clearMaze()

    def generate_maze(self):
        self.maze_widget.clearMaze()
        self.controller.generate_maze()
        self.maze_widget.update()
