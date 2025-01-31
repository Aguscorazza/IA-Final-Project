from PyQt5.QtWidgets import QWidget, QMainWindow, QVBoxLayout, QPushButton, QRadioButton, QHBoxLayout, QButtonGroup, \
    QDesktopWidget, QLabel, QAction, QToolBar, QStackedWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt
from Solver import AStarSolver, DijkstraSolver, ModifiedAStarSolver
from STRIPSBoxOrderWidget import BoxOrderWindow
from ObjectRecognitionWindow import ObjectRecognitionWindow
from MazeWidget import  MazeWidget

class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.setWindowTitle("Maze Generator & Solver")
        # Get screen resolution
        screen_resolution = QDesktopWidget().screenGeometry()
        width, height = int(screen_resolution.width()), int(screen_resolution.height())

        title_font = QFont("Arial", 15)
        normal_font = QFont("Arial", 10)

        self.setGeometry(0, 0, width, height)

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


        # STRIPS Box Order Widget
        container = QWidget()
        container.setLayout(main_layout)

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(container)
        self.box_order_window = BoxOrderWindow(self.controller)
        self.stacked_widget.addWidget(self.box_order_window)
        self.object_recognition_window = ObjectRecognitionWindow(controller)
        self.stacked_widget.addWidget(self.object_recognition_window)

        self.setCentralWidget(self.stacked_widget)
        self.controller.set_solver(self.get_selected_solver())

        self.setCentralWidget(self.stacked_widget)

        self.controller.set_solver(self.get_selected_solver())

        self.astar_radio.toggled.connect(self.on_solver_changed)
        self.dijkstra_radio.toggled.connect(self.on_solver_changed)
        self.modified_astar_radio.toggled.connect(self.on_solver_changed)

        self.setStyleSheet("background-color: lightblue;")

        # Add label to show path_stats
        self.path_stats_label = QLabel()
        self.path_stats_label.setContentsMargins(15, 10, 0, 0)
        self.path_stats_label.setFont(normal_font)
        self.path_stats_label.setStyleSheet("color: black; font-size: 20px; padding-top: 20px")
        left_layout.addWidget(self.path_stats_label)

        self.controller.solver.solve_completed.connect(self.update_path_stats_label)

        self.create_toolbar()

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setStyleSheet("background-color: #AAAAAA;")
        self.addToolBar(toolbar)

        maze_action = QAction("Maze Solver", self)
        maze_action.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        toolbar.addAction(maze_action)

        box_order_action = QAction("Box Order", self)
        box_order_action.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        toolbar.addAction(box_order_action)

        object_recognition_action = QAction("Object Recognition", self)
        object_recognition_action.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        toolbar.addAction(object_recognition_action)

    def update_path_stats_label(self):
        path_stats = self.maze_widget.path_stats
        text = f"""Solver Information: 
        \tNumber of visited cells: {path_stats["N_Visited"]}
        \tTime Spent: {path_stats["Time_Spent"]}
        \tRoute Cost: {path_stats["Route_Cost"]} steps"""

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