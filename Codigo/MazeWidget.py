from PyQt5.QtGui import QPainter, QFont, QColor
from PyQt5.QtWidgets import QWidget


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
                    painter.setBrush(QColor(100, 100, 100))  # Gray for closed set nodes
                elif (rowID, columnID) in self.maze.openSetNodes:
                    painter.setBrush(QColor(0, 0, 200))  # Dark blue for open set nodes
                else:
                    painter.setBrush(QColor(255, 255, 255))
                painter.drawRect(columnID * self.cell_size, rowID * self.cell_size, self.cell_size, self.cell_size)

        painter.setBrush(QColor(255, 255, 0))
        for rowID, columnID in self.path:
            if self.maze.grid[rowID][columnID] == 0:
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