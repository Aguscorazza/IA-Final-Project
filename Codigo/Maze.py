import numpy as np


class Maze:
    def __init__(self, rows, columns):
        self.nbRows = rows
        self.nbColumns = columns
        self.grid = np.zeros((rows, columns), dtype=int)
        self.startCell = None
        self.endCell = None

    def setStartCell(self, rowID, columnID):
        self.startCell = (rowID, columnID)
        self.grid[rowID][columnID] = 2  # Start point

    def setEndCell(self, rowID, columnID):
        self.endCell = (rowID, columnID)
        self.grid[rowID][columnID] = 3  # End point

    def setWallCell(self, rowID, columnID):
        self.grid[rowID][columnID] = 1  # Wall

    def isWithinBounds(self, rowID, columnID):
        return 0 <= rowID < self.nbRows and 0 <= columnID < self.nbColumns

    def isWalkable(self, rowID, columnID):
        return self.grid[rowID][columnID] == 0 or self.grid[rowID][columnID] == 3

    def __repr__(self):
        return str(self.grid)