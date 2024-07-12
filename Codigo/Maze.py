import numpy as np
import random


class Maze:
    def __init__(self, rows, columns):
        self.nbRows = rows
        self.nbColumns = columns
        self.grid = np.zeros((rows, columns), dtype=int)
        self.startCell = None
        self.endCell = None
        self.openSetNodes = set()
        self.closedSetNodes = set()

    def setStartCell(self, rowID, columnID):
        self.startCell = (rowID, columnID)
        self.grid[rowID][columnID] = 2  # Start point

    def setEndCell(self, rowID, columnID):
        self.endCell = (rowID, columnID)
        self.grid[rowID][columnID] = 3  # End point

    def setWallCell(self, rowID, columnID):
        self.grid[rowID][columnID] = 1  # Wall

    def addOpenSetNode(self, rowID, columnID):
        self.openSetNodes.add((rowID, columnID))

    def addClosedSetNode(self, rowID, columnID):
        self.closedSetNodes.add((rowID, columnID))

    def clearSetNodes(self):
        self.openSetNodes.clear()
        self.closedSetNodes.clear()

    def isWithinBounds(self, rowID, columnID):
        return 0 <= rowID < self.nbRows and 0 <= columnID < self.nbColumns

    def isWalkable(self, rowID, columnID):
        return self.grid[rowID][columnID] == 0 or self.grid[rowID][columnID] == 3

    def __repr__(self):
        return str(self.grid)

    def clearMaze(self):
        self.startCell = None
        self.endCell = None
        self.clearSetNodes()
        # Initially all cells are considered as walls
        self.grid = np.zeros((self.nbRows, self.nbColumns), dtype=int)

    def generateMaze(self):
        in_cells = set()
        frontier_cells = set()

        self.clearMaze()

        # Initially all cells are considered as walls
        self.grid = np.ones((self.nbRows, self.nbColumns), dtype=int)

        start_row, start_col = random.randrange(0, self.nbRows), random.randrange(0, self.nbColumns)
        self.grid[start_row][start_col] = 0
        in_cells.add((start_row, start_col))

        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

        for direction in directions:
            new_row, new_col = start_row + direction[0], start_col + direction[1]
            if self.isWithinBounds(new_row, new_col) and self.grid[new_row][new_col] == 1:
                frontier_cells.add((new_row, new_col))

        while len(frontier_cells) > 0:
            cf = random.choice(list(frontier_cells))
            cf_row, cf_col = cf
            random.shuffle(directions)
            found_in_cell = False
            for direction in directions:
                in_row, in_col = cf_row + direction[0], cf_col + direction[1]
                if (in_row, in_col) in in_cells and not found_in_cell:
                    found_in_cell = True
                    self.grid[cf_row][cf_col] = 0
                    self.grid[cf_row + direction[0] // 2][cf_col + direction[1] // 2] = 0
                    in_cells.add(cf)
                    frontier_cells.remove(cf)

                elif (in_row, in_col) not in in_cells:
                    if self.isWithinBounds(in_row, in_col) and self.grid[in_row][in_col] == 1:
                        frontier_cells.add((in_row, in_col))

