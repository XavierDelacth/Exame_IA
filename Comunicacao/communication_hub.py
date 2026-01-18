from typing import List, Optional, Set
from data_types import Point, Cell

class CommunicationHub:
    def __init__(self):
        self.exploredCells: Set[str] = set()
        self.knownBombs: Set[str] = set()
        self.flagLocation: Optional[Point] = None

    def registerExploration(self, cell: Cell):
        key = f"{cell.x},{cell.y}"
        self.exploredCells.add(key)
        
        if cell.type == 'B':
            self.knownBombs.add(key)
        elif cell.type == 'F':
            self.flagLocation = Point(x=cell.x, y=cell.y)

    def isExplored(self, x: int, y: int) -> bool:
        return f"{x},{y}" in self.exploredCells

    def isKnownBomb(self, x: int, y: int) -> bool:
        return f"{x},{y}" in self.knownBombs

    def getKnownBombs(self) -> List[Point]:
        return [Point(x=int(k.split(',')[0]), y=int(k.split(',')[1])) for k in self.knownBombs]

    def getFlagLocation(self) -> Optional[Point]:
        return self.flagLocation

    def reset(self):
        self.exploredCells.clear()
        self.knownBombs.clear()
        self.flagLocation = None

sharedHub = CommunicationHub()