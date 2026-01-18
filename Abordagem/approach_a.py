from typing import List
from data_types import Cell

def evaluateApproachA(grid: List[List[Cell]]) -> bool:
    """
    Abordagem A:
    Considerar sucesso se o número de tesouros descobertos for superior a 50% 
    do total de tesouros inicialmente distribuídos.
    """
    flat = [cell for row in grid for cell in row]
    totalTreasures = len([c for c in flat if c.type == 'T'])
    foundTreasures = len([c for c in flat if c.type == 'T' and c.isExplored])
    
    if totalTreasures == 0:
        return False
    return foundTreasures > (totalTreasures / 2)