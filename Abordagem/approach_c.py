from typing import List
from data_types import Cell

def evaluateApproachC(grid: List[List[Cell]]) -> bool:
    """
    Abordagem C:
    Sucesso se pelo menos um agente encontrar a bandeira.
    """
    return any(cell.type == 'F' and cell.isExplored for row in grid for cell in row)