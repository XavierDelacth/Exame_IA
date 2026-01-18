from typing import List
from data_types import Cell, Agent

def evaluateApproachB(grid: List[List[Cell]], agents: List[Agent]) -> bool:
    """
    Abordagem B:
    Sucesso se o ambiente estiver completamente explorado e pelo menos um agente sobreviver.
    """
    allExplored = all(cell.isExplored for row in grid for cell in row)
    survivors = any(agent.isAlive for agent in agents)
    
    return allExplored and survivors