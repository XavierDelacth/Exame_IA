import random
from typing import List, Optional
from data_types import Point, Cell
from communication_hub import CommunicationHub

def decisionTreePredictNextMove(
    current: Point,
    possibleMoves: List[Point],
    grid: List[List[Cell]],
    hub: CommunicationHub
) -> Point:
    """
    Heurística do Modelo de Árvore de Decisão:
    1. Se a posição da Bandeira for conhecida, mover-se na direção dela.
    2. Caso contrário, mover-se para um vizinho não explorado válido aleatório.
    """
    if not possibleMoves:
        return current

    flag = hub.getFlagLocation()
    if flag:
        return min(possibleMoves, key=lambda p: abs(p.x - flag.x) + abs(p.y - flag.y))

    return random.choice(possibleMoves)