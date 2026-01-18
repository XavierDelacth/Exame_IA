import math
from typing import List
from data_types import Point, Cell
from communication_hub import CommunicationHub

def knnPredictNextMove(
    current: Point,
    possibleMoves: List[Point],
    grid: List[List[Cell]],
    hub: CommunicationHub
) -> Point:
    """
    Heurística do Modelo KNN:
    Seleciona o movimento que está mais próximo do "centroide" do território não explorado,
    efetivamente agrupando o movimento em direção a áreas desconhecidas.
    """
    if not possibleMoves:
        return current

    # Encontrar todos os pontos não explorados
    unexplored: List[Point] = []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if not hub.isExplored(x, y):
                unexplored.append(Point(x=x, y=y))

    if not unexplored:
        return possibleMoves[0]

    # Mover-se na direção da posição média das células não exploradas
    avgX = sum(p.x for p in unexplored) / len(unexplored)
    avgY = sum(p.y for p in unexplored) / len(unexplored)

    return min(possibleMoves, key=lambda p: math.sqrt((p.x - avgX)**2 + (p.y - avgY)**2))