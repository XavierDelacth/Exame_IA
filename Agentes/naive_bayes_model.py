from typing import List
from data_types import Point, Cell
from communication_hub import CommunicationHub

def naiveBayesPredictNextMove(
    current: Point,
    possibleMoves: List[Point],
    grid: List[List[Cell]],
    hub: CommunicationHub
) -> Point:
    """
    Heurística do Modelo Naive Bayes:
    Calcula a probabilidade de 'Seguro' de cada movimento baseado nas bombas conhecidas circundantes.
    P(Seguro | Vizinhos)
    """
    if not possibleMoves:
        return current

    def getDangerScore(p: Point) -> float:
        score = 0.0
        bombs = hub.getKnownBombs()
        for b in bombs:
            dist = abs(b.x - p.x) + abs(b.y - p.y)
            if dist == 1:
                score += 1.0  # Diretamente adjacente a uma bomba conhecida
            elif dist == 2:
                score += 0.3  # Próximo
        return score

    # Escolher o movimento com a pontuação de perigo mais baixa
    return min(possibleMoves, key=getDangerScore)