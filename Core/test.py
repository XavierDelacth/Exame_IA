#!/usr/bin/env python3

# Teste simples para validar a versão Python

from data_types import Cell, Agent, MLModelType, Point
from communication_hub import CommunicationHub
from approach_a import evaluateApproachA
from approach_b import evaluateApproachB
from approach_c import evaluateApproachC
from decision_tree_model import decisionTreePredictNextMove
from knn_model import knnPredictNextMove
from naive_bayes_model import naiveBayesPredictNextMove

def test_basic():
    # Criar uma grelha simples 2x2
    grid = [
        [Cell(0, 0, 'L', False), Cell(1, 0, 'T', False)],
        [Cell(0, 1, 'B', False), Cell(1, 1, 'F', False)]
    ]

    # Criar um agente
    agent = Agent(1, MLModelType.KNN, 0, 0, True, False, [], 'red')

    # Testar abordagens
    print("Abordagem A (tesouros):", evaluateApproachA(grid))
    print("Abordagem B (exploração + sobrevivência):", evaluateApproachB(grid, [agent]))
    print("Abordagem C (bandeira):", evaluateApproachC(grid))

    # Testar hub
    hub = CommunicationHub()
    hub.registerExploration(grid[0][0])
    print("Célula 0,0 explorada:", hub.isExplored(0, 0))

    # Testar modelos
    current = Point(0, 0)
    possible = [Point(1, 0), Point(0, 1)]
    move = decisionTreePredictNextMove(current, possible, grid, hub)
    print("Decision Tree move:", move)

    move2 = knnPredictNextMove(current, possible, grid, hub)
    print("KNN move:", move2)

    move3 = naiveBayesPredictNextMove(current, possible, grid, hub)
    print("Naive Bayes move:", move3)

    print("All tests passed!")

if __name__ == "__main__":
    test_basic()