#!/usr/bin/env python3
"""
Script de teste para verificar a dispersão inicial
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Teste simples da dispersão
from simulation import Simulation
import numpy as np

def test_dispersion():
    print("=== Teste de Dispersão Inicial ===")

    # Criar simulação com seed para reprodutibilidade
    np.random.seed(42)
    sim = Simulation('A', 3, 0.3, 'homogeneous')

    print(f"Ambiente criado: {sim.environment.size}x{sim.environment.size}")
    print(f"Posições iniciais dos agentes: {[agent.position for agent in sim.agents]}")

    # Executar alguns passos para ver movimento
    for i in range(3):
        for agent in sim.agents:
            if agent.alive:
                move = agent.decide_next_move(sim.environment)
                if move:
                    agent.move(sim.environment, move)

    print(f"Posições após alguns movimentos: {[agent.position for agent in sim.agents]}")
    print(f"Células exploradas: {np.sum(sim.environment.explored)}")

    # Verificar se nenhum agente ficou em (0,0)
    agents_in_start = sum(1 for agent in sim.agents if agent.position == (0, 0))
    print(f"Agentes ainda em (0,0): {agents_in_start}")

    # Verificar se posições são diferentes
    positions = [agent.position for agent in sim.agents]
    unique_positions = len(set(positions))
    print(f"Posições únicas: {unique_positions}/{len(positions)}")

    print("✅ Teste concluído!")

if __name__ == "__main__":
    test_dispersion()