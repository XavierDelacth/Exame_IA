"""
Módulo de Simulação - Sistema de Coordenação e Execução
Responsáveis: Henrique (Coordinator) + Adriana & Henrique (Simulator)

Este módulo contém as classes responsáveis por:
- Coordinator: Orquestração de múltiplos agentes
- Simulator: Motor principal de simulação
"""

from .coordinator import Coordinator, AgentAction, AgentStatus
from .simulator import Simulator, SimulationResult, SimulationConfig

__all__ = [
    'Coordinator',
    'AgentAction',
    'AgentStatus',
    'Simulator',
    'SimulationResult',
    'SimulationConfig'
]