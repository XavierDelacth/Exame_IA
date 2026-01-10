"""
Módulo de Agentes Inteligentes
Responsáveis: Adriana (base, perception, DT, KNN) + Henrique (NB)

Este módulo contém todas as classes de agentes:
- BaseAgent: Classe abstrata base
- Perception: Sistema de percepção de agentes
- DecisionTreeAgent: Agente com Decision Tree (Adriana)
- KNNAgent: Agente com KNN (Adriana)
- NaiveBayesAgent: Agente com Naive Bayes (Henrique)
"""

from .base_agent import BaseAgent
from .perception import PerceptionSystem
from .decision_tree_agent import DecisionTreeAgent
from .knn_agent import KNNAgent
from .naive_bayes_agent import NaiveBayesAgent

__all__ = [
    'BaseAgent',
    'PerceptionSystem',
    'DecisionTreeAgent',
    'KNNAgent',
    'NaiveBayesAgent'
]