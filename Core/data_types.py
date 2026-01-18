from enum import Enum
from typing import List, Optional
from dataclasses import dataclass

CellType = str  # 'L' | 'B' | 'F' | 'T' | 'U'  # Livre, Bomba, Bandeira, Tesouro, Desconhecido

@dataclass
class Cell:
    x: int
    y: int
    type: CellType
    isExplored: bool

class MLModelType(Enum):
    KNN = 'KNN'
    DECISION_TREE = 'Árvore de Decisão'
    NAIVE_BAYES = 'Naive Bayes'

class Approach(Enum):
    A = 'A'
    B = 'B'
    C = 'C'

@dataclass
class Agent:
    id: int
    model: MLModelType
    x: int
    y: int
    isAlive: bool
    hasShield: bool  # Capacidade de desativar bomba após recolher tesouro
    path: List[dict]  # Lista de {'x': int, 'y': int}
    color: str

@dataclass
class LogEntry:
    timestamp: float
    message: str
    type: str  # 'info' | 'success' | 'error' | 'warning'

@dataclass
class SimulationStats:
    cellsExplored: int
    agentsAlive: int
    totalAgents: int
    executionTime: float
    success: bool
    approach: Approach
    treasuresFound: Optional[int] = None

@dataclass
class ModelRanking:
    model: MLModelType
    successRate: float
    avgExplorationTime: float

@dataclass
class Point:
    x: int
    y: int