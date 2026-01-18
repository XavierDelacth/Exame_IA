from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field

CellType = str  # 'L' | 'B' | 'F' | 'T' | 'U'  # Livre, Bomba, Bandeira, Tesouro, Desconhecido

@dataclass
class Cell:
    x: int
    y: int
    type: CellType
    isExplored: bool
    collected: bool = False
    neutralized: bool = False

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
    shield_count: int = 0
    path: List[dict] = field(default_factory=list)
    color: str = ""

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