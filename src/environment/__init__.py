"""
Módulo de Ambiente - Sistema de Simulação
Responsável: Adriana

Este módulo contém todas as classes relacionadas ao ambiente de exploração:
- Grid: Matriz 10x10 do ambiente
- Cell: Representação de células individuais
- Config: Configurações do ambiente
"""

from .grid import Grid
from .cell import Cell, CellType
from .config import EnvironmentConfig

__all__ = ['Grid', 'Cell', 'CellType', 'EnvironmentConfig']