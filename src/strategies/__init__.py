"""
Módulo de Estratégias de Exploração
Responsáveis: 
- Adriana: Approach A (>50% tesouros)
- Xavier Delacth: Approach B (exploração completa) e Approach C (encontrar bandeira)

Data: 12/01/2025

Este módulo contém as três abordagens de exploração:
- ApproachA: Descobrir mais de 50% dos tesouros
- ApproachB: Exploração completa com pelo menos 1 sobrevivente
- ApproachC: Encontrar a bandeira escondida
"""

from .approach_a import ApproachA
from .approach_b import ApproachB
from .approach_c import ApproachC

__all__ = [
    'ApproachA',
    'ApproachB',
    'ApproachC'
]