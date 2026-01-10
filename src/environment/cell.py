"""
Módulo de Cell - Tipos de Células
Responsável: Adriana
Data: 7/01/2026
"""

from enum import Enum
from typing import Optional, List


class CellType(Enum):
    """
    Tipos de células no grid de exploração
    """
    EMPTY = 'E'     # Vazia (não explorada)
    FREE = 'L'      # Livre (explorada e segura)
    BOMB = 'B'      # Bomba (perigosa)
    TREASURE = 'T'  # Tesouro (benéfica)
    FLAG = 'F'      # Bandeira (objetivo da Abordagem C)
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> 'CellType':
        """
        Converte string para CellType
        
        Args:
            value: String representando o tipo ('E', 'L', 'B', 'T', 'F')
            
        Returns:
            CellType correspondente
            
        Raises:
            ValueError: Se o valor não for válido
        """
        for cell_type in cls:
            if cell_type.value == value:
                return cell_type
        raise ValueError(f"Tipo de célula inválido: {value}")


class Cell:
    """
    Classe que representa uma célula individual do grid
    
    Attributes:
        x (int): Coordenada X (linha)
        y (int): Coordenada Y (coluna)
        type (CellType): Tipo da célula
        explored (bool): Se a célula foi explorada
        visited_by (List[int]): IDs dos agentes que visitaram
        discovery_step (int): Step em que foi descoberta
    """
    
    def __init__(self, x: int, y: int, cell_type: CellType = CellType.EMPTY):
        """
        Inicializa uma célula
        
        Args:
            x: Coordenada X (linha)
            y: Coordenada Y (coluna)
            cell_type: Tipo inicial da célula
        """
        self.x = x
        self.y = y
        self.type = cell_type
        self.explored = False
        self.visited_by: List[int] = []
        self.discovery_step: Optional[int] = None
        self._original_type = cell_type  # Guardar tipo original
        
    def mark_explored(self, agent_id: int, step: int):
        """
        Marca a célula como explorada por um agente
        
        Args:
            agent_id: ID do agente que explorou
            step: Step da simulação em que foi explorada
        """
        if not self.explored:
            self.explored = True
            self.discovery_step = step
        
        if agent_id not in self.visited_by:
            self.visited_by.append(agent_id)
    
    def is_safe(self) -> bool:
        """
        Verifica se a célula é segura (não é bomba)
        
        Returns:
            True se a célula não for bomba
        """
        return self.type != CellType.BOMB
    
    def is_bomb(self) -> bool:
        """Verifica se a célula é uma bomba"""
        return self.type == CellType.BOMB
    
    def is_treasure(self) -> bool:
        """Verifica se a célula contém tesouro"""
        return self.type == CellType.TREASURE
    
    def is_flag(self) -> bool:
        """Verifica se a célula contém a bandeira"""
        return self.type == CellType.FLAG
    
    def is_free(self) -> bool:
        """Verifica se a célula é livre"""
        return self.type == CellType.FREE
    
    def is_empty(self) -> bool:
        """Verifica se a célula ainda não foi definida"""
        return self.type == CellType.EMPTY
    
    def consume_treasure(self):
        """
        Remove o tesouro da célula (após ser coletado)
        Transforma a célula em FREE
        """
        if self.type == CellType.TREASURE:
            self.type = CellType.FREE
    
    def deactivate_bomb(self):
        """
        Desativa uma bomba (quando agente tem poder de tesouro)
        Transforma a célula em FREE
        """
        if self.type == CellType.BOMB:
            self.type = CellType.FREE
    
    def get_position(self) -> tuple:
        """
        Retorna a posição da célula
        
        Returns:
            Tupla (x, y) com as coordenadas
        """
        return (self.x, self.y)
    
    def distance_to(self, other: 'Cell') -> float:
        """
        Calcula distância Manhattan até outra célula
        
        Args:
            other: Outra célula
            
        Returns:
            Distância Manhattan (int)
        """
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def euclidean_distance_to(self, other: 'Cell') -> float:
        """
        Calcula distância Euclidiana até outra célula
        
        Args:
            other: Outra célula
            
        Returns:
            Distância Euclidiana (float)
        """
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    def to_dict(self) -> dict:
        """
        Converte a célula para dicionário (para serialização)
        
        Returns:
            Dicionário com os atributos da célula
        """
        return {
            'x': self.x,
            'y': self.y,
            'type': self.type.value,
            'explored': self.explored,
            'visited_by': self.visited_by.copy(),
            'discovery_step': self.discovery_step
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Cell':
        """
        Cria uma célula a partir de um dicionário
        
        Args:
            data: Dicionário com os atributos
            
        Returns:
            Nova instância de Cell
        """
        cell_type = CellType.from_string(data['type'])
        cell = cls(data['x'], data['y'], cell_type)
        cell.explored = data['explored']
        cell.visited_by = data['visited_by'].copy()
        cell.discovery_step = data.get('discovery_step')
        return cell
    
    def reset(self):
        """
        Reseta o estado da célula (mantém tipo, remove exploração)
        """
        self.explored = False
        self.visited_by.clear()
        self.discovery_step = None
        self.type = self._original_type
    
    def __repr__(self) -> str:
        """Representação em string para debug"""
        explored_str = "✓" if self.explored else "✗"
        return f"Cell({self.x},{self.y})[{self.type.value}]{explored_str}"
    
    def __str__(self) -> str:
        """Representação em string simples"""
        return self.type.value
    
    def __eq__(self, other) -> bool:
        """Comparação de igualdade baseada na posição"""
        if not isinstance(other, Cell):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self) -> int:
        """Hash baseado na posição (permite usar Cell em sets/dicts)"""
        return hash((self.x, self.y))