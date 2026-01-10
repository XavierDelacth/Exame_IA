"""
Módulo de Grid - Ambiente de Simulação
Responsável: Adriana
Data: 7/01/2026
"""

import numpy as np
import random
from typing import Tuple, List, Optional, Set
from .cell import Cell, CellType
from .config import EnvironmentConfig


class Grid:
    """
    Classe que representa o ambiente de exploração (matriz NxN)
    
    Attributes:
        size (int): Tamanho da matriz
        config (EnvironmentConfig): Configurações do ambiente
        cells (np.ndarray): Matriz de células
        flag_position (Optional[Tuple[int, int]]): Posição da bandeira
    """
    
    def __init__(self, config: EnvironmentConfig = None):
        """
        Inicializa o grid
        
        Args:
            config: Configurações do ambiente 
        """
        self.config = config or EnvironmentConfig()
        self.size = self.config.grid_size
        self.cells: np.ndarray = None
        self.flag_position: Optional[Tuple[int, int]] = None
        self._initialize_grid()
        
    def _initialize_grid(self):
        """Inicializa a matriz de células vazias"""
        self.cells = np.empty((self.size, self.size), dtype=object)
        for i in range(self.size):
            for j in range(self.size):
                self.cells[i, j] = Cell(i, j, CellType.EMPTY)
    
    def setup_random(self, seed: Optional[int] = None):
        """
        Configura o grid aleatoriamente com bombas, tesouros e bandeira
        
        Args:
            seed: Seed para reproducibilidade 
        """
        seed = seed or self.config.seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Resetar grid
        self._initialize_grid()
        
        total_cells = self.size * self.size
        available_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        random.shuffle(available_positions)
        
        position_idx = 0
        
        # Colocar bombas
        num_bombs = self.config.get_total_bombs()
        for _ in range(num_bombs):
            if position_idx >= total_cells:
                break
            x, y = available_positions[position_idx]
            self.cells[x, y].type = CellType.BOMB
            position_idx += 1
        
        # Colocar tesouros
        for _ in range(self.config.treasures_count):
            if position_idx >= total_cells:
                break
            x, y = available_positions[position_idx]
            self.cells[x, y].type = CellType.TREASURE
            position_idx += 1
        
        # Colocar bandeira (se habilitado)
        if self.config.enable_flag:
            if position_idx >= total_cells:
                raise ValueError("Não há espaço suficiente para a bandeira")
            x, y = available_positions[position_idx]
            self.cells[x, y].type = CellType.FLAG
            self.flag_position = (x, y)
            position_idx += 1
        
        # Resto são células livres
        for i in range(position_idx, total_cells):
            x, y = available_positions[i]
            self.cells[x, y].type = CellType.FREE
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """
        Verifica se a posição está dentro do grid
        
        Args:
            x, y: Coordenadas
            
        Returns:
            True se a posição for válida
        """
        return 0 <= x < self.size and 0 <= y < self.size
    
    def get_cell(self, x: int, y: int) -> Optional[Cell]:
        """
        Retorna a célula na posição especificada
        
        Args:
            x, y: Coordenadas
            
        Returns:
            Objeto Cell ou None se posição inválida
        """
        if not self.is_valid_position(x, y):
            return None
        return self.cells[x, y]
    
    def get_neighbors(self, x: int, y: int, include_diagonals: bool = False) -> List[Cell]:
        """
        Retorna células vizinhas
        
        Args:
            x, y: Coordenadas centrais
            include_diagonals: Se deve incluir diagonais
            
        Returns:
            Lista de células vizinhas
        """
        neighbors = []
        
        # Direções: Norte, Sul, Leste, Oeste
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        
        # Adicionar diagonais se solicitado
        if include_diagonals:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            cell = self.get_cell(nx, ny)
            if cell is not None:
                neighbors.append(cell)
        
        return neighbors
    
    def get_cells_in_radius(self, x: int, y: int, radius: int) -> List[Cell]:
        """
        Retorna todas as células dentro de um raio (distância Manhattan)
        
        Args:
            x, y: Coordenadas centrais
            radius: Raio de busca
            
        Returns:
            Lista de células no raio
        """
        cells = []
        for i in range(max(0, x - radius), min(self.size, x + radius + 1)):
            for j in range(max(0, y - radius), min(self.size, y + radius + 1)):
                if abs(i - x) + abs(j - y) <= radius:
                    cells.append(self.cells[i, j])
        return cells
    
    def get_explored_cells(self) -> List[Cell]:
        """
        Retorna todas as células exploradas
        
        Returns:
            Lista de células exploradas
        """
        return [cell for row in self.cells for cell in row if cell.explored]
    
    def get_unexplored_cells(self) -> List[Cell]:
        """
        Retorna todas as células não exploradas
        
        Returns:
            Lista de células não exploradas
        """
        return [cell for row in self.cells for cell in row if not cell.explored]
    
    def get_bombs(self) -> List[Cell]:
        """Retorna todas as células com bombas"""
        return [cell for row in self.cells for cell in row if cell.is_bomb()]
    
    def get_treasures(self) -> List[Cell]:
        """Retorna todas as células com tesouros (não coletados)"""
        return [cell for row in self.cells for cell in row if cell.is_treasure()]
    
    def get_free_cells(self) -> List[Cell]:
        """Retorna todas as células livres"""
        return [cell for row in self.cells for cell in row if cell.is_free()]
    
    def count_bombs(self) -> int:
        """Conta o número de bombas no grid"""
        return len(self.get_bombs())
    
    def count_treasures(self) -> int:
        """Conta o número de tesouros restantes no grid"""
        return len(self.get_treasures())
    
    def count_explored(self) -> int:
        """Conta o número de células exploradas"""
        return len(self.get_explored_cells())
    
    def get_exploration_percentage(self) -> float:
        """
        Calcula o percentual de exploração do grid
        
        Returns:
            Percentual de células exploradas (0.0 a 1.0)
        """
        total_cells = self.size * self.size
        explored = self.count_explored()
        return explored / total_cells if total_cells > 0 else 0.0
    
    def is_fully_explored(self) -> bool:
        """Verifica se o grid foi completamente explorado"""
        return self.get_exploration_percentage() >= 1.0
    
    def get_safe_starting_positions(self, count: int = 1) -> List[Tuple[int, int]]:
        """
        Encontra posições seguras para iniciar agentes
        
        Args:
            count: Número de posições desejadas
            
        Returns:
            Lista de tuplas (x, y) com posições seguras
        """
        free_cells = [cell for row in self.cells for cell in row if cell.is_free()]
        
        if len(free_cells) < count:
            # Se não houver células livres suficientes, usar células não-bomba
            non_bomb_cells = [cell for row in self.cells for cell in row if not cell.is_bomb()]
            free_cells = non_bomb_cells
        
        random.shuffle(free_cells)
        return [(cell.x, cell.y) for cell in free_cells[:count]]
    
    def mark_cell_explored(self, x: int, y: int, agent_id: int, step: int):
        """
        Marca uma célula como explorada
        
        Args:
            x, y: Coordenadas
            agent_id: ID do agente
            step: Step atual da simulação
        """
        cell = self.get_cell(x, y)
        if cell:
            cell.mark_explored(agent_id, step)
    
    def get_cell_type(self, x: int, y: int) -> Optional[CellType]:
        """
        Retorna o tipo da célula
        
        Args:
            x, y: Coordenadas
            
        Returns:
            CellType ou None se posição inválida
        """
        cell = self.get_cell(x, y)
        return cell.type if cell else None
    
    def to_matrix(self, show_explored_only: bool = False) -> np.ndarray:
        """
        Converte o grid para matriz de strings (para visualização)
        
        Args:
            show_explored_only: Se True, mostra apenas células exploradas
            
        Returns:
            Matriz numpy de strings
        """
        matrix = np.empty((self.size, self.size), dtype=str)
        
        for i in range(self.size):
            for j in range(self.size):
                cell = self.cells[i, j]
                if show_explored_only and not cell.explored:
                    matrix[i, j] = '?'
                else:
                    matrix[i, j] = cell.type.value
        
        return matrix
    
    def print_grid(self, show_explored_only: bool = False):
        """
        Imprime o grid no console
        
        Args:
            show_explored_only: Se True, mostra apenas células exploradas
        """
        matrix = self.to_matrix(show_explored_only)
        
        print("  ", end="")
        for j in range(self.size):
            print(f" {j}", end="")
        print()
        
        for i in range(self.size):
            print(f"{i:2}", end=" ")
            for j in range(self.size):
                print(f" {matrix[i, j]}", end="")
            print()
    
    def to_dict(self) -> dict:
        """
        Converte o grid para dicionário (para serialização)
        
        Returns:
            Dicionário representando o grid
        """
        return {
            'config': self.config.to_dict(),
            'cells': [[cell.to_dict() for cell in row] for row in self.cells],
            'flag_position': self.flag_position
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Grid':
        """
        Cria um grid a partir de dicionário
        
        Args:
            data: Dicionário com dados do grid
            
        Returns:
            Nova instância de Grid
        """
        config = EnvironmentConfig.from_dict(data['config'])
        grid = cls(config)
        
        for i, row_data in enumerate(data['cells']):
            for j, cell_data in enumerate(row_data):
                grid.cells[i, j] = Cell.from_dict(cell_data)
        
        grid.flag_position = data.get('flag_position')
        return grid
    
    def reset(self):
        """Reseta o estado de exploração do grid (mantém configuração)"""
        for row in self.cells:
            for cell in row:
                cell.reset()
    
    def clone(self) -> 'Grid':
        """
        Cria uma cópia profunda do grid
        
        Returns:
            Nova instância de Grid com os mesmos dados
        """
        return Grid.from_dict(self.to_dict())
    
    def __repr__(self) -> str:
        """Representação em string"""
        return f"Grid({self.size}x{self.size}, explored={self.get_exploration_percentage():.1%})"
    
    def __str__(self) -> str:
        """String representation"""
        return self.__repr__()