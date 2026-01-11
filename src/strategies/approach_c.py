"""
Abordagem C: Encontrar a Bandeira
Responsável: Xavier Delacth
Data: 12/01/2025

Critério de Sucesso:
- Pelo menos 1 agente deve ENCONTRAR a bandeira escondida

Estratégia:
- Busca eficiente por toda a área
- Comunicação sobre áreas já pesquisadas
- Priorizar áreas não exploradas
- Estratégia de busca em espiral ou sistemática
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.grid import Grid
from environment.cell import Cell, CellType
from agents.base_agent import BaseAgent


class SearchPattern(Enum):
    """Padrões de busca disponíveis"""
    SPIRAL = "spiral"           # Busca em espiral
    SYSTEMATIC = "systematic"   # Busca sistemática linha por linha
    RANDOM_WALK = "random_walk" # Caminhada aleatória
    GRID_SWEEP = "grid_sweep"   # Varredura em grade
    INFORMED = "informed"       # Busca informada (áreas menos exploradas)


@dataclass
class SearchZone:
    """
    Zona de busca atribuída temporariamente
    
    Attributes:
        center: Centro da zona (x, y)
        radius: Raio de busca
        searched_cells: Células já pesquisadas
        priority: Prioridade da zona (0-1)
    """
    center: Tuple[int, int]
    radius: int
    searched_cells: Set[Tuple[int, int]]
    priority: float = 0.5
    
    def is_complete(self, grid: Grid) -> bool:
        """Verifica se a zona foi completamente pesquisada"""
        cx, cy = self.center
        total_cells_in_radius = 0
        
        for x in range(max(0, cx - self.radius), 
                       min(grid.size, cx + self.radius + 1)):
            for y in range(max(0, cy - self.radius), 
                          min(grid.size, cy + self.radius + 1)):
                if abs(x - cx) + abs(y - cy) <= self.radius:
                    total_cells_in_radius += 1
        
        return len(self.searched_cells) >= total_cells_in_radius


class ApproachC:
    """
    Implementação da Abordagem C: Encontrar a Bandeira
    
    Esta abordagem coordena múltiplos agentes na busca pela bandeira,
    usando diferentes padrões de busca e comunicação eficiente.
    
    Attributes:
        grid: Grid do ambiente
        agents: Lista de agentes
        flag_position: Posição da bandeira (quando encontrada)
        search_pattern: Padrão de busca utilizado
        searched_areas: Áreas já pesquisadas
        search_zones: Zonas de busca por agente
    """
    
    def __init__(
        self, 
        grid: Grid, 
        agents: List[BaseAgent],
        search_pattern: SearchPattern = SearchPattern.INFORMED
    ):
        """
        Inicializa a Abordagem C
        
        Args:
            grid: Grid do ambiente
            agents: Lista de agentes participantes
            search_pattern: Padrão de busca a utilizar
        """
        self.grid = grid
        self.agents = agents
        self.search_pattern = search_pattern
        self.flag_position: Optional[Tuple[int, int]] = None
        self.flag_found_by: Optional[int] = None
        self.searched_areas: Set[Tuple[int, int]] = set()
        self.search_zones: Dict[int, SearchZone] = {}
        self.step_count = 0
        
        # Atribuir zonas iniciais de busca
        self._assign_initial_search_zones()
    
    def _assign_initial_search_zones(self):
        """
        Atribui zonas iniciais de busca aos agentes
        """
        num_agents = len(self.agents)
        if num_agents == 0:
            return
        
        grid_size = self.grid.size
        
        # Distribuir agentes em diferentes áreas do grid
        if num_agents == 1:
            center = (grid_size // 2, grid_size // 2)
            self.search_zones[self.agents[0].id] = SearchZone(
                center=center,
                radius=grid_size,
                searched_cells=set(),
                priority=1.0
            )
        
        elif num_agents == 2:
            # Dividir em duas metades
            self.search_zones[self.agents[0].id] = SearchZone(
                center=(grid_size // 4, grid_size // 2),
                radius=grid_size // 2,
                searched_cells=set(),
                priority=1.0
            )
            self.search_zones[self.agents[1].id] = SearchZone(
                center=(3 * grid_size // 4, grid_size // 2),
                radius=grid_size // 2,
                searched_cells=set(),
                priority=1.0
            )
        
        elif num_agents == 4:
            # Dividir em quadrantes
            positions = [
                (grid_size // 4, grid_size // 4),
                (grid_size // 4, 3 * grid_size // 4),
                (3 * grid_size // 4, grid_size // 4),
                (3 * grid_size // 4, 3 * grid_size // 4)
            ]
            
            for i, agent in enumerate(self.agents[:4]):
                self.search_zones[agent.id] = SearchZone(
                    center=positions[i],
                    radius=grid_size // 2,
                    searched_cells=set(),
                    priority=1.0
                )
        
        else:
            # Distribuir uniformemente
            for i, agent in enumerate(self.agents):
                angle = (2 * np.pi * i) / num_agents
                cx = int(grid_size // 2 + (grid_size // 3) * np.cos(angle))
                cy = int(grid_size // 2 + (grid_size // 3) * np.sin(angle))
                
                self.search_zones[agent.id] = SearchZone(
                    center=(cx, cy),
                    radius=grid_size // 3,
                    searched_cells=set(),
                    priority=1.0
                )
    
    def register_flag_found(self, agent_id: int, position: Tuple[int, int]):
        """
        Registra que a bandeira foi encontrada
        
        Args:
            agent_id: ID do agente que encontrou
            position: Posição da bandeira
        """
        self.flag_position = position
        self.flag_found_by = agent_id
    
    def is_flag_found(self) -> bool:
        """Verifica se a bandeira já foi encontrada"""
        return self.flag_position is not None
    
    def get_unexplored_cells_prioritized(self) -> List[Tuple[Cell, float]]:
        """
        Retorna células não exploradas com prioridade
        
        Returns:
            Lista de tuplas (célula, prioridade)
        """
        unexplored = self.grid.get_unexplored_cells()
        
        if not unexplored:
            return []
        
        prioritized = []
        
        for cell in unexplored:
            pos = (cell.x, cell.y)
            
            # Calcular prioridade baseada em:
            # 1. Distância até áreas já exploradas
            # 2. Centralidade no grid
            # 3. Densidade de exploração ao redor
            
            neighbors = self.grid.get_neighbors(cell.x, cell.y, include_diagonals=True)
            explored_neighbors = sum(1 for n in neighbors if n.explored)
            
            # Prioridade maior para células com vizinhos explorados
            # (bordas da área explorada)
            priority = explored_neighbors / len(neighbors) if neighbors else 0.5
            
            # Bônus para centralidade
            center_x, center_y = self.grid.size // 2, self.grid.size // 2
            distance_to_center = abs(cell.x - center_x) + abs(cell.y - center_y)
            centrality_bonus = 1.0 - (distance_to_center / (self.grid.size * 2))
            
            priority = (priority * 0.7) + (centrality_bonus * 0.3)
            
            prioritized.append((cell, priority))
        
        # Ordenar por prioridade (maior primeiro)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        return prioritized
    
    def get_next_target_spiral(
        self, 
        agent: BaseAgent, 
        zone: SearchZone
    ) -> Optional[Tuple[int, int]]:
        """
        Próximo alvo usando busca em espiral
        
        Args:
            agent: Agente
            zone: Zona de busca
            
        Returns:
            Tupla (x, y) ou None
        """
        cx, cy = zone.center
        
        # Busca em espiral crescente
        for radius in range(1, zone.radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) != radius:
                        continue
                    
                    x, y = cx + dx, cy + dy
                    
                    if not self.grid.is_valid_position(x, y):
                        continue
                    
                    cell = self.grid.get_cell(x, y)
                    if cell and not cell.explored:
                        return (x, y)
        
        return None
    
    def get_next_target_systematic(
        self, 
        agent: BaseAgent
    ) -> Optional[Tuple[int, int]]:
        """
        Próximo alvo usando busca sistemática (linha por linha)
        
        Args:
            agent: Agente
            
        Returns:
            Tupla (x, y) ou None
        """
        # Buscar linha por linha
        for x in range(self.grid.size):
            for y in range(self.grid.size):
                cell = self.grid.get_cell(x, y)
                if cell and not cell.explored:
                    return (x, y)
        
        return None
    
    def get_next_target_informed(
        self, 
        agent: BaseAgent
    ) -> Optional[Tuple[int, int]]:
        """
        Próximo alvo usando busca informada (áreas menos exploradas)
        
        Args:
            agent: Agente
            
        Returns:
            Tupla (x, y) ou None
        """
        prioritized = self.get_unexplored_cells_prioritized()
        
        if not prioritized:
            return None
        
        # Escolher célula de maior prioridade mais próxima
        for cell, priority in prioritized[:10]:  # Top 10
            # Verificar se é acessível
            distance = abs(cell.x - agent.x) + abs(cell.y - agent.y)
            if distance < self.grid.size * 2:  # Razoavelmente próxima
                return (cell.x, cell.y)
        
        # Se nenhuma próxima, pegar a de maior prioridade
        if prioritized:
            return (prioritized[0][0].x, prioritized[0][0].y)
        
        return None
    
    def suggest_next_move(self, agent: BaseAgent) -> Optional[Tuple[int, int]]:
        """
        Sugere próximo movimento baseado na Abordagem C
        
        Args:
            agent: Agente que precisa decidir
            
        Returns:
            Tupla (x, y) com posição sugerida ou None
        """
        # Se bandeira já foi encontrada, não precisa continuar
        if self.is_flag_found():
            return None
        
        # Obter zona do agente
        zone = self.search_zones.get(agent.id)
        
        # Escolher estratégia baseada no padrão
        if self.search_pattern == SearchPattern.SPIRAL and zone:
            target = self.get_next_target_spiral(agent, zone)
        elif self.search_pattern == SearchPattern.SYSTEMATIC:
            target = self.get_next_target_systematic(agent)
        elif self.search_pattern == SearchPattern.INFORMED:
            target = self.get_next_target_informed(agent)
        else:
            # Default: informed
            target = self.get_next_target_informed(agent)
        
        return target
    
    def update_search_progress(self, agent_id: int, explored_positions: Set[Tuple[int, int]]):
        """
        Atualiza progresso de busca
        
        Args:
            agent_id: ID do agente
            explored_positions: Posições exploradas
        """
        self.searched_areas.update(explored_positions)
        
        zone = self.search_zones.get(agent_id)
        if zone:
            zone.searched_cells.update(explored_positions)
    
    def check_success(self) -> bool:
        """
        Verifica se a Abordagem C foi bem-sucedida
        
        Returns:
            True se a bandeira foi encontrada
        """
        return self.is_flag_found()
    
    def get_statistics(self) -> Dict:
        """
        Retorna estatísticas da abordagem
        
        Returns:
            Dicionário com estatísticas
        """
        total_agents = len(self.agents)
        alive_agents = sum(1 for agent in self.agents if agent.alive)
        
        return {
            'approach': 'C',
            'objective': 'Encontrar a Bandeira',
            'success': self.check_success(),
            'flag_found': self.is_flag_found(),
            'flag_position': self.flag_position,
            'found_by_agent': self.flag_found_by,
            'total_agents': total_agents,
            'alive_agents': alive_agents,
            'exploration_percentage': self.grid.get_exploration_percentage(),
            'searched_cells': len(self.searched_areas),
            'total_cells': self.grid.size * self.grid.size,
            'search_pattern': self.search_pattern.value,
            'steps_taken': self.step_count
        }
    
    def print_progress(self):
        """Imprime progresso da busca"""
        stats = self.get_statistics()
        
        print("=" * 60)
        print("ABORDAGEM C - ENCONTRAR A BANDEIRA")
        print("=" * 60)
        print(f"Padrão de Busca: {stats['search_pattern'].upper()}")
        print(f"Exploração: {stats['exploration_percentage']:.1%}")
        print(f"Células Pesquisadas: {stats['searched_cells']}/{stats['total_cells']}")
        print(f"Agentes Vivos: {stats['alive_agents']}/{stats['total_agents']}")
        print(f"Steps: {stats['steps_taken']}")
        print()