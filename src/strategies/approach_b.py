"""
Abordagem B: Exploração Completa + 1 Sobrevivente
Responsável: Xavier Delacth
Data: 12/01/2025

Critério de Sucesso:
- O ambiente deve ser COMPLETAMENTE explorado (100%)
- Pelo menos 1 agente deve permanecer VIVO ao final

Estratégia:
- Coordenação eficiente para cobrir todo o grid
- Divisão de áreas entre agentes
- Priorizar segurança dos agentes
- Comunicação constante sobre áreas exploradas
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.grid import Grid
from environment.cell import Cell, CellType
from agents.base_agent import BaseAgent


@dataclass
class ExplorationZone:
    """
    Zona de exploração atribuída a um agente
    
    Attributes:
        agent_id: ID do agente responsável
        start_x, start_y: Início da zona
        end_x, end_y: Fim da zona
        cells: Conjunto de células na zona
        explored_count: Células já exploradas
    """
    agent_id: int
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    cells: Set[Tuple[int, int]]
    explored_count: int = 0
    
    def is_complete(self) -> bool:
        """Verifica se a zona foi completamente explorada"""
        return self.explored_count >= len(self.cells)
    
    def get_progress(self) -> float:
        """Retorna progresso de exploração (0.0 a 1.0)"""
        if len(self.cells) == 0:
            return 1.0
        return self.explored_count / len(self.cells)


class ApproachB:
    """
    Implementação da Abordagem B: Exploração Completa
    
    Esta abordagem divide o grid em zonas e atribui cada zona
    a um agente específico, garantindo cobertura completa e
    minimizando redundância.
    
    Attributes:
        grid: Grid do ambiente
        agents: Lista de agentes
        zones: Zonas de exploração por agente
        priority_cells: Células de alta prioridade
        safe_paths: Caminhos seguros conhecidos
    """
    
    def __init__(self, grid: Grid, agents: List[BaseAgent]):
        """
        Inicializa a Abordagem B
        
        Args:
            grid: Grid do ambiente
            agents: Lista de agentes participantes
        """
        self.grid = grid
        self.agents = agents
        self.zones: Dict[int, ExplorationZone] = {}
        self.priority_cells: Set[Tuple[int, int]] = set()
        self.safe_paths: Set[Tuple[int, int]] = set()
        self.fully_explored_zones: Set[int] = set()
        
        # Dividir grid em zonas
        self._divide_grid_into_zones()
    
    def _divide_grid_into_zones(self):
        """
        Divide o grid em zonas balanceadas entre os agentes
        """
        num_agents = len(self.agents)
        if num_agents == 0:
            return
        
        grid_size = self.grid.size
        total_cells = grid_size * grid_size
        cells_per_agent = total_cells // num_agents
        
        # Calcular dimensões aproximadas de cada zona
        # Tentar criar zonas retangulares
        if num_agents == 1:
            # Um único agente explora tudo
            zone = ExplorationZone(
                agent_id=self.agents[0].id,
                start_x=0,
                start_y=0,
                end_x=grid_size - 1,
                end_y=grid_size - 1,
                cells={(x, y) for x in range(grid_size) for y in range(grid_size)}
            )
            self.zones[self.agents[0].id] = zone
        
        elif num_agents == 2:
            # Dividir verticalmente
            mid = grid_size // 2
            
            zone1 = ExplorationZone(
                agent_id=self.agents[0].id,
                start_x=0,
                start_y=0,
                end_x=grid_size - 1,
                end_y=mid - 1,
                cells={(x, y) for x in range(grid_size) for y in range(mid)}
            )
            
            zone2 = ExplorationZone(
                agent_id=self.agents[1].id,
                start_x=0,
                start_y=mid,
                end_x=grid_size - 1,
                end_y=grid_size - 1,
                cells={(x, y) for x in range(grid_size) for y in range(mid, grid_size)}
            )
            
            self.zones[self.agents[0].id] = zone1
            self.zones[self.agents[1].id] = zone2
        
        elif num_agents == 4:
            # Dividir em quadrantes
            mid_x = grid_size // 2
            mid_y = grid_size // 2
            
            quadrants = [
                (0, 0, mid_x - 1, mid_y - 1),  # Superior esquerdo
                (0, mid_y, mid_x - 1, grid_size - 1),  # Inferior esquerdo
                (mid_x, 0, grid_size - 1, mid_y - 1),  # Superior direito
                (mid_x, mid_y, grid_size - 1, grid_size - 1)  # Inferior direito
            ]
            
            for i, agent in enumerate(self.agents[:4]):
                start_x, start_y, end_x, end_y = quadrants[i]
                cells = {
                    (x, y) 
                    for x in range(start_x, end_x + 1) 
                    for y in range(start_y, end_y + 1)
                }
                
                zone = ExplorationZone(
                    agent_id=agent.id,
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    cells=cells
                )
                self.zones[agent.id] = zone
        
        else:
            # Para mais agentes, dividir em faixas horizontais
            rows_per_agent = grid_size // num_agents
            
            for i, agent in enumerate(self.agents):
                start_x = i * rows_per_agent
                end_x = start_x + rows_per_agent - 1 if i < num_agents - 1 else grid_size - 1
                
                cells = {
                    (x, y) 
                    for x in range(start_x, end_x + 1) 
                    for y in range(grid_size)
                }
                
                zone = ExplorationZone(
                    agent_id=agent.id,
                    start_x=start_x,
                    start_y=0,
                    end_x=end_x,
                    end_y=grid_size - 1,
                    cells=cells
                )
                self.zones[agent.id] = zone
    
    def get_agent_zone(self, agent_id: int) -> Optional[ExplorationZone]:
        """
        Retorna a zona atribuída a um agente
        
        Args:
            agent_id: ID do agente
            
        Returns:
            ExplorationZone ou None
        """
        return self.zones.get(agent_id)
    
    def update_zone_progress(self, agent_id: int, explored_cells: Set[Tuple[int, int]]):
        """
        Atualiza o progresso de exploração de uma zona
        
        Args:
            agent_id: ID do agente
            explored_cells: Células exploradas pelo agente
        """
        zone = self.zones.get(agent_id)
        if zone:
            zone.explored_count = len(zone.cells.intersection(explored_cells))
            
            if zone.is_complete() and agent_id not in self.fully_explored_zones:
                self.fully_explored_zones.add(agent_id)
    
    def get_next_target_in_zone(
        self, 
        agent: BaseAgent, 
        zone: ExplorationZone
    ) -> Optional[Tuple[int, int]]:
        """
        Encontra o próximo alvo na zona do agente
        
        Args:
            agent: Agente
            zone: Zona de exploração
            
        Returns:
            Tupla (x, y) ou None se zona completa
        """
        # Células não exploradas na zona
        unexplored = []
        
        for x, y in zone.cells:
            cell = self.grid.get_cell(x, y)
            if cell and not cell.explored:
                # Calcular distância até o agente
                distance = abs(x - agent.x) + abs(y - agent.y)
                unexplored.append(((x, y), distance))
        
        if not unexplored:
            return None
        
        # Ordenar por distância (mais próximo primeiro)
        unexplored.sort(key=lambda item: item[1])
        
        # Retornar célula mais próxima
        return unexplored[0][0]
    
    def get_unexplored_cell_outside_zone(
        self,
        agent: BaseAgent
    ) -> Optional[Tuple[int, int]]:
        """
        Encontra célula não explorada fora da zona (para ajudar outros agentes)
        
        Args:
            agent: Agente
            
        Returns:
            Tupla (x, y) ou None
        """
        unexplored_cells = self.grid.get_unexplored_cells()
        
        if not unexplored_cells:
            return None
        
        # Encontrar célula mais próxima
        closest_cell = None
        min_distance = float('inf')
        
        for cell in unexplored_cells:
            distance = abs(cell.x - agent.x) + abs(cell.y - agent.y)
            if distance < min_distance:
                min_distance = distance
                closest_cell = cell
        
        if closest_cell:
            return (closest_cell.x, closest_cell.y)
        
        return None
    
    def suggest_next_move(self, agent: BaseAgent) -> Optional[Tuple[int, int]]:
        """
        Sugere próximo movimento para um agente baseado na Abordagem B
        
        Args:
            agent: Agente que precisa decidir movimento
            
        Returns:
            Tupla (x, y) com posição sugerida ou None
        """
        # Obter zona do agente
        zone = self.get_agent_zone(agent.id)
        
        if zone and not zone.is_complete():
            # Explorar zona própria
            target = self.get_next_target_in_zone(agent, zone)
            if target:
                return target
        
        # Se zona completa ou sem zona, ajudar outros
        target = self.get_unexplored_cell_outside_zone(agent)
        return target
    
    def check_success(self) -> bool:
        """
        Verifica se a Abordagem B foi bem-sucedida
        
        Returns:
            True se ambiente completamente explorado E há pelo menos 1 agente vivo
        """
        # Verificar exploração completa
        fully_explored = self.grid.is_fully_explored()
        
        # Verificar se há sobreviventes
        has_survivor = any(agent.alive for agent in self.agents)
        
        return fully_explored and has_survivor
    
    def get_statistics(self) -> Dict:
        """
        Retorna estatísticas da abordagem
        
        Returns:
            Dicionário com estatísticas
        """
        total_agents = len(self.agents)
        alive_agents = sum(1 for agent in self.agents if agent.alive)
        
        zone_progress = {}
        for agent_id, zone in self.zones.items():
            zone_progress[agent_id] = zone.get_progress()
        
        avg_zone_progress = (
            sum(zone_progress.values()) / len(zone_progress) 
            if zone_progress else 0.0
        )
        
        return {
            'approach': 'B',
            'objective': 'Exploração Completa + 1 Sobrevivente',
            'success': self.check_success(),
            'total_agents': total_agents,
            'alive_agents': alive_agents,
            'exploration_percentage': self.grid.get_exploration_percentage(),
            'fully_explored': self.grid.is_fully_explored(),
            'zones_completed': len(self.fully_explored_zones),
            'total_zones': len(self.zones),
            'average_zone_progress': avg_zone_progress,
            'zone_details': {
                agent_id: {
                    'progress': zone.get_progress(),
                    'complete': zone.is_complete(),
                    'explored': zone.explored_count,
                    'total': len(zone.cells)
                }
                for agent_id, zone in self.zones.items()
            }
        }
    
    def print_progress(self):
        """Imprime progresso da exploração"""
        stats = self.get_statistics()
        
        print("=" * 60)
        print("ABORDAGEM B - EXPLORAÇÃO COMPLETA")
        print("=" * 60)
        print(f"Exploração: {stats['exploration_percentage']:.1%}")
        print(f"Agentes Vivos: {stats['alive_agents']}/{stats['total_agents']}")
        print(f"Zonas Completas: {stats['zones_completed']}/{stats['total_zones']}")
        print()
        
        for agent_id, details in stats['zone_details'].items():
            status = "✓" if details['complete'] else "⏳"
            print(f"{status} Agente {agent_id}: {details['progress']:.1%} "
                  f"({details['explored']}/{details['total']})")
        
        print()
        success = "✓ SUCESSO" if stats['success'] else "✗ EM PROGRESSO"
        print(f"Status: {success}")
        print("=" * 60)
    
    def __repr__(self):
        stats = self.get_statistics()
        return (f"ApproachB(exploration={stats['exploration_percentage']:.1%}, "
                f"alive={stats['alive_agents']}/{stats['total_agents']})")