"""
Módulo de Coordenação de Agentes
Responsável: Xavier Delacth
Data: 10/01/2026

Este módulo gerencia a execução de múltiplos agentes de forma coordenada,
prevenindo colisões e garantindo comunicação eficiente.
"""

from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.grid import Grid
from environment.cell import CellType
from agents.base_agent import BaseAgent


class AgentStatus(Enum):
    """Status possíveis de um agente"""
    ACTIVE = "active"           # Agente ativo e explorando
    DEAD = "dead"              # Agente morto (bomba)
    SUCCESS = "success"        # Agente completou objetivo
    WAITING = "waiting"        # Agente aguardando turno
    BLOCKED = "blocked"        # Agente bloqueado (sem movimentos válidos)


@dataclass
class AgentAction:
    """
    Representa uma ação de um agente
    
    Attributes:
        agent_id: ID do agente
        action_type: Tipo de ação ('move', 'explore', 'collect', 'wait')
        from_pos: Posição de origem (x, y)
        to_pos: Posição de destino (x, y)
        step: Step da simulação
    """
    agent_id: int
    action_type: str
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    step: int
    
    def __repr__(self):
        return (f"AgentAction(id={self.agent_id}, {self.action_type}, "
                f"{self.from_pos}->{self.to_pos}, step={self.step})")


class Coordinator:
    """
    Coordenador de múltiplos agentes
    
    Gerencia a execução sincronizada de agentes, prevenindo colisões
    e coordenando ações no ambiente compartilhado.
    
    Attributes:
        grid: Grid do ambiente
        agents: Lista de agentes
        agent_status: Status de cada agente
        current_step: Step atual da simulação
        action_history: Histórico de ações
    """
    
    def __init__(self, grid: Grid, agents: List[BaseAgent] = None):
        """
        Inicializa o coordenador
        
        Args:
            grid: Grid do ambiente
            agents: Lista de agentes (pode ser adicionada depois)
        """
        self.grid = grid
        self.agents: List[BaseAgent] = agents or []
        self.agent_status: Dict[int, AgentStatus] = {}
        self.current_step = 0
        self.action_history: List[AgentAction] = []
        self._agent_positions: Dict[int, Tuple[int, int]] = {}
        self._reserved_positions: Set[Tuple[int, int]] = set()
        
        # Inicializar status dos agentes
        for agent in self.agents:
            self.agent_status[agent.id] = AgentStatus.ACTIVE
            self._agent_positions[agent.id] = (agent.x, agent.y)
    
    def add_agent(self, agent: BaseAgent):
        """
        Adiciona um agente ao coordenador
        
        Args:
            agent: Agente a ser adicionado
        """
        if agent.id in self.agent_status:
            raise ValueError(f"Agente {agent.id} já existe no coordenador")
        
        self.agents.append(agent)
        self.agent_status[agent.id] = AgentStatus.ACTIVE
        self._agent_positions[agent.id] = (agent.x, agent.y)
    
    def remove_agent(self, agent_id: int):
        """
        Remove um agente do coordenador
        
        Args:
            agent_id: ID do agente a remover
        """
        self.agents = [a for a in self.agents if a.id != agent_id]
        if agent_id in self.agent_status:
            del self.agent_status[agent_id]
        if agent_id in self._agent_positions:
            del self._agent_positions[agent_id]
    
    def get_active_agents(self) -> List[BaseAgent]:
        """
        Retorna lista de agentes ativos
        
        Returns:
            Lista de agentes com status ACTIVE
        """
        return [
            agent for agent in self.agents 
            if self.agent_status.get(agent.id) == AgentStatus.ACTIVE
        ]
    
    def get_alive_agents(self) -> List[BaseAgent]:
        """
        Retorna lista de agentes vivos (não mortos)
        
        Returns:
            Lista de agentes que não estão mortos
        """
        return [
            agent for agent in self.agents 
            if self.agent_status.get(agent.id) != AgentStatus.DEAD
        ]
    
    def is_position_occupied(self, x: int, y: int, exclude_agent_id: int = None) -> bool:
        """
        Verifica se uma posição está ocupada por outro agente
        
        Args:
            x, y: Coordenadas a verificar
            exclude_agent_id: ID de agente a ignorar na verificação
            
        Returns:
            True se a posição está ocupada
        """
        for agent_id, (ax, ay) in self._agent_positions.items():
            if agent_id != exclude_agent_id and (ax, ay) == (x, y):
                return True
        return False
    
    def is_position_reserved(self, x: int, y: int) -> bool:
        """
        Verifica se uma posição está reservada para movimento
        
        Args:
            x, y: Coordenadas
            
        Returns:
            True se a posição está reservada
        """
        return (x, y) in self._reserved_positions
    
    def reserve_position(self, x: int, y: int):
        """Reserva uma posição temporariamente"""
        self._reserved_positions.add((x, y))
    
    def clear_reservations(self):
        """Limpa todas as reservas de posição"""
        self._reserved_positions.clear()
    
    def get_valid_moves(self, agent: BaseAgent) -> List[Tuple[int, int]]:
        """
        Retorna movimentos válidos para um agente
        
        Args:
            agent: Agente
            
        Returns:
            Lista de posições (x, y) válidas
        """
        valid_moves = []
        
        # Direções possíveis: Norte, Sul, Leste, Oeste
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        
        for dx, dy in directions:
            new_x, new_y = agent.x + dx, agent.y + dy
            
            # Verificar se está dentro do grid
            if not self.grid.is_valid_position(new_x, new_y):
                continue
            
            # Verificar se não está ocupado por outro agente
            if self.is_position_occupied(new_x, new_y, exclude_agent_id=agent.id):
                continue
            
            # Verificar se não está reservado
            if self.is_position_reserved(new_x, new_y):
                continue
            
            valid_moves.append((new_x, new_y))
        
        return valid_moves
    
    def execute_agent_action(
        self, 
        agent: BaseAgent, 
        new_x: int, 
        new_y: int,
        knowledge_base = None
    ) -> bool:
        """
        Executa a ação de movimento de um agente
        
        Args:
            agent: Agente que vai se mover
            new_x, new_y: Nova posição
            knowledge_base: Base de conhecimento compartilhada (opcional)
            
        Returns:
            True se a ação foi executada com sucesso
        """
        # Verificar se o movimento é válido
        if not self.grid.is_valid_position(new_x, new_y):
            return False
        
        if self.is_position_occupied(new_x, new_y, exclude_agent_id=agent.id):
            return False
        
        # Registrar ação
        action = AgentAction(
            agent_id=agent.id,
            action_type='move',
            from_pos=(agent.x, agent.y),
            to_pos=(new_x, new_y),
            step=self.current_step
        )
        self.action_history.append(action)
        
        # Obter célula de destino
        cell = self.grid.get_cell(new_x, new_y)
        
        # Mover agente
        old_x, old_y = agent.x, agent.y
        agent.move(new_x, new_y)
        self._agent_positions[agent.id] = (new_x, new_y)
        
        # Marcar célula como explorada
        self.grid.mark_cell_explored(new_x, new_y, agent.id, self.current_step)
        
        # Compartilhar descoberta (se knowledge_base existir)
        if knowledge_base:
            knowledge_base.share_discovery(
                agent_id=agent.id,
                position=(new_x, new_y),
                cell_type=cell.type,
                step=self.current_step
            )
        
        # Processar interação com a célula
        if cell.is_bomb():
            if agent.has_treasure_power:
                # Agente com poder de tesouro desativa a bomba
                cell.deactivate_bomb()
                agent.has_treasure_power = False  # Consumir poder
                if knowledge_base:
                    knowledge_base.share_bomb_deactivation(
                        agent_id=agent.id,
                        position=(new_x, new_y),
                        step=self.current_step
                    )
            else:
                # Agente morre
                agent.die()
                self.agent_status[agent.id] = AgentStatus.DEAD
                if knowledge_base:
                    knowledge_base.share_agent_death(
                        agent_id=agent.id,
                        position=(new_x, new_y),
                        step=self.current_step
                    )
                return False
        
        elif cell.is_treasure():
            # Coletar tesouro
            agent.collect_treasure()
            cell.consume_treasure()
            if knowledge_base:
                knowledge_base.share_treasure_collection(
                    agent_id=agent.id,
                    position=(new_x, new_y),
                    step=self.current_step
                )
        
        elif cell.is_flag():
            # Encontrou a bandeira!
            self.agent_status[agent.id] = AgentStatus.SUCCESS
            if knowledge_base:
                knowledge_base.share_flag_discovery(
                    agent_id=agent.id,
                    position=(new_x, new_y),
                    step=self.current_step
                )
        
        return True
    
    def step(self, knowledge_base=None) -> Dict[str, any]:
        """
        Executa um step de coordenação (todos os agentes ativos se movem)
        
        Args:
            knowledge_base: Base de conhecimento compartilhada
            
        Returns:
            Dicionário com estatísticas do step
        """
        self.current_step += 1
        self.clear_reservations()
        
        active_agents = self.get_active_agents()
        moves_executed = 0
        agents_died = 0
        treasures_collected = 0
        flags_found = 0
        
        # Executar ações de cada agente
        for agent in active_agents:
            if not agent.alive:
                continue
            
            # Obter células visíveis
            visible_cells = self.grid.get_neighbors(
                agent.x, agent.y, 
                include_diagonals=True
            )
            
            # Agente decide próximo movimento
            try:
                next_x, next_y = agent.decide_next_move(visible_cells)
            except Exception as e:
                # Se agente falhar, pular turno
                print(f"Agente {agent.id} falhou ao decidir movimento: {e}")
                continue
            
            # Verificar se o movimento é válido
            valid_moves = self.get_valid_moves(agent)
            
            if (next_x, next_y) not in valid_moves:
                # Movimento inválido, agente fica bloqueado neste turno
                continue
            
            # Reservar posição para evitar colisões
            self.reserve_position(next_x, next_y)
            
            # Executar movimento
            old_status = self.agent_status[agent.id]
            success = self.execute_agent_action(
                agent, next_x, next_y, knowledge_base
            )
            
            if success:
                moves_executed += 1
                
                # Verificar mudanças de status
                new_status = self.agent_status[agent.id]
                
                if new_status == AgentStatus.DEAD and old_status != AgentStatus.DEAD:
                    agents_died += 1
                
                if new_status == AgentStatus.SUCCESS:
                    flags_found += 1
                
                # Verificar se coletou tesouro
                cell = self.grid.get_cell(next_x, next_y)
                if agent.treasures_collected > 0:
                    treasures_collected += 1
        
        return {
            'step': self.current_step,
            'active_agents': len(active_agents),
            'moves_executed': moves_executed,
            'agents_died': agents_died,
            'treasures_collected': treasures_collected,
            'flags_found': flags_found,
            'alive_agents': len(self.get_alive_agents())
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Retorna estatísticas gerais da coordenação
        
        Returns:
            Dicionário com estatísticas
        """
        total_agents = len(self.agents)
        alive_agents = len(self.get_alive_agents())
        active_agents = len(self.get_active_agents())
        dead_agents = sum(
            1 for status in self.agent_status.values() 
            if status == AgentStatus.DEAD
        )
        success_agents = sum(
            1 for status in self.agent_status.values() 
            if status == AgentStatus.SUCCESS
        )
        
        total_treasures = sum(agent.treasures_collected for agent in self.agents)
        total_cells_explored = self.grid.count_explored()
        exploration_percentage = self.grid.get_exploration_percentage()
        
        return {
            'total_agents': total_agents,
            'alive_agents': alive_agents,
            'active_agents': active_agents,
            'dead_agents': dead_agents,
            'success_agents': success_agents,
            'total_treasures_collected': total_treasures,
            'total_cells_explored': total_cells_explored,
            'exploration_percentage': exploration_percentage,
            'current_step': self.current_step,
            'total_actions': len(self.action_history)
        }
    
    def reset(self):
        """Reseta o coordenador para uma nova simulação"""
        self.current_step = 0
        self.action_history.clear()
        self._reserved_positions.clear()
        
        for agent in self.agents:
            self.agent_status[agent.id] = AgentStatus.ACTIVE
            self._agent_positions[agent.id] = (agent.x, agent.y)
    
    def __repr__(self):
        stats = self.get_statistics()
        return (f"Coordinator(agents={stats['total_agents']}, "
                f"alive={stats['alive_agents']}, "
                f"step={self.current_step})")