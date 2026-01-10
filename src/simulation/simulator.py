"""
Módulo de Simulação Principal
Responsáveis: Adriana & Xavier Delacth
Data: 31/12/2025

Este módulo contém o motor principal de simulação que:
- Gerencia o ciclo completo de uma simulação
- Integra Grid, Agentes e Coordenador
- Executa as diferentes abordagens (A, B, C)
- Coleta métricas e resultados
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.grid import Grid
from environment.config import EnvironmentConfig
from agents.base_agent import BaseAgent
from simulation.coordinator import Coordinator, AgentStatus


class ApproachType(Enum):
    """Tipos de abordagem de simulação"""
    APPROACH_A = "A"  # >50% tesouros descobertos
    APPROACH_B = "B"  # Exploração completa + 1 sobrevivente
    APPROACH_C = "C"  # Encontrar a bandeira


@dataclass
class SimulationConfig:
    """
    Configurações da simulação
    
    Attributes:
        max_steps: Número máximo de steps
        approach: Tipo de abordagem
        num_agents: Número de agentes
        enable_logging: Habilitar logs
        timeout_seconds: Timeout em segundos
    """
    max_steps: int = 1000
    approach: ApproachType = ApproachType.APPROACH_A
    num_agents: int = 5
    enable_logging: bool = True
    timeout_seconds: int = 300
    seed: Optional[int] = None


@dataclass
class SimulationResult:
    """
    Resultado de uma simulação
    
    Attributes:
        success: Se a simulação foi bem-sucedida
        approach: Tipo de abordagem executada
        steps_taken: Número de steps executados
        time_elapsed: Tempo decorrido em segundos
        agents_alive: Número de agentes vivos no final
        treasures_collected: Tesouros coletados
        exploration_percentage: Percentual explorado
        flag_found: Se a bandeira foi encontrada
        metrics: Métricas detalhadas
    """
    success: bool
    approach: ApproachType
    steps_taken: int
    time_elapsed: float
    agents_alive: int
    treasures_collected: int
    exploration_percentage: float
    flag_found: bool = False
    metrics: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return {
            'success': self.success,
            'approach': self.approach.value,
            'steps_taken': self.steps_taken,
            'time_elapsed': self.time_elapsed,
            'agents_alive': self.agents_alive,
            'treasures_collected': self.treasures_collected,
            'exploration_percentage': self.exploration_percentage,
            'flag_found': self.flag_found,
            'metrics': self.metrics
        }


class Simulator:
    """
    Motor principal de simulação
    
    Gerencia a execução completa de uma simulação, integrando
    todos os componentes do sistema.
    
    Attributes:
        grid: Grid do ambiente
        agents: Lista de agentes
        coordinator: Coordenador de agentes
        config: Configurações da simulação
        knowledge_base: Base de conhecimento compartilhada
    """
    
    def __init__(
        self,
        grid: Grid,
        agents: List[BaseAgent],
        config: SimulationConfig = None,
        knowledge_base=None
    ):
        """
        Inicializa o simulador
        
        Args:
            grid: Grid do ambiente
            agents: Lista de agentes
            config: Configurações (usa padrão se None)
            knowledge_base: Base de conhecimento (opcional)
        """
        self.grid = grid
        self.agents = agents
        self.config = config or SimulationConfig()
        self.knowledge_base = knowledge_base
        
        # Criar coordenador
        self.coordinator = Coordinator(grid, agents)
        
        # Estado da simulação
        self.current_step = 0
        self.start_time = None
        self.end_time = None
        self.running = False
        
        # Callbacks para eventos
        self.on_step_callback: Optional[Callable] = None
        self.on_success_callback: Optional[Callable] = None
        self.on_failure_callback: Optional[Callable] = None
    
    def set_on_step_callback(self, callback: Callable):
        """Define callback chamado a cada step"""
        self.on_step_callback = callback
    
    def set_on_success_callback(self, callback: Callable):
        """Define callback chamado em caso de sucesso"""
        self.on_success_callback = callback
    
    def set_on_failure_callback(self, callback: Callable):
        """Define callback chamado em caso de falha"""
        self.on_failure_callback = callback
    
    def check_approach_a_success(self) -> bool:
        """
        Verifica sucesso da Abordagem A: >50% tesouros descobertos
        
        Returns:
            True se >50% dos tesouros foram descobertos
        """
        initial_treasures = self.grid.config.treasures_count
        if initial_treasures == 0:
            return True
        
        treasures_collected = sum(
            agent.treasures_collected for agent in self.agents
        )
        
        percentage = treasures_collected / initial_treasures
        return percentage > 0.5
    
    def check_approach_b_success(self) -> bool:
        """
        Verifica sucesso da Abordagem B: exploração completa + 1 vivo
        
        Returns:
            True se o ambiente foi completamente explorado e há pelo menos 1 agente vivo
        """
        fully_explored = self.grid.is_fully_explored()
        has_survivor = len(self.coordinator.get_alive_agents()) > 0
        
        return fully_explored and has_survivor
    
    def check_approach_c_success(self) -> bool:
        """
        Verifica sucesso da Abordagem C: encontrar a bandeira
        
        Returns:
            True se algum agente encontrou a bandeira
        """
        for agent in self.agents:
            if self.coordinator.agent_status.get(agent.id) == AgentStatus.SUCCESS:
                return True
        return False
    
    def check_success(self) -> bool:
        """
        Verifica se a simulação foi bem-sucedida baseado na abordagem
        
        Returns:
            True se os critérios de sucesso foram atendidos
        """
        if self.config.approach == ApproachType.APPROACH_A:
            return self.check_approach_a_success()
        elif self.config.approach == ApproachType.APPROACH_B:
            return self.check_approach_b_success()
        elif self.config.approach == ApproachType.APPROACH_C:
            return self.check_approach_c_success()
        return False
    
    def should_continue(self) -> bool:
        """
        Verifica se a simulação deve continuar
        
        Returns:
            True se deve continuar executando
        """
        # Verificar timeout
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.config.timeout_seconds:
                return False
        
        # Verificar máximo de steps
        if self.current_step >= self.config.max_steps:
            return False
        
        # Verificar se há agentes vivos
        if len(self.coordinator.get_alive_agents()) == 0:
            return False
        
        # Verificar se já atingiu sucesso (apenas para Abordagem C)
        if self.config.approach == ApproachType.APPROACH_C:
            if self.check_success():
                return False
        
        return True
    
    def step(self) -> Dict:
        """
        Executa um step da simulação
        
        Returns:
            Dicionário com estatísticas do step
        """
        self.current_step += 1
        
        # Executar coordenação
        step_stats = self.coordinator.step(self.knowledge_base)
        
        # Adicionar informações adicionais
        step_stats['success'] = self.check_success()
        step_stats['exploration'] = self.grid.get_exploration_percentage()
        
        # Chamar callback se existir
        if self.on_step_callback:
            self.on_step_callback(step_stats)
        
        return step_stats
    
    def run(self) -> SimulationResult:
        """
        Executa a simulação completa
        
        Returns:
            SimulationResult com os resultados
        """
        self.running = True
        self.start_time = time.time()
        self.current_step = 0
        
        # Loop principal
        while self.should_continue():
            self.step()
        
        self.end_time = time.time()
        self.running = False
        
        # Criar resultado
        result = self._create_result()
        
        # Chamar callbacks
        if result.success and self.on_success_callback:
            self.on_success_callback(result)
        elif not result.success and self.on_failure_callback:
            self.on_failure_callback(result)
        
        return result
    
    def _create_result(self) -> SimulationResult:
        """
        Cria objeto SimulationResult com os resultados finais
        
        Returns:
            SimulationResult
        """
        stats = self.coordinator.get_statistics()
        
        # Verificar se a bandeira foi encontrada
        flag_found = False
        if self.config.approach == ApproachType.APPROACH_C:
            flag_found = self.check_approach_c_success()
        
        result = SimulationResult(
            success=self.check_success(),
            approach=self.config.approach,
            steps_taken=self.current_step,
            time_elapsed=self.end_time - self.start_time if self.end_time else 0,
            agents_alive=stats['alive_agents'],
            treasures_collected=stats['total_treasures_collected'],
            exploration_percentage=stats['exploration_percentage'],
            flag_found=flag_found,
            metrics=stats
        )
        
        return result
    
    def reset(self):
        """Reseta o simulador para uma nova execução"""
        self.current_step = 0
        self.start_time = None
        self.end_time = None
        self.running = False
        self.coordinator.reset()
        self.grid.reset()
    
    def __repr__(self):
        return (f"Simulator(approach={self.config.approach.value}, "
                f"agents={len(self.agents)}, "
                f"step={self.current_step})")