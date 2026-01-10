"""
Agente com Naive Bayes
Responsável: Xavier Delacth
Data: 10/01/2025

Este agente usa o algoritmo Naive Bayes para decidir seus movimentos
baseado em probabilidades calculadas sobre o ambiente.
"""

import numpy as np
from typing import List, Tuple, Optional
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from agents.perception import PerceptionSystem
from environment.cell import Cell, CellType


class NaiveBayesAgent(BaseAgent):
    """
    Agente que usa Naive Bayes para tomada de decisão
    
    O agente calcula probabilidades de cada célula adjacente conter
    bomba, tesouro ou ser livre, e escolhe o movimento mais seguro.
    
    Attributes:
        model: Modelo Naive Bayes treinado
        perception: Sistema de percepção
        risk_tolerance: Tolerância a risco (0.0 a 1.0)
        exploration_bonus: Bônus para células não exploradas
    """
    
    def __init__(
        self, 
        agent_id: int, 
        start_x: int, 
        start_y: int,
        model=None,
        risk_tolerance: float = 0.3,
        exploration_bonus: float = 0.2
    ):
        """
        Inicializa o agente Naive Bayes
        
        Args:
            agent_id: ID único do agente
            start_x, start_y: Posição inicial
            model: Modelo Naive Bayes treinado (sklearn)
            risk_tolerance: Tolerância a risco (default: 0.3)
            exploration_bonus: Bônus para exploração (default: 0.2)
        """
        super().__init__(agent_id, start_x, start_y)
        
        self.model = model
        self.perception = PerceptionSystem()
        self.risk_tolerance = risk_tolerance
        self.exploration_bonus = exploration_bonus
        
        # Conhecimento acumulado
        self.known_bombs: set = set()  # Posições de bombas conhecidas
        self.known_treasures: set = set()  # Posições de tesouros conhecidos
        self.known_safe: set = set()  # Posições seguras conhecidas
        
        # Estratégia
        self.strategy = "cautious"  # "cautious", "aggressive", "balanced"
    
    def set_strategy(self, strategy: str):
        """
        Define a estratégia do agente
        
        Args:
            strategy: "cautious", "aggressive" ou "balanced"
        """
        if strategy not in ["cautious", "aggressive", "balanced"]:
            raise ValueError(f"Estratégia inválida: {strategy}")
        
        self.strategy = strategy
        
        # Ajustar parâmetros baseado na estratégia
        if strategy == "cautious":
            self.risk_tolerance = 0.2
            self.exploration_bonus = 0.1
        elif strategy == "aggressive":
            self.risk_tolerance = 0.5
            self.exploration_bonus = 0.3
        else:  # balanced
            self.risk_tolerance = 0.3
            self.exploration_bonus = 0.2
    
    def update_knowledge(self, visible_cells: List[Cell]):
        """
        Atualiza conhecimento baseado nas células visíveis
        
        Args:
            visible_cells: Lista de células que o agente pode ver
        """
        for cell in visible_cells:
            pos = (cell.x, cell.y)
            
            if cell.explored:
                if cell.is_bomb():
                    self.known_bombs.add(pos)
                elif cell.is_treasure():
                    self.known_treasures.add(pos)
                elif cell.is_free():
                    self.known_safe.add(pos)
    
    def extract_features(self, x: int, y: int, visible_cells: List[Cell]) -> np.ndarray:
        """
        Extrai features de uma posição para o modelo Naive Bayes
        
        Args:
            x, y: Posição a avaliar
            visible_cells: Células visíveis
            
        Returns:
            Array de features
        """
        features = []
        
        # Feature 1: Distância Manhattan até o agente
        manhattan_dist = abs(x - self.x) + abs(y - self.y)
        features.append(manhattan_dist)
        
        # Feature 2: Distância Euclidiana até o agente
        euclidean_dist = np.sqrt((x - self.x)**2 + (y - self.y)**2)
        features.append(euclidean_dist)
        
        # Feature 3-6: Número de cada tipo de célula ao redor
        neighbors_info = self.perception.analyze_neighbors_at(
            x, y, visible_cells
        )
        features.append(neighbors_info['bombs'])
        features.append(neighbors_info['treasures'])
        features.append(neighbors_info['free'])
        features.append(neighbors_info['unknown'])
        
        # Feature 7: Se a posição já foi visitada
        features.append(1 if (x, y) in self.known_safe else 0)
        
        # Feature 8: Se há tesouro nas proximidades
        has_nearby_treasure = any(
            (cell.x, cell.y) in self.known_treasures 
            for cell in visible_cells
        )
        features.append(1 if has_nearby_treasure else 0)
        
        # Feature 9: Densidade de bombas na área
        bomb_density = neighbors_info['bombs'] / max(neighbors_info['total'], 1)
        features.append(bomb_density)
        
        # Feature 10: Células exploradas ao redor
        explored_ratio = neighbors_info['explored'] / max(neighbors_info['total'], 1)
        features.append(explored_ratio)
        
        return np.array(features).reshape(1, -1)
    
    def calculate_cell_score(
        self, 
        x: int, 
        y: int, 
        visible_cells: List[Cell]
    ) -> float:
        """
        Calcula score de uma célula usando Naive Bayes
        
        Args:
            x, y: Posição a avaliar
            visible_cells: Células visíveis
            
        Returns:
            Score da célula (maior = melhor)
        """
        pos = (x, y)
        
        # Se já sabemos que é bomba, score mínimo
        if pos in self.known_bombs:
            return -1000.0
        
        # Se já sabemos que é seguro, score base alto
        base_score = 100.0 if pos in self.known_safe else 0.0
        
        # Se é tesouro conhecido e ainda não coletado, score máximo
        if pos in self.known_treasures:
            # Verificar se célula ainda tem tesouro
            cell = next((c for c in visible_cells if (c.x, c.y) == pos), None)
            if cell and cell.is_treasure():
                return 1000.0
        
        # Usar modelo Naive Bayes se disponível
        if self.model is not None:
            try:
                features = self.extract_features(x, y, visible_cells)
                
                # Obter probabilidades de cada classe
                probabilities = self.model.predict_proba(features)[0]
                
                # Assumindo classes: [BOMB, FREE, TREASURE]
                # Ajustar índices conforme seu treinamento
                prob_bomb = probabilities[0] if len(probabilities) > 0 else 0.5
                prob_free = probabilities[1] if len(probabilities) > 1 else 0.5
                prob_treasure = probabilities[2] if len(probabilities) > 2 else 0.0
                
                # Calcular score baseado nas probabilidades
                score = base_score
                score += prob_free * 50.0  # Recompensa por ser livre
                score += prob_treasure * 100.0  # Grande recompensa por tesouro
                score -= prob_bomb * 200.0  # Penalidade por bomba
                
                # Aplicar tolerância a risco
                if prob_bomb > self.risk_tolerance:
                    score -= 500.0  # Penalidade extra se muito arriscado
                
            except Exception as e:
                # Se modelo falhar, usar heurística simples
                print(f"Erro ao usar modelo NB: {e}")
                score = base_score
        else:
            # Sem modelo, usar heurística baseada em vizinhos
            neighbors_info = self.perception.analyze_neighbors_at(
                x, y, visible_cells
            )
            
            score = base_score
            score -= neighbors_info['bombs'] * 30.0  # Penalidade por bombas próximas
            score += neighbors_info['treasures'] * 40.0  # Bônus por tesouros próximos
            score += neighbors_info['free'] * 10.0  # Bônus por células livres
        
        # Bônus por exploração
        if pos not in self.known_safe:
            score += self.exploration_bonus * 20.0
        
        # Penalidade por distância (preferir células mais próximas)
        distance_penalty = abs(x - self.x) + abs(y - self.y)
        score -= distance_penalty * 2.0
        
        return score
    
    def get_adjacent_positions(self) -> List[Tuple[int, int]]:
        """
        Retorna posições adjacentes ao agente
        
        Returns:
            Lista de tuplas (x, y) das posições adjacentes
        """
        positions = []
        
        # Direções: Norte, Sul, Leste, Oeste
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            positions.append((new_x, new_y))
        
        return positions
    
    def decide_next_move(self, visible_cells: List[Cell]) -> Tuple[int, int]:
        """
        Decide o próximo movimento usando Naive Bayes
        
        Args:
            visible_cells: Lista de células visíveis ao redor
            
        Returns:
            Tupla (x, y) com a próxima posição
        """
        # Atualizar conhecimento
        self.update_knowledge(visible_cells)
        
        # Obter posições adjacentes
        adjacent_positions = self.get_adjacent_positions()
        
        # Calcular score para cada posição
        position_scores = []
        for x, y in adjacent_positions:
            score = self.calculate_cell_score(x, y, visible_cells)
            position_scores.append(((x, y), score))
        
        # Ordenar por score (maior primeiro)
        position_scores.sort(key=lambda item: item[1], reverse=True)
        
        # Se a melhor opção tem score muito negativo, ficar parado
        best_pos, best_score = position_scores[0]
        
        if best_score < -100.0:
            # Ficar na posição atual (movimento inválido será tratado pelo coordenador)
            return self.x, self.y
        
        return best_pos
    
    def receive_shared_knowledge(self, knowledge: dict):
        """
        Recebe conhecimento compartilhado de outros agentes
        
        Args:
            knowledge: Dicionário com informações compartilhadas
        """
        # Atualizar conhecimento de bombas
        if 'bombs' in knowledge:
            for pos in knowledge['bombs']:
                self.known_bombs.add(tuple(pos))
        
        # Atualizar conhecimento de tesouros
        if 'treasures' in knowledge:
            for pos in knowledge['treasures']:
                self.known_treasures.add(tuple(pos))
        
        # Atualizar conhecimento de células seguras
        if 'safe_cells' in knowledge:
            for pos in knowledge['safe_cells']:
                self.known_safe.add(tuple(pos))
    
    def get_knowledge_to_share(self) -> dict:
        """
        Retorna conhecimento para compartilhar com outros agentes
        
        Returns:
            Dicionário com conhecimento acumulado
        """
        return {
            'agent_id': self.id,
            'bombs': list(self.known_bombs),
            'treasures': list(self.known_treasures),
            'safe_cells': list(self.known_safe),
            'position': (self.x, self.y),
            'treasures_collected': self.treasures_collected,
            'has_treasure_power': self.has_treasure_power
        }
    
    def reset_knowledge(self):
        """Limpa todo o conhecimento acumulado"""
        self.known_bombs.clear()
        self.known_treasures.clear()
        self.known_safe.clear()
    
    def get_statistics(self) -> dict:
        """
        Retorna estatísticas do agente
        
        Returns:
            Dicionário com estatísticas
        """
        return {
            'id': self.id,
            'type': 'NaiveBayes',
            'position': (self.x, self.y),
            'alive': self.alive,
            'treasures_collected': self.treasures_collected,
            'cells_explored': self.cells_explored,
            'has_treasure_power': self.has_treasure_power,
            'known_bombs': len(self.known_bombs),
            'known_treasures': len(self.known_treasures),
            'known_safe_cells': len(self.known_safe),
            'strategy': self.strategy,
            'risk_tolerance': self.risk_tolerance
        }
    
    def __repr__(self):
        status = "ALIVE" if self.alive else "DEAD"
        return (f"NaiveBayesAgent(id={self.id}, pos=({self.x},{self.y}), "
                f"status={status}, treasures={self.treasures_collected})")
    
    def __str__(self):
        return f"NB-Agent-{self.id}"