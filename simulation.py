import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import time
from collections import deque, defaultdict
import heapq
from datetime import datetime

# Cores para agentes na visualização
AGENT_COLORS = ['🔴', '🔵', '🟡', '🟢', '🟣', '🟠', '⚪', '🟤', '🟥', '🟦']

class CommunicationHub:
    """
    Hub central de comunicação entre agentes
    
    Responsabilidades:
    - Armazenar conhecimento global compartilhado
    - Facilitar troca de mensagens entre agentes
    - Coordenar estratégias coletivas
    - Mapear territórios explorados
    """
    
    def __init__(self, environment_size=10):
        self.size = environment_size
        
        # ===== CONHECIMENTO COMPARTILHADO =====
        
        # Mapa de exploração: {(x,y): [agent_ids que passaram]}
        self.exploration_map = defaultdict(list)
        
        # Células seguras confirmadas: {(x,y): 'L'/'T'/'F'}
        self.safe_cells = {}
        
        # Células perigosas confirmadas: {(x,y): 'B'}
        self.danger_cells = {}
        
        # Tesouros encontrados: {(x,y): agent_id_que_encontrou}
        self.treasures_found = {}
        
        # Bombas desativadas: {(x,y): agent_id_que_desativou}
        self.bombs_deactivated = {}
        
        # ===== ROTAS E CAMINHOS =====
        
        # Rotas seguras conhecidas: [(x1,y1), (x2,y2), ...]
        self.safe_routes = []
        
        # Caminhos individuais: {agent_id: [(x,y), ...]}
        self.agent_paths = defaultdict(list)
        
        # ===== COORDENAÇÃO =====
        
        # Território atribuído: {agent_id: [(x,y), ...]}
        self.territory_assignments = {}
        
        # Alvos prioritários: [(x,y), prioridade, tipo]
        self.priority_targets = []
        
        # Áreas bloqueadas (sem saída): {(x,y)}
        self.blocked_areas = set()
        
        # ===== MENSAGENS =====
        
        # Histórico de mensagens: [{timestamp, sender, type, data}]
        self.message_history = []
        
        # Fila de mensagens pendentes: {agent_id: [messages]}
        self.message_queue = defaultdict(list)
        
        # ===== ESTATÍSTICAS =====
        
        self.stats = {
            'total_messages': 0,
            'cells_explored': 0,
            'safe_cells_found': 0,
            'dangers_found': 0,
            'treasures_discovered': 0,
            'coordination_events': 0
        }
    
    # ========================================================================
    # MÉTODOS DE REGISTRO DE DESCOBERTAS
    # ========================================================================
    
    def register_cell_visit(self, agent_id, position, cell_type):
        """
        Registra visita de um agente a uma célula
        
        Args:
            agent_id: ID do agente
            position: (x, y) da célula
            cell_type: 'L', 'B', 'T', 'F'
        """
        x, y = position
        
        # Registrar no mapa de exploração
        if agent_id not in self.exploration_map[position]:
            self.exploration_map[position].append(agent_id)
            self.stats['cells_explored'] += 1
        
        # Adicionar ao caminho do agente
        if position not in self.agent_paths[agent_id]:
            self.agent_paths[agent_id].append(position)
        
        # Classificar célula
        if cell_type == 'B':
            self.danger_cells[position] = 'B'
            self.stats['dangers_found'] += 1
            self._broadcast_danger_alert(agent_id, position)
        
        elif cell_type in ['L', 'T', 'F']:
            self.safe_cells[position] = cell_type
            self.stats['safe_cells_found'] += 1
            
            if cell_type == 'T':
                self.treasures_found[position] = agent_id
                self.stats['treasures_discovered'] += 1
                self._broadcast_treasure_found(agent_id, position)
    
    def register_bomb_deactivation(self, agent_id, position):
        """Registra desativação de bomba (com proteção de tesouro)"""
        self.bombs_deactivated[position] = agent_id
        self._broadcast_message(agent_id, 'bomb_deactivated', {
            'position': position,
            'safe_for': agent_id
        })
    
    def register_blocked_area(self, position):
        """Marca área como bloqueada (sem saída)"""
        self.blocked_areas.add(position)
    
    # ========================================================================
    # MÉTODOS DE CONSULTA DE INFORMAÇÃO
    # ========================================================================
    
    def get_safe_cells(self):
        """Retorna todas as células seguras conhecidas"""
        return dict(self.safe_cells)
    
    def get_danger_cells(self):
        """Retorna todas as células perigosas conhecidas"""
        return dict(self.danger_cells)
    
    def get_unexplored_neighbors(self, position, environment):
        """
        Retorna vizinhos não explorados de uma posição
        
        Args:
            position: (x, y)
            environment: objeto Environment
            
        Returns:
            [(x, y), ...] células não exploradas
        """
        x, y = position
        unexplored = []
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.size and 0 <= ny < self.size):
                if (nx, ny) not in self.exploration_map:
                    unexplored.append((nx, ny))
        
        return unexplored
    
    def get_agent_territory(self, agent_id):
        """Retorna território atribuído a um agente"""
        return self.territory_assignments.get(agent_id, [])
    
    def is_cell_safe_for_agent(self, agent_id, position):
        """
        Verifica se célula é segura para um agente específico
        
        Considera:
        - Células conhecidas como seguras
        - Bombas desativadas pelo próprio agente
        """
        # Célula segura global
        if position in self.safe_cells:
            return True
        
        # Bomba desativada por este agente
        if position in self.bombs_deactivated:
            if self.bombs_deactivated[position] == agent_id:
                return True
        
        # Célula perigosa para este agente
        if position in self.danger_cells:
            return False
        
        # Desconhecida
        return None  # Incerto
    
    def is_cell_explored(self, position):
        """Verifica se célula já foi explorada por algum agente"""
        return position in self.exploration_map
    
    def is_cell_safe(self, position):
        """Verifica se célula é conhecida como segura"""
        return position in self.safe_cells
    
    def is_bomb_active(self, position):
        """Verifica se há bomba ativa conhecida na célula"""
        return position in self.danger_cells and position not in self.bombs_deactivated
    
    def get_safe_route_to(self, start, goal):
        """
        Encontra rota segura entre dois pontos usando células conhecidas
        
        Returns:
            [(x,y), ...] ou None se não encontrar
        """
        from collections import deque
        
        if start == goal:
            return [start]
        
        # BFS apenas em células seguras conhecidas
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == goal:
                return path
            
            x, y = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                if neighbor not in visited:
                    # Apenas células seguras conhecidas
                    if neighbor in self.safe_cells:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        return None  # Sem rota segura conhecida
    
    # ========================================================================
    # MÉTODOS DE MENSAGENS
    # ========================================================================
    
    def _broadcast_message(self, sender_id, msg_type, data):
        """Envia mensagem broadcast para todos os agentes"""
        message = {
            'timestamp': datetime.now().isoformat(),
            'sender': sender_id,
            'type': msg_type,
            'data': data
        }
        
        self.message_history.append(message)
        self.stats['total_messages'] += 1
        
        # Adicionar a todas as filas (exceto sender)
        for agent_id in self.agent_paths.keys():
            if agent_id != sender_id:
                self.message_queue[agent_id].append(message)
    
    def _broadcast_danger_alert(self, sender_id, position):
        """Alerta de perigo encontrado"""
        self._broadcast_message(sender_id, 'danger_alert', {
            'position': position,
            'warning': 'Bomba detectada!'
        })
    
    def _broadcast_treasure_found(self, sender_id, position):
        """Alerta de tesouro encontrado"""
        self._broadcast_message(sender_id, 'treasure_found', {
            'position': position,
            'finder': sender_id
        })
    
    def send_direct_message(self, sender_id, receiver_id, msg_type, data):
        """Envia mensagem direta entre dois agentes"""
        message = {
            'timestamp': datetime.now().isoformat(),
            'sender': sender_id,
            'receiver': receiver_id,
            'type': msg_type,
            'data': data
        }
        
        self.message_queue[receiver_id].append(message)
        self.stats['total_messages'] += 1
    
    def get_messages(self, agent_id):
        """Recupera mensagens pendentes de um agente"""
        messages = self.message_queue[agent_id].copy()
        self.message_queue[agent_id].clear()
        return messages
    
    # ========================================================================
    # MÉTODOS DE COORDENAÇÃO ESTRATÉGICA
    # ========================================================================
    
    def assign_territories(self, agent_ids, approach='grid'):
        """
        Divide ambiente em territórios para cada agente
        
        Estratégias:
        - 'grid': divisão em grade
        - 'quadrant': divisão em quadrantes
        - 'dynamic': baseado em posição atual
        """
        if approach == 'grid':
            self._assign_grid_territories(agent_ids)
        elif approach == 'quadrant':
            self._assign_quadrant_territories(agent_ids)
        else:
            self._assign_dynamic_territories(agent_ids)
        
        self.stats['coordination_events'] += 1
    
    def _assign_grid_territories(self, agent_ids):
        """Divisão em faixas horizontais"""
        num_agents = len(agent_ids)
        rows_per_agent = self.size // num_agents
        
        for i, agent_id in enumerate(agent_ids):
            start_row = i * rows_per_agent
            end_row = start_row + rows_per_agent if i < num_agents - 1 else self.size
            
            territory = []
            for x in range(start_row, end_row):
                for y in range(self.size):
                    territory.append((x, y))
            
            self.territory_assignments[agent_id] = territory
    
    def _assign_quadrant_territories(self, agent_ids):
        """Divisão em quadrantes"""
        num_agents = len(agent_ids)
        mid = self.size // 2
        
        quadrants = [
            [(x, y) for x in range(mid) for y in range(mid)],           # Q1
            [(x, y) for x in range(mid) for y in range(mid, self.size)], # Q2
            [(x, y) for x in range(mid, self.size) for y in range(mid)], # Q3
            [(x, y) for x in range(mid, self.size) for y in range(mid, self.size)] # Q4
        ]
        
        for i, agent_id in enumerate(agent_ids):
            if i < len(quadrants):
                self.territory_assignments[agent_id] = quadrants[i]
    
    def _assign_dynamic_territories(self, agent_ids):
        """Atribuição dinâmica baseada em exploração"""
        # Cada agente continua explorando sua vizinhança
        # (implementação mais complexa - omitida por simplicidade)
        pass
    
    def suggest_next_target(self, agent_id, current_position, approach):
        """
        Sugere próximo alvo para um agente baseado em:
        - Células não exploradas no território
        - Objetivos da abordagem (A, B, C)
        - Coordenação com outros agentes
        
        Returns:
            (x, y) ou None
        """
        territory = self.get_agent_territory(agent_id)
        
        # Encontrar células não exploradas no território
        unexplored_in_territory = [
            cell for cell in territory 
            if cell not in self.exploration_map
        ]
        
        if not unexplored_in_territory:
            # Território completo - explorar fora
            unexplored_in_territory = [
                (x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in self.exploration_map
            ]
        
        if not unexplored_in_territory:
            return None
        
        # Escolher célula mais próxima
        def distance(pos):
            return abs(pos[0] - current_position[0]) + abs(pos[1] - current_position[1])
        
        return min(unexplored_in_territory, key=distance)
    
    # ========================================================================
    # MÉTODOS DE ANÁLISE
    # ========================================================================
    
    def get_exploration_heatmap(self):
        """
        Retorna mapa de calor de exploração
        
        Returns:
            numpy array 10x10 com contagem de visitas
        """
        heatmap = np.zeros((self.size, self.size), dtype=int)
        
        for (x, y), agents in self.exploration_map.items():
            heatmap[x, y] = len(agents)
        
        return heatmap
    
    def get_communication_stats(self):
        """Retorna estatísticas de comunicação"""
        return {
            **self.stats,
            'unique_cells_explored': len(self.exploration_map),
            'safe_routes_known': len(self.safe_routes),
            'active_agents': len(self.agent_paths),
            'messages_in_queue': sum(len(q) for q in self.message_queue.values())
        }
    
    def export_knowledge_base(self):
        """Exporta base de conhecimento como JSON"""
        return {
            'safe_cells': {str(k): v for k, v in self.safe_cells.items()},
            'danger_cells': {str(k): v for k, v in self.danger_cells.items()},
            'treasures_found': {str(k): v for k, v in self.treasures_found.items()},
            'exploration_coverage': len(self.exploration_map) / (self.size * self.size) * 100,
            'stats': self.get_communication_stats()
        }

class Environment:
    """Ambiente 10x10 com células L, B, T e opcionalmente F"""
    
    def __init__(self, bomb_ratio=0.5, treasure_count=5, approach='A', seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.size = 10
        self.grid = np.empty((self.size, self.size), dtype=object)
        self.bomb_ratio = bomb_ratio
        self.treasure_count = treasure_count if approach != 'B' else 0
        self.approach = approach
        self.explored = np.zeros((self.size, self.size), dtype=bool)
        self.shared_knowledge = {}  # {(x,y): 'L'/'B'/'T'/'F'} - LEGACY: manter para compatibilidade
        self.collect_states = False
        
        # ⭐ NOVO: Sistema de comunicação explícito
        self.communication_hub = CommunicationHub()
        
        #  NOVO: Armazenar posição da bandeira (apenas para referência, NÃO compartilhada)
        self.flag_position = None
        
        self._generate_environment()
        
        #  IMPORTANTE: NÃO adicionar bandeira ao shared_knowledge
        # Os agentes devem DESCOBRIR a bandeira explorando

    
    def _generate_environment(self):
        """Gera ambiente aleatório garantindo factibilidade"""
        max_attempts = 50
        
        for attempt in range(max_attempts):
            total_cells = self.size * self.size
            num_bombs = int(total_cells * self.bomb_ratio)
            
            # Inicializar tudo como livre
            self.grid.fill('L')
            
            # Posicionar bombas (excluindo posição inicial 0,0)
            positions = [(i, j) for i in range(self.size) for j in range(self.size) 
                        if not (i == 0 and j == 0)]
            np.random.shuffle(positions)
            
            available_positions = min(len(positions), num_bombs)
            
            for i in range(available_positions):
                x, y = positions[i]
                self.grid[x, y] = 'B'
            
            # Posicionar tesouros (apenas se não for abordagem B)
            if self.approach != 'B':
                treasure_positions = [p for p in positions[num_bombs:] if self.grid[p[0], p[1]] == 'L']
                for i in range(min(self.treasure_count, len(treasure_positions))):
                    x, y = treasure_positions[i]
                    self.grid[x, y] = 'T'
            
            # ⭐ MODIFICADO: Posicionar bandeira e armazenar posição (apenas abordagem C)
            if self.approach == 'C':
                flag_positions = [p for p in positions if self.grid[p[0], p[1]] == 'L']
                if flag_positions:
                    x, y = flag_positions[0]
                    self.grid[x, y] = 'F'
                    self.flag_position = (x, y)  # ⭐ Apenas para referência interna
            
            # Verificar se o ambiente é factível
            if self.is_feasible():
                break
        else:
            print(f"Aviso: Não foi possível gerar ambiente factível após {max_attempts} tentativas.")
            self._create_guaranteed_feasible_environment()

    
    def get_cell(self, x, y):
        """Retorna conteúdo da célula"""
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[x, y]
        return None
    
    def mark_explored(self, x, y):
        """Marca célula como explorada"""
        if 0 <= x < self.size and 0 <= y < self.size:
            self.explored[x, y] = True
            cell_type = self.grid[x, y]
            self.shared_knowledge[(x, y)] = cell_type
            return cell_type
        return None
    
    def get_neighbors(self, x, y):
        """Retorna vizinhos válidos (4-conectados)"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors
    
    def get_exploration_percentage(self):
        """Percentagem do ambiente explorado"""
        return (np.sum(self.explored) / (self.size * self.size)) * 100
    
    def count_treasures(self):
        """Conta tesouros totais no ambiente"""
        return np.sum(self.grid == 'T')
    
    def _create_guaranteed_feasible_environment(self):
        """Cria um ambiente garantidamente factível usando estratégia inteligente"""
        from collections import deque
        
        # Reset grid
        self.grid = np.full((self.size, self.size), 'L', dtype='<U1')
        self.grid[0, 0] = 'L'
        
        # Para abordagem A
        if self.approach == 'A':
            treasure_pos = (self.size-1, self.size-1)
            self._create_path_to_target((0, 0), treasure_pos)
            self.grid[treasure_pos[0], treasure_pos[1]] = 'T'
            
            safe_positions = [(i, j) for i in range(self.size) for j in range(self.size) 
                            if self.grid[i, j] == 'L' and (i, j) != (0, 0)]
            if len(safe_positions) > 5:
                treasure_positions = safe_positions[:min(3, len(safe_positions)//2)]
                for pos in treasure_positions:
                    self.grid[pos[0], pos[1]] = 'T'
        
        # Para abordagem B
        elif self.approach == 'B':
            targets = [(self.size-1, 0), (0, self.size-1), (self.size-1, self.size-1)]
            for target in targets:
                self._create_path_to_target((0, 0), target)
        
        #  MODIFICADO: Para abordagem C - Armazenar posição da bandeira
        elif self.approach == 'C':
            flag_pos = (self.size-1, self.size-1)
            self._create_path_to_target((0, 0), flag_pos)
            self.grid[flag_pos[0], flag_pos[1]] = 'F'
            self.flag_position = flag_pos  #  Apenas para referência interna
        
        # Adicionar bombas (código permanece o mesmo)
        all_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        safe_positions = [pos for pos in all_positions if self.grid[pos[0], pos[1]] == 'L']
        
        max_bombs = int(len(all_positions) * 0.3)
        bomb_candidates = [pos for pos in safe_positions if pos != (0, 0)]
        
        viable_bomb_positions = []
        for pos in bomb_candidates:
            temp_grid = self.grid.copy()
            temp_grid[pos[0], pos[1]] = 'B'
            if self._is_connected_after_removal(temp_grid, pos):
                viable_bomb_positions.append(pos)
        
        bomb_positions = viable_bomb_positions[:max_bombs]
        for pos in bomb_positions:
            self.grid[pos[0], pos[1]] = 'B'
    
    def _create_path_to_target(self, start, target):
        """Cria um caminho garantido do início até o alvo"""
        from collections import deque
        
        # Usar BFS para encontrar caminho
        queue = deque([(start, [])])
        visited = set([start])
        
        while queue:
            (x, y), path = queue.popleft()
            current_path = path + [(x, y)]
            
            if (x, y) == target:
                # Marcar caminho como seguro (exceto se já é objetivo)
                for px, py in current_path[:-1]:  # Não sobrescrever o alvo
                    if self.grid[px, py] == 'L':  # Só sobrescrever células vazias
                        self.grid[px, py] = 'L'
                return True
            
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), current_path))
        
        return False
    
    def _is_connected_after_removal(self, temp_grid, removed_pos):
        """Verifica se o ambiente permanece conectado após remover uma posição"""
        from collections import deque
        
        # Verificar se ainda há caminho do início até os objetivos
        safe_cells = [(i, j) for i in range(self.size) for j in range(self.size) 
                     if temp_grid[i, j] != 'B']
        
        if len(safe_cells) < 2:
            return False
        
        # BFS para verificar conectividade
        visited = set()
        queue = deque([(0, 0)])
        visited.add((0, 0))
        
        while queue:
            x, y = queue.popleft()
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) not in visited and temp_grid[nx, ny] != 'B':
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        # Verificar se objetivos principais ainda são alcançáveis
        reachable_cells = len(visited)
        total_safe = len(safe_cells)
        
        if reachable_cells / total_safe < 0.5:  # Pelo menos 50% conectividade
            return False
        
        # Verificações específicas por abordagem
        if self.approach == 'A':
            treasures = [(i, j) for i, j in safe_cells if temp_grid[i, j] == 'T']
            reachable_treasures = [pos for pos in treasures if pos in visited]
            return len(reachable_treasures) >= 1
        
        elif self.approach == 'B':
            return reachable_cells >= 10  # Mínimo para exploração
        
        elif self.approach == 'C':
            flag_pos = None
            for i in range(self.size):
                for j in range(self.size):
                    if temp_grid[i, j] == 'F':
                        flag_pos = (i, j)
                        break
                if flag_pos:
                    break
            return flag_pos in visited
        
        return True

    def is_feasible(self):
        """Verifica se o ambiente é matematicamente possível com chances reais de sucesso"""
        from collections import deque
        
        # Células seguras (não bombas)
        safe_cells = [(i, j) for i in range(self.size) for j in range(self.size) 
                     if self.grid[i, j] != 'B']
        
        if len(safe_cells) < 5:  # Mínimo necessário para qualquer abordagem
            return False
        
        # Verificar conectividade básica usando BFS
        visited = set()
        queue = deque([(0, 0)])  # Começar da posição inicial
        visited.add((0, 0))
        
        while queue:
            x, y = queue.popleft()
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) not in visited and self.grid[nx, ny] != 'B':
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        # Calcular métricas de conectividade
        reachable_cells = len(visited)
        total_safe = len(safe_cells)
        connectivity_ratio = reachable_cells / total_safe if total_safe > 0 else 0
        
        # Verificações específicas por abordagem com critérios mais rigorosos
        if self.approach == 'A':
            # Abordagem A: Encontrar tesouros
            treasures = [(i, j) for i, j in safe_cells if self.grid[i, j] == 'T']
            reachable_treasures = [pos for pos in treasures if pos in visited]
            
            # Critérios para Abordagem A:
            # 1. Pelo menos 30% das células seguras devem ser alcançáveis
            # 2. Pelo menos 1 tesouro deve ser alcançável
            # 3. Deve haver pelo menos 3 células seguras alcançáveis (espaço mínimo)
            min_reachable = max(3, int(total_safe * 0.3))
            
            return (connectivity_ratio >= 0.3 and 
                   reachable_cells >= min_reachable and 
                   len(reachable_treasures) >= 1)
        
        elif self.approach == 'B':
            # Abordagem B: Explorar o ambiente
            # Critérios para Abordagem B:
            # 1. Pelo menos 50% das células seguras devem ser alcançáveis
            # 2. Deve haver pelo menos 15 células seguras alcançáveis
            # 3. Razão de conectividade deve ser alta para exploração
            
            return (connectivity_ratio >= 0.5 and 
                   reachable_cells >= 15)
        
        elif self.approach == 'C':
            # Abordagem C: Encontrar bandeira
            flag_pos = None
            for i in range(self.size):
                for j in range(self.size):
                    if self.grid[i, j] == 'F':
                        flag_pos = (i, j)
                        break
                if flag_pos:
                    break
            
            # Critérios para Abordagem C:
            # 1. Bandeira deve existir
            # 2. Bandeira deve ser alcançável
            # 3. Pelo menos 25% das células seguras devem ser alcançáveis
            
            return (flag_pos is not None and 
                   flag_pos in visited and 
                   connectivity_ratio >= 0.25)
        
        return False


class MLModel:
    """Modelo de aprendizagem de máquina wrapper"""
    
    def __init__(self, model_type='knn', **params):
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=params.get('k', 3))
        elif model_type == 'naive_bayes':
            self.model = GaussianNB()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 10),
                max_depth=params.get('max_depth', 5),
                random_state=42
            )
        
        self.is_trained = False
    
    def train(self, X, y):
        """Treina o modelo"""
        if len(X) > 0:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
    
    def predict(self, X):
        """Faz predição"""
        if not self.is_trained or len(X) == 0:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Retorna probabilidades de predição"""
        if not self.is_trained or len(X) == 0:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # Para modelos sem predict_proba, usar predição binária
            pred = self.model.predict(X_scaled)
            proba = np.zeros((len(pred), 2))
            proba[np.arange(len(pred)), pred.astype(int)] = 1.0
            return proba



class Agent:
    """Agente inteligente com múltiplos modelos"""
    
    def __init__(self, agent_id, start_pos, weights=None):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.has_treasure_protection = False
        self.treasures_found = 0
        self.steps_taken = 0
        self.bombs_activated = 0
        self.bombs_deactivated = 0  # Rastreamento de bombas desativadas com proteção de tesouro
        self.deactivated_bombs = set()  # Conjunto de posições (x,y) de bombas desativadas por este agente
        
        # Modelos de aprendizagem
        self.models = {
            'knn': MLModel('knn', k=3),
            'naive_bayes': MLModel('naive_bayes'),
            'random_forest': MLModel('random_forest', n_estimators=10, max_depth=5)
        }
        
        # Pesos dos modelos (padrão: todos iguais)
        if weights is None:
            weights = {'knn': 1/3, 'naive_bayes': 1/3, 'random_forest': 1/3}
        self.weights = weights
        
        # Histórico de observações
        self.observations = []
        self.labels = []
    
    def observe(self, environment, x, y):
        """Cria vetor de features baseado na observação com comunicação inter-agentes"""
        features = []
        
        # Posição normalizada
        features.append(x / environment.size)
        features.append(y / environment.size)
        
        # Informações dos vizinhos conhecidos (comunicação inter-agentes)
        neighbors = environment.get_neighbors(x, y)
        bomb_neighbors = 0
        treasure_neighbors = 0
        safe_neighbors = 0  # Células conhecidas como seguras
        unknown_neighbors = 0
        
        for nx, ny in neighbors:
            neighbor_pos = (nx, ny)
            if neighbor_pos in environment.shared_knowledge:
                cell = environment.shared_knowledge[neighbor_pos]
                if cell == 'B':
                    # Verifica se este agente desativou esta bomba
                    if neighbor_pos in self.deactivated_bombs:
                        safe_neighbors += 1  # Segura para mim
                    else:
                        bomb_neighbors += 1  # Perigosa para mim
                elif cell in ['T', 'T_FOUND', 'F']:
                    treasure_neighbors += 1
                elif cell == 'L':
                    safe_neighbors += 1
            else:
                unknown_neighbors += 1
        
        features.append(bomb_neighbors / 4)
        features.append(treasure_neighbors / 4)
        features.append(safe_neighbors / 4)  # Usar safe_neighbors ao invés de unknown
        
        # Distância até o centro
        center_dist = np.sqrt((x - environment.size/2)**2 + (y - environment.size/2)**2)
        features.append(center_dist / (environment.size * np.sqrt(2)))
        
        return np.array(features)
    
    def update_knowledge(self, environment, x, y, result):
        """Atualiza conhecimento do agente"""
        obs = self.observe(environment, x, y)
        self.observations.append(obs)
        
        # Label: 1 se seguro (L, T, F), 0 se perigoso (B)
        label = 1 if result in ['L', 'T', 'F'] else 0
        self.labels.append(label)
        
        # Retreinar modelos periodicamente
        if len(self.observations) >= 5:
            X = np.array(self.observations)
            y = np.array(self.labels)
            
            for model in self.models.values():
                model.train(X, y)
    
    def decide_next_move(self, environment):
        """Decide próximo movimento usando ensemble de modelos com comunicação explícita"""
        if not self.alive:
            return None
        
        # ⭐ NOVO: Usar CommunicationHub para priorizar células
        possible_moves = []
        unexplored_moves = []
        known_safe_moves = []
        known_dangerous_moves = []
        
        for nx, ny in environment.get_neighbors(*self.position):
            move = (nx, ny)
            
            # Classificar movimento baseado no conhecimento compartilhado
            if not environment.communication_hub.is_cell_explored(move):
                unexplored_moves.append(move)
            elif environment.communication_hub.is_cell_safe(move):
                known_safe_moves.append(move)
            elif environment.communication_hub.is_bomb_active(move):
                # Bomba ativa conhecida - evitar completamente
                known_dangerous_moves.append(move)
            else:
                # Explorada mas não classificada como segura ou bomba ativa
                possible_moves.append(move)
        
        # ⭐ ORDEM DE PRIORIDADE usando CommunicationHub:
        # 1. Células não exploradas (exploração)
        # 2. Células conhecidas como seguras
        # 3. Outras células exploradas (fallback)
        # ❌ EVITAR: Bombas ativas conhecidas
        
        candidate_moves = []
        if unexplored_moves:
            candidate_moves = unexplored_moves
        elif known_safe_moves:
            candidate_moves = known_safe_moves
        elif possible_moves:
            candidate_moves = possible_moves
        
        if not candidate_moves:
            return None
        
        # Se modelos não estão treinados, usar conhecimento compartilhado
        if not any(m.is_trained for m in self.models.values()):
            # Priorizar baseado no CommunicationHub
            if known_safe_moves and known_safe_moves[0] in candidate_moves:
                return known_safe_moves[np.random.randint(len(known_safe_moves))]
            elif unexplored_moves:
                return unexplored_moves[np.random.randint(len(unexplored_moves))]
            else:
                return candidate_moves[np.random.randint(len(candidate_moves))]
        
        # Avaliar cada movimento possível com modelos ML
        move_scores = []
        for move in candidate_moves:
            # ⭐ NOVO: Bonus de segurança baseado no CommunicationHub
            safety_bonus = 0
            
            if environment.communication_hub.is_cell_safe(move):
                safety_bonus = 3.0  # Muito seguro (comunicação confirma)
            elif environment.communication_hub.is_bomb_active(move):
                safety_bonus = -5.0  # Perigoso (evitar!)
            elif environment.communication_hub.is_cell_explored(move):
                safety_bonus = 1.0  # Explorada mas não confirmada como segura
            
            # Penalizar revisitar células já exploradas (exceto se são seguras)
            if environment.explored[move[0], move[1]] and not environment.communication_hub.is_cell_safe(move):
                safety_bonus -= 1.0
            
            obs = self.observe(environment, move[0], move[1]).reshape(1, -1)
            
            # Combinar predições dos modelos
            ensemble_score = safety_bonus
            total_weight = 0
            
            for model_name, model in self.models.items():
                if model.is_trained:
                    proba = model.predict_proba(obs)
                    if len(proba) > 0:
                        safe_prob = proba[0][1] if proba.shape[1] > 1 else proba[0][0]
                        ensemble_score += self.weights[model_name] * safe_prob
                        total_weight += self.weights[model_name]
            
            if total_weight > 0:
                ensemble_score /= total_weight
            
            move_scores.append(ensemble_score)
        
        # Escolher movimento com maior score
        best_idx = np.argmax(move_scores)
        return candidate_moves[best_idx]

    
    def move(self, environment, target_pos):
        """Move agente para nova posição com comunicação de informações de segurança"""
        if not self.alive or target_pos is None:
            return None
        
        self.position = target_pos
        self.steps_taken += 1
        
        # ⭐ NOVO: Verificar se célula já foi explorada
        already_explored = environment.explored[target_pos[0], target_pos[1]]
        
        # Marcar célula como explorada e obter resultado
        result = environment.mark_explored(*target_pos)
        
        # ⭐ NOVO: Comunicação explícita baseada no resultado
        if result == 'L':
            # Célula livre e segura
            if not already_explored:
                environment.communication_hub.register_cell_visit(self.id, target_pos, 'L')
            # Se já explorada, apenas confirmar que é segura (não envia nova mensagem)
            
        elif result == 'B':
            # Bomba encontrada
            if not already_explored:
                environment.communication_hub.register_cell_visit(self.id, target_pos, 'B')
            
            # Verificar se este agente pode desativar
            if self.has_treasure_protection:
                # Proteção do tesouro desativa a bomba PARA ESTE AGENTE
                self.has_treasure_protection = False
                self.bombs_activated += 1
                self.bombs_deactivated += 1
                self.deactivated_bombs.add(target_pos)  # Registra bomba desativada individualmente
                # Comunicar desativação
                environment.communication_hub.register_bomb_deactivation(self.id, target_pos)
            else:
                # Agente é destruído
                self.alive = False
                self.bombs_activated += 1
                
        elif result == 'T':
            # Tesouro encontrado
            self.treasures_found += 1
            self.has_treasure_protection = True
            environment.shared_knowledge[target_pos] = 'T_FOUND'
            if not already_explored:
                environment.communication_hub.register_cell_visit(self.id, target_pos, 'T')
            
        elif result == 'F':
            # Bandeira encontrada (abordagem C)
            if not already_explored:
                environment.communication_hub.register_cell_visit(self.id, target_pos, 'F')
        
        # ⭐ NOVO: Se já estava explorada, verificar se é uma bomba não desativada por este agente
        if already_explored:
            # Se é uma bomba que este agente não desativou, ele morre
            if result == 'B' and target_pos not in self.deactivated_bombs:
                self.alive = False
                self.bombs_activated += 1
            # Caso contrário, apenas atualizar conhecimento
        
        # Atualizar conhecimento próprio (ML)
        self.update_knowledge(environment, *target_pos, result)
        
        return result


class BaselineAgent(Agent):
    """Agente que usa algoritmos clássicos em vez de ML"""
    
    def __init__(self, agent_id, start_pos, approach):
        super().__init__(agent_id, start_pos)
        self.approach = approach
        self.models = {}  # Sem modelos ML
        self.weights = {}
        
        # Estado para algoritmos clássicos
        self.visited_cells = set([start_pos])
        self.exploration_queue = []
        
    def decide_next_move(self, environment):
        """Decide movimento usando algoritmo clássico apropriado"""
        if not self.alive:
            return None
        
        # Coletar movimentos possíveis
        possible_moves = []
        unexplored_moves = []
        
        for nx, ny in environment.get_neighbors(*self.position):
            if not environment.explored[nx, ny]:
                unexplored_moves.append((nx, ny))
            else:
                possible_moves.append((nx, ny))
        
        # Priorizar células não exploradas
        if unexplored_moves:
            possible_moves = unexplored_moves
        elif not possible_moves:
            return None
        
        # Aplicar estratégia baseada na abordagem
        if self.approach == 'A':
            return self._greedy_best_first_move(environment, possible_moves, unexplored_moves)
        elif self.approach == 'B':
            return self._bfs_move(environment, possible_moves, unexplored_moves)
        elif self.approach == 'C':
            return self._a_star_move(environment, possible_moves, unexplored_moves)
        
        return possible_moves[0]
    
    def _greedy_best_first_move(self, environment, possible_moves, unexplored_moves):
        """
        ABORDAGEM A - Greedy Best-First Search
        Heurística: Priorizar células com maior chance de ter tesouros
        Baseado em:
        1. Distância até regiões não exploradas
        2. Proximidade a tesouros já conhecidos
        3. Evitar áreas com muitas bombas conhecidas
        """
        if not possible_moves:
            return None
        
        best_move = None
        best_score = -float('inf')
        
        for move in possible_moves:
            score = 0
            
            # 1. BONUS: Células não exploradas (prioridade máxima)
            if move in unexplored_moves:
                score += 100
            
            # 2. HEURÍSTICA: Proximidade a tesouros conhecidos
            # Tesouros atraem o agente (podem indicar clusters)
            for (tx, ty), cell_type in environment.shared_knowledge.items():
                if cell_type in ['T', 'T_FOUND']:
                    distance = abs(move[0] - tx) + abs(move[1] - ty)
                    if distance > 0:
                        score += 50 / distance  # Mais próximo = maior score
            
            # 3. PENALIDADE: Proximidade a bombas conhecidas
            neighbors_with_bombs = 0
            for nx, ny in environment.get_neighbors(*move):
                if (nx, ny) in environment.shared_knowledge:
                    if environment.shared_knowledge[(nx, ny)] == 'B':
                        # Verificar se este agente desativou
                        if (nx, ny) not in self.deactivated_bombs:
                            neighbors_with_bombs += 1
            
            score -= neighbors_with_bombs * 30  # Penalizar áreas perigosas
            
            # 4. DIVERSIFICAÇÃO: Explorar longe de onde já esteve
            if move not in self.visited_cells:
                score += 20
            
            # 5. EXPLORAÇÃO: Preferir células mais distantes do centro já explorado
            # (evita ficar preso em áreas pequenas)
            center_x = sum(x for x, y in self.visited_cells) / max(len(self.visited_cells), 1)
            center_y = sum(y for x, y in self.visited_cells) / max(len(self.visited_cells), 1)
            distance_from_visited = abs(move[0] - center_x) + abs(move[1] - center_y)
            score += distance_from_visited * 5
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move:
            self.visited_cells.add(best_move)
        
        return best_move
    
    def _bfs_move(self, environment, possible_moves, unexplored_moves):
        """
        ABORDAGEM B - Breadth-First Search (BFS)
        Exploração sistemática e uniforme do ambiente
        Prioriza células não exploradas em ordem de descoberta
        """
        if not possible_moves:
            return None
        
        # BFS puro: priorizar células não exploradas na ordem
        if unexplored_moves:
            # Escolher a célula não explorada mais "próxima" na ordem BFS
            # (menor distância Manhattan da posição inicial)
            best_move = min(unexplored_moves, 
                          key=lambda pos: abs(pos[0] - 0) + abs(pos[1] - 0))
            return best_move
        
        # Se não há não exploradas, mover para qualquer adjacente segura conhecida
        safe_moves = []
        for move in possible_moves:
            if move in environment.shared_knowledge:
                cell_type = environment.shared_knowledge[move]
                if cell_type in ['L', 'T', 'F']:
                    safe_moves.append(move)
                elif cell_type == 'B' and move in self.deactivated_bombs:
                    safe_moves.append(move)
        
        if safe_moves:
            return safe_moves[0]
        
        return possible_moves[0]
    
    def _a_star_move(self, environment, possible_moves, unexplored_moves):
        """
        ABORDAGEM C - A* Search Adaptativo
        
        PROBLEMA CRÍTICO: A bandeira NÃO é conhecida previamente!
        
        SOLUÇÃO: Usar A* modificado:
        1. Se bandeira não foi encontrada → explorar sistematicamente (como BFS)
        2. Quando bandeira for DESCOBERTA → usar A* clássico para ir direto
        """
        # Verificar se já encontramos a bandeira
        flag_position = None
        for (x, y), cell_type in environment.shared_knowledge.items():
            if cell_type == 'F':
                flag_position = (x, y)
                break
        
        # CASO 1: Bandeira ainda NÃO foi descoberta
        if flag_position is None:
            # Usar estratégia de exploração sistemática
            # Similar a BFS, mas com preferência por células mais distantes
            if unexplored_moves:
                # Priorizar células mais distantes da origem (exploração agressiva)
                best_move = max(unexplored_moves,
                              key=lambda pos: abs(pos[0] - 0) + abs(pos[1] - 0))
                return best_move
            
            # Fallback: mover para qualquer célula disponível
            return possible_moves[0] if possible_moves else None
        
        # CASO 2: Bandeira JÁ foi descoberta → usar A* clássico
        def heuristic(pos):
            """Distância Manhattan até a bandeira"""
            return abs(pos[0] - flag_position[0]) + abs(pos[1] - flag_position[1])
        
        # Se já estamos na bandeira, missão cumprida
        if self.position == flag_position:
            return None
        
        # Escolher movimento que minimiza distância até a bandeira
        # considerando células seguras conhecidas
        best_move = None
        best_h = float('inf')
        
        for move in possible_moves:
            # Verificar segurança
            is_safe = True
            if move in environment.shared_knowledge:
                cell_type = environment.shared_knowledge[move]
                if cell_type == 'B' and move not in self.deactivated_bombs:
                    is_safe = False
            
            # Se não explorada ou segura, considerar
            if move in unexplored_moves or is_safe:
                h = heuristic(move)
                if h < best_h:
                    best_h = h
                    best_move = move
        
        # ✅ CORREÇÃO: Return final obrigatório

class Simulation:
    def __init__(self, approach='A', num_agents=2, bomb_ratio=0.5, 
                 group_type='homogeneous', seed=None):
        self.approach = approach
        self.num_agents = num_agents
        self.bomb_ratio = bomb_ratio
        self.group_type = group_type
        
        # Criar ambiente
        treasure_count = 10 if approach == 'A' else 0
        self.environment = Environment(bomb_ratio, treasure_count, approach, seed)
        
        # ⭐ NOVO: Marcar posição inicial como explorada
        self.environment.mark_explored(0, 0)
        
        # Estado da simulação
        self.collect_states = False
        self.states = []
        
        # Criar agentes
        self.agents = []
        self._create_agents()
        
        # ⭐ NOVO: Atribuir territórios para coordenação (exceto baseline)
        if self.group_type != 'baseline':
            self.environment.communication_hub.assign_territories(
                [a.id for a in self.agents],
                approach='grid'
            )
        
        # Métricas
        self.metrics = {
            'approach': approach,
            'group_type': group_type,
            'num_agents': num_agents,
            'bomb_ratio': bomb_ratio,
            'start_time': None,
            'end_time': None,
            'execution_time': 0,
            'total_steps': 0,
            'exploration_percentage': 0,
            'bombs_activated': 0,
            'agents_destroyed': 0,
            'agents_alive': 0,
            'treasures_found': 0,
            'success': False
        }


    def _get_current_state(self):
        """Retorna o estado atual do ambiente e agentes (para animação)"""
        return {
            'grid': [[str(cell) for cell in row] for row in self.environment.grid.tolist()],
            'explored': [[bool(cell) for cell in row] for row in self.environment.explored.tolist()],
            'agents': [
                {
                    'id': agent.id,
                    'position': agent.position,
                    'alive': agent.alive,
                    'treasures_found': agent.treasures_found,
                    'steps_taken': agent.steps_taken
                }
                for agent in self.agents
            ]
        }
    
    def _create_agents(self):
        """Cria agentes homogêneos, heterogêneos ou baseline - todos começam em (0,0)"""
        # Todos os agentes começam na posição (0, 0)
        start_position = (0, 0)
        
        for i in range(self.num_agents):
            if self.group_type == 'baseline':
                # Agente baseline usa algoritmo clássico
                agent = BaselineAgent(i, start_position, self.approach)
            elif self.group_type == 'homogeneous':
                # Todos com mesmos pesos
                weights = {'knn': 1/3, 'naive_bayes': 1/3, 'random_forest': 1/3}
                agent = Agent(i, start_position, weights)
            else:  # heterogeneous
                # Pesos diferentes para cada agente
                if i % 3 == 0:
                    weights = {'knn': 0.7, 'naive_bayes': 0.2, 'random_forest': 0.1}
                elif i % 3 == 1:
                    weights = {'knn': 0.1, 'naive_bayes': 0.7, 'random_forest': 0.2}
                else:
                    weights = {'knn': 0.2, 'naive_bayes': 0.1, 'random_forest': 0.7}
                agent = Agent(i, start_position, weights)
            
            self.agents.append(agent)
    
    def run(self, max_iterations=200, timeout_seconds=30):
        """Executa simulação com opção de coletar estados intermediários e timeout de segurança"""
        self.metrics['start_time'] = time.time()
        timeout_deadline = self.metrics['start_time'] + timeout_seconds
        
        # Estado inicial (se animação ativada)
        if self.collect_states:
            self.states.append(self._get_current_state())
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # Verificar timeout de segurança
            if time.time() > timeout_deadline:
                print(f"[TIMEOUT] Simulação atingiu limite de tempo ({timeout_seconds}s) na iteração {iteration}")
                break
            
            # Cada agente decide e move
            any_agent_moved = False
            for agent in self.agents:
                if agent.alive:
                    next_move = agent.decide_next_move(self.environment)
                    if next_move:
                        agent.move(self.environment, next_move)
                        any_agent_moved = True
            
            # Salva estado após cada rodada (para animação)
            if self.collect_states:
                self.states.append(self._get_current_state())
            
            # Verificar condições de parada
            if self._check_termination():
                break
            
            if not any_agent_moved:
                break
        
        self.metrics['end_time'] = time.time()
        self.metrics['iterations_executed'] = int(iteration)
        self._calculate_metrics()
        
        # Adiciona lista de estados no resultado final
        if self.collect_states:
            self.metrics['states'] = self.states
        
        return self.metrics
    
    def _check_termination(self):
        """Verifica se simulação deve terminar baseado na abordagem"""
        # A simulação termina quando todos os agentes estão mortos OU quando é forçada a parar
        all_agents_dead = all(not agent.alive for agent in self.agents)
        
        # Verificações específicas por abordagem
        if self.approach == 'A':
            # Abordagem A: Termina se encontrou 50% dos tesouros OU se todos agentes foram destruídos
            total_treasures = self.environment.count_treasures()
            treasures_found = sum(a.treasures_found for a in self.agents)
            if total_treasures > 0 and treasures_found >= total_treasures * 0.5:
                return True
            if all_agents_dead:
                return True
        
        elif self.approach == 'B':
            # Abordagem B: Termina se explorou 100% do ambiente
            if self.environment.get_exploration_percentage() >= 100:
                return True
            if all_agents_dead:
                return True
        
        elif self.approach == 'C':
            # Abordagem C: Termina se encontrou a bandeira
            for i in range(self.environment.size):
                for j in range(self.environment.size):
                    if self.environment.grid[i, j] == 'F' and self.environment.explored[i, j]:
                        return True
            if all_agents_dead:
                return True
        
        # Termina se todos agentes mortos
        return all_agents_dead
    
    def _calculate_metrics(self):
        """Calcula métricas finais"""
        self.metrics['execution_time'] = float(self.metrics['end_time'] - self.metrics['start_time'])
        self.metrics['total_steps'] = int(sum(a.steps_taken for a in self.agents))
        self.metrics['exploration_percentage'] = float(self.environment.get_exploration_percentage())
        self.metrics['bombs_activated'] = int(sum(a.bombs_activated for a in self.agents))
        self.metrics['bombs_deactivated'] = int(sum(a.bombs_deactivated for a in self.agents))
        self.metrics['agents_destroyed'] = int(sum(1 for a in self.agents if not a.alive))
        self.metrics['agents_alive'] = int(sum(1 for a in self.agents if a.alive))
        self.metrics['treasures_found'] = int(sum(a.treasures_found for a in self.agents))
        
        # Definir sucesso baseado na abordagem
        if self.approach == 'A':
            total_treasures = int(self.environment.count_treasures())
            # Sucesso: encontrou pelo menos 50% dos tesouros OU todos agentes foram destruídos
            treasures_threshold = total_treasures > 0 and self.metrics['treasures_found'] >= total_treasures * 0.5
            all_destroyed = self.metrics['agents_destroyed'] >= self.num_agents
            self.metrics['success'] = bool(treasures_threshold or all_destroyed)
            self.metrics['treasure_percentage'] = float((self.metrics['treasures_found'] / max(total_treasures, 1)) * 100)
        
        elif self.approach == 'B':
            self.metrics['success'] = bool((self.metrics['exploration_percentage'] >= 100 and 
                                      self.metrics['agents_alive'] >= 1))
        
        elif self.approach == 'C':
            flag_found = False
            for i in range(self.environment.size):
                for j in range(self.environment.size):
                    if self.environment.grid[i, j] == 'F' and self.environment.explored[i, j]:
                        flag_found = True
                        break
            self.metrics['success'] = bool(flag_found)
            self.metrics['flag_found'] = bool(flag_found)
        
        # ⭐ NOVO: Adicionar estatísticas de comunicação
        self.metrics['communication'] = self.environment.communication_hub.get_communication_stats()


def run_experiment(approach, num_agents, bomb_ratio, group_type, repetitions=30):
    results = []
    for rep in range(repetitions):
        sim = Simulation(approach, num_agents, bomb_ratio, group_type)
        metrics = sim.run()
        metrics['repetition'] = rep
        results.append(metrics)
    return pd.DataFrame(results)



# Exemplo de uso
if __name__ == "__main__":
    print("=== Sistema de Simulação de Agentes Inteligentes ===\n")
    
    # Teste rápido
    print("Executando simulação de teste (Abordagem A)...")
    sim = Simulation(approach='A', num_agents=3, bomb_ratio=0.5, group_type='homogeneous', seed=42)
    metrics = sim.run()
    
    print(f"\nResultados:")
    print(f"  Sucesso: {metrics['success']}")
    print(f"  Tempo: {metrics['execution_time']:.3f}s")
    print(f"  Passos: {metrics['total_steps']}")
    print(f"  Exploração: {metrics['exploration_percentage']:.1f}%")
    print(f"  Tesouros: {metrics['treasures_found']}")
    print(f"  Agentes vivos: {metrics['agents_alive']}/{metrics['num_agents']}")
