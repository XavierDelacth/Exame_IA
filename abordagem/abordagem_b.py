# abordagem_b.py 

import numpy as np
import random
import time
import warnings
from collections import defaultdict, deque
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================
# 1. AMBIENTE PARA ABORDAGEM B (SEM TESOUROS)
# ============================================

class EnvironmentB:
    def __init__(self, size=10, bomb_ratio=0.3):
        self.size = size
        self.grid = np.empty((size, size), dtype=object)
        self.original_grid = None
        # Garantir que bomb_ratio esteja entre 20% e 80%
        self.bomb_ratio = max(0.2, min(0.8, bomb_ratio))
        self.treasure_count = 0  # SEM TESOUROS na abordagem B
        self.generate_environment()
        
    def _bfs_reachable(self, grid, start):
        """Calcula todas as cÃƒÂ©lulas livres alcanÃƒÂ§ÃƒÂ¡veis a partir de start via BFS."""
        visited = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if (nx, ny) not in visited and grid[nx, ny] != 'B':
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return visited

    def generate_environment(self):
        """Gera ambiente SEM tesouros, apenas bombas e cÃƒÂ©lulas livres.
        
        CORREÃƒâ€¡ÃƒÆ’O CRÃƒÂTICA: 
        1. Garante que (0,0) tem pelo menos 2 vizinhos livres (nÃƒÂ£o fica encurralado)
        2. Garante que pelo menos 50% das cÃƒÂ©lulas livres sÃƒÂ£o alcanÃƒÂ§ÃƒÂ¡veis a partir de (0,0)
        3. Se o ambiente gerado nÃƒÂ£o ÃƒÂ© resolÃƒÂºvel, regenera (mÃƒÂ¡x 50 tentativas)
        """
        total_cells = self.size * self.size
        
        for attempt in range(50):
            all_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
            
            # Proteger (0,0) e garantir pelo menos 2 dos seus vizinhos livres
            protected = {(0, 0)}
            start_neighbors = [(0, 1), (1, 0)]
            protected.update(start_neighbors)
            
            remaining_positions = [pos for pos in all_positions if pos not in protected]
            
            # Calcular nÃƒÂºmero de bombas (descontar as protegidas)
            bomb_count = int(total_cells * self.bomb_ratio)
            bomb_count = min(bomb_count, len(remaining_positions))
            bomb_positions = set(random.sample(remaining_positions, bomb_count))
            
            # Inicializar grid
            for i in range(self.size):
                for j in range(self.size):
                    pos = (i, j)
                    if pos in bomb_positions:
                        self.grid[i, j] = 'B'
                    else:
                        self.grid[i, j] = 'L'
            
            # Verificar resolubilidade: pelo menos 50% das cÃƒÂ©lulas livres sÃƒÂ£o alcanÃƒÂ§ÃƒÂ¡veis
            total_free = sum(1 for i in range(self.size) for j in range(self.size) if self.grid[i, j] != 'B')
            reachable = self._bfs_reachable(self.grid, (0, 0))
            reachable_free = len(reachable)
            
            if reachable_free >= total_free * 0.5 and reachable_free >= 10:
                # Ambiente vÃƒÂ¡lido
                break
        
        self.original_grid = self.grid.copy()
        
    def get_cell(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[x, y]
        return None
    
    def get_neighbors(self, x, y):
        """Retorna vizinhas vÃƒÂ¡lidas (apenas horizontal/vertical)"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Apenas 4 direÃƒÂ§ÃƒÂµes
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors

# ============================================
# 2. MEMÃƒâ€œRIA COMPARTILHADA PARA ABORDAGEM B
# ============================================

class SharedMemoryB:
    def __init__(self, env_size=10):
        self.explored = set()
        self.bombs_found = set()
        self.agent_positions = {}
        self.agent_status = {}
        self.cell_knowledge = {}
        self.env_size = env_size
        
        # Compatibilidade com GUI (abordagem B nÃƒÂ£o tem tesouros)
        self.treasures_found = set()
        self.treasures_collected = set()
        
        # NOVO: DicionÃƒÂ¡rio de cÃƒÂ©lulas reservadas (pos -> agent_id) para evitar colisÃƒÂµes
        self.reserved_cells = {}
        
        # Inicializar conhecimento
        for i in range(env_size):
            for j in range(env_size):
                self.cell_knowledge[(i, j)] = {
                    'type': 'U',  # U = Unknown
                    'explored': False,
                    'safe': False  # Inicialmente desconhecido (nÃƒÂ£o seguro)
                }
        
        # Marcar posiÃƒÂ§ÃƒÂ£o inicial (0,0) como explorada
        self.explored.add((0, 0))
        self.cell_knowledge[(0, 0)]['explored'] = True
        self.cell_knowledge[(0, 0)]['type'] = 'L'
        self.cell_knowledge[(0, 0)]['safe'] = True
        
        # Marcar vizinhas imediatas de (0,0) como seguras tambÃƒÂ©m
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = 0 + dx, 0 + dy
            if 0 <= nx < env_size and 0 <= ny < env_size:
                self.cell_knowledge[(nx, ny)]['safe'] = True
    
    def is_known_bomb(self, position):
        """Verifica se posiÃƒÂ§ÃƒÂ£o ÃƒÂ© uma bomba jÃƒÂ¡ conhecida"""
        return position in self.bombs_found
    
    def update_explored(self, position, content, agent_id, env):
        """Atualiza memÃƒÂ³ria com nova exploraÃƒÂ§ÃƒÂ£o"""
        x, y = position
        self.explored.add(position)
        self.agent_positions[agent_id] = position
        
        # Atualizar conhecimento da cÃƒÂ©lula
        self.cell_knowledge[position]['explored'] = True
        self.cell_knowledge[position]['type'] = content
        
        log_msg = f"Agente {agent_id}: {content} em {position}"
        
        if content == 'B':
            self.bombs_found.add(position)
            self.cell_knowledge[position]['safe'] = False
            log_msg += " (BOMBA)"
        else:
            self.cell_knowledge[position]['safe'] = True
            log_msg += " (seguro)"
            
            # IMPORTANTE: Marcar vizinhas nÃƒÂ£o exploradas como seguras
            neighbors = env.get_neighbors(x, y)
            for neighbor in neighbors:
                if not self.cell_knowledge[neighbor]['explored']:
                    if neighbor not in self.bombs_found:
                        self.cell_knowledge[neighbor]['safe'] = True
            
        return log_msg
    
    def is_safe_cell(self, position):
        """Verifica se cÃƒÂ©lula ÃƒÂ© segura para visitar"""
        return self.cell_knowledge[position]['safe']
    
    def get_best_unknown_neighbor(self, position, env):
        """Retorna melhor vizinha nÃƒÂ£o explorada"""
        x, y = position
        neighbors = env.get_neighbors(x, y)
        
        # Filtrar apenas cÃƒÂ©lulas seguras e nÃƒÂ£o exploradas
        safe_unknown = []
        for pos in neighbors:
            if not self.cell_knowledge[pos]['explored'] and self.cell_knowledge[pos]['safe']:
                safe_unknown.append(pos)
        
        return safe_unknown
    
    def reserve_cell(self, position, agent_id):
        """Reserva uma cÃƒÂ©lula para um agente especÃƒÂ­fico"""
        self.reserved_cells[position] = agent_id
    
    def clear_reservations(self):
        """Limpa as reservas no inÃƒÂ­cio de cada passo"""
        self.reserved_cells.clear()
    
    def is_reserved_by_other(self, position, agent_id):
        """Verifica se uma cÃƒÂ©lula estÃƒÂ¡ reservada por OUTRO agente (nÃƒÂ£o pelo prÃƒÂ³prio)"""
        if position in self.reserved_cells:
            return self.reserved_cells[position] != agent_id
        return False

# ============================================
# 3. AGENTE BFS PARA BASELINE (N AGENTES) - CORRIGIDO ANTI-LOOP
# ============================================

class AgentBFS:
    """
    Agente BFS PURO para baseline
    - Executa APENAS BFS (sem ML, sem inferÃƒÂªncia)
    - Todos os agentes usam a MESMA polÃƒÂ­tica
    - ColaboraÃƒÂ§ÃƒÂ£o apenas via memÃƒÂ³ria compartilhada
    """
    def __init__(self, agent_id, start_pos=(0, 0)):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.steps_taken = 0
        self.treasures_collected = 0  # Sempre 0 para baseline B
        self.bombs_defused = 0
        
        # BFS state
        self.exploration_queue = deque([start_pos])
        self.personal_visited = set([start_pos])
        
        # DireÃƒÂ§ÃƒÂ£o preferencial baseada no ID (para dividir espaÃƒÂ§o)
        self.preferred_direction = agent_id % 4  # 0=cima, 1=baixo, 2=esquerda, 3=direita
        
        # CORREÃƒâ€¡ÃƒÆ’O CRÃƒÂTICA: HistÃƒÂ³rico de posiÃƒÂ§ÃƒÂµes para detecÃƒÂ§ÃƒÂ£o de loop
        self.position_history = deque(maxlen=20)
        self.stuck_counter = 0
        
    def choose_action(self, shared_memory, env):
        """
        BFS PURO: escolhe prÃƒÂ³xima cÃƒÂ©lula da fila
        SEM pesos, SEM ML, SEM decisÃƒÂ£o inteligente.
        
        CORREÃƒâ€¡ÃƒÆ’O: Adicionada detecÃƒÂ§ÃƒÂ£o de loop e movimento de escape.
        """
        if not self.alive:
            return None

        # Registrar posiÃƒÂ§ÃƒÂ£o atual no histÃƒÂ³rico
        self.position_history.append(self.position)
        
        # DETECÃƒâ€¡ÃƒÆ’O DE LOOP: Verificar oscilaÃƒÂ§ÃƒÂ£o entre 2 posiÃƒÂ§ÃƒÂµes (padrÃƒÂ£o A-B-A-B-A-B)
        if len(self.position_history) >= 6:
            hist = list(self.position_history)
            if (hist[-1] == hist[-3] == hist[-5] and 
                hist[-2] == hist[-4] == hist[-6]):
                self.stuck_counter += 1
                # ForÃƒÂ§ar movimento de escape
                return self._escape_move(shared_memory, env)
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)

        x, y = self.position
        neighbors = env.get_neighbors(x, y)

        # 1. Prioridade: vizinhos IMEDIATOS nÃƒÂ£o explorados, nÃƒÂ£o bomba, NÃƒÆ’O reservados por outros
        unexplored = [n for n in neighbors
                      if n not in shared_memory.explored
                      and not shared_memory.is_known_bomb(n)
                      and not shared_memory.is_reserved_by_other(n, self.id)]

        if unexplored:
            unexplored.sort(key=lambda pos: self._direction_priority(pos))
            chosen = unexplored[0]
            shared_memory.reserve_cell(chosen, self.id)
            return chosen

        # 2. Se preso, tentar qualquer vizinho seguro nÃƒÂ£o reservado por outros (mesmo explorado)
        safe_neighbors = [n for n in neighbors
                        if not shared_memory.is_known_bomb(n)
                        and not shared_memory.is_reserved_by_other(n, self.id)]
        
        # Priorizar os que tÃƒÂªm vizinhos nÃƒÂ£o explorados
        for candidate in safe_neighbors:
            candidate_neighbors = env.get_neighbors(*candidate)
            has_unexplored = any(cn not in shared_memory.explored
                               and not shared_memory.is_known_bomb(cn)
                               for cn in candidate_neighbors)
            if has_unexplored:
                shared_memory.reserve_cell(candidate, self.id)
                return candidate

        # 3. Se ainda nÃƒÂ£o encontrou, tentar BFS atÃƒÂ© target prÃƒÂ³ximo
        nearby = self._find_nearby_unexplored(shared_memory, env, radius=5)
        if nearby:
            target = nearby[0]
            path = self._find_safe_path(self.position, target, shared_memory, env)
            if path and len(path) > 1:
                next_step = path[1]
                if not shared_memory.is_known_bomb(next_step) and not shared_memory.is_reserved_by_other(next_step, self.id):
                    shared_memory.reserve_cell(next_step, self.id)
                    return next_step

        # 4. BFS da fila
        while self.exploration_queue:
            candidate = self.exploration_queue.popleft()
            if (candidate not in shared_memory.explored 
                and not shared_memory.is_known_bomb(candidate)
                and not shared_memory.is_reserved_by_other(candidate, self.id)):
                path = self._find_safe_path(self.position, candidate, shared_memory, env)
                if path and len(path) > 1:
                    next_step = path[1]
                    if next_step in neighbors and not shared_memory.is_reserved_by_other(next_step, self.id):
                        shared_memory.reserve_cell(next_step, self.id)
                        return next_step

        # 5. Expandir fila e tentar de novo
        self._expand_queue(env, shared_memory)
        if self.exploration_queue:
            candidate = self.exploration_queue.popleft()
            if (not shared_memory.is_known_bomb(candidate) 
                and not shared_memory.is_reserved_by_other(candidate, self.id)):
                shared_memory.reserve_cell(candidate, self.id)
                return candidate

        # 6. CORREÃƒâ€¡ÃƒÆ’O: Se tudo falhou, usar movimento de escape inteligente
        return self._escape_move(shared_memory, env)
    
    def _escape_move(self, shared_memory, env):
        """
        Movimento de escape quando o agente estÃƒÂ¡ preso em loop.
        Escolhe a cÃƒÂ©lula que:
        - Foi menos visitada recentemente
        - Tem mais vizinhos nÃƒÂ£o explorados
        - EstÃƒÂ¡ mais longe de outros agentes
        - NÃƒÆ’O ÃƒÂ© a posiÃƒÂ§ÃƒÂ£o anterior (evita oscilaÃƒÂ§ÃƒÂ£o)
        """
        x, y = self.position
        neighbors = env.get_neighbors(x, y)
        
        # Filtrar apenas cÃƒÂ©lulas seguras
        safe_neighbors = [n for n in neighbors 
                         if not shared_memory.is_known_bomb(n)]
        
        if not safe_neighbors:
            return self.position  # Ficar parado se nÃƒÂ£o hÃƒÂ¡ opÃƒÂ§ÃƒÂµes
        
        # Calcular score para cada vizinho
        best_score = -float('inf')
        best_move = safe_neighbors[0]
        
        for neighbor in safe_neighbors:
            score = 0
            
            # Penalizar cÃƒÂ©lulas recentemente visitadas (peso ALTO)
            visit_count = sum(1 for pos in self.position_history if pos == neighbor)
            score -= visit_count * 100
            
            # Bonus por ter vizinhos nÃƒÂ£o explorados
            nn = env.get_neighbors(*neighbor)
            unexplored_neighbors = sum(1 for n in nn 
                                      if n not in shared_memory.explored 
                                      and not shared_memory.is_known_bomb(n))
            score += unexplored_neighbors * 50
            
            # Bonus por distÃƒÂ¢ncia de outros agentes (evitar aglomeraÃƒÂ§ÃƒÂ£o)
            for other_id, other_pos in shared_memory.agent_positions.items():
                if other_id != self.id and other_pos is not None:
                    dist = abs(neighbor[0] - other_pos[0]) + abs(neighbor[1] - other_pos[1])
                    score += dist * 5
            
            # Penalizar VOLTAR para a posiÃƒÂ§ÃƒÂ£o anterior (quebrar oscilaÃƒÂ§ÃƒÂ£o) - peso MUITO ALTO
            if len(self.position_history) >= 1:
                if neighbor == self.position_history[-1]:
                    score -= 200
            
            if score > best_score:
                best_score = score
                best_move = neighbor
        
        shared_memory.reserve_cell(best_move, self.id)
        return best_move
    
    def _distance_to_other_agents(self, pos, shared_memory):
        """Calcula distÃƒÂ¢ncia mÃƒÂ­nima para outros agentes"""
        min_dist = float('inf')
        for other_id, other_pos in shared_memory.agent_positions.items():
            if other_id != self.id and other_pos is not None:
                dist = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
                min_dist = min(min_dist, dist)
        return min_dist if min_dist != float('inf') else 100
    
    def _find_safe_path(self, start, goal, shared_memory, env):
        """BFS para encontrar caminho seguro (evitando bombas conhecidas)"""
        if start == goal:
            return [start]
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            current, path = queue.popleft()
            for neighbor in env.get_neighbors(*current):
                if neighbor not in visited and not shared_memory.is_known_bomb(neighbor):
                    new_path = path + [neighbor]
                    if neighbor == goal:
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        return None
    
    def _direction_priority(self, neighbor):
        """Calcula prioridade baseada na direÃ§Ã£o preferencial"""
        nx, ny = neighbor
        cx, cy = self.position
        dx, dy = nx - cx, ny - cy
        
        # Mapear direÃ§Ã£o
        if dx == -1:
            direction = 0  # cima
        elif dx == 1:
            direction = 1  # baixo
        elif dy == -1:
            direction = 2  # esquerda
        else:
            direction = 3  # direita
        
        # âœ… CORREÃ‡ÃƒO: Desempate Ãºnico por agente
        # Evita que agentes com mesma preferred_direction escolham a mesma cÃ©lula
        tiebreaker = (hash(neighbor) + self.id * 17) % 100 / 1000.0
        
        # Prioridade: menor valor = maior prioridade
        if direction == self.preferred_direction:
            return 0 + tiebreaker
        return abs(direction - self.preferred_direction) + tiebreaker
    
    
    def _find_nearby_unexplored(self, shared_memory, env, radius=5):
        """Encontra cÃƒÂ©lulas nÃƒÂ£o exploradas prÃƒÂ³ximas, excluindo bombas conhecidas"""
        x, y = self.position
        candidates = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < env.size and 0 <= ny < env.size:
                    pos = (nx, ny)
                    if (pos not in shared_memory.explored 
                        and not shared_memory.is_known_bomb(pos)
                        and not shared_memory.is_reserved_by_other(pos, self.id)):
                        distance = abs(dx) + abs(dy)
                        candidates.append((distance, pos))
        
        candidates.sort(key=lambda c: c[0])
        return [pos for _, pos in candidates]
    
    def _expand_queue(self, env, shared_memory):
        """Expande fila de exploraÃƒÂ§ÃƒÂ£o"""
        x, y = self.position
        neighbors = env.get_neighbors(x, y)
        
        for neighbor in neighbors:
            if neighbor not in self.personal_visited:
                self.personal_visited.add(neighbor)
                self.exploration_queue.append(neighbor)
    
    def move_to(self, new_position, shared_memory, env):
        """Move agente para nova posiÃƒÂ§ÃƒÂ£o"""
        if not self.alive:
            return "Agente inativo"
        
        # CORREÃƒâ€¡ÃƒÆ’O: Se new_position ÃƒÂ© None ou igual ÃƒÂ  posiÃƒÂ§ÃƒÂ£o atual, apenas logar
        if new_position is None:
            return f"Agente {self.id}: sem movimento vÃƒÂ¡lido"
        
        # Se ficou parado, nÃƒÂ£o incrementar steps_taken mas logar
        if new_position == self.position:
            return f"Agente {self.id}: aguardando em {self.position}"
        
        self.position = new_position
        self.steps_taken += 1
        
        x, y = new_position
        cell_content = env.get_cell(x, y)
        log_msg = shared_memory.update_explored(new_position, cell_content, self.id, env)
        
        # Adicionar vizinhos ÃƒÂ  fila
        neighbors = env.get_neighbors(x, y)
        for neighbor in neighbors:
            if neighbor not in self.personal_visited:
                self.personal_visited.add(neighbor)
                self.exploration_queue.append(neighbor)
        
        # ConsequÃƒÂªncias
        if cell_content == 'B':
            if self.bombs_defused > 0:
                self.bombs_defused -= 1
                log_msg += " | BOMBA DESATIVADA!"
                shared_memory.cell_knowledge[new_position]['safe'] = True
            else:
                self.alive = False
                log_msg += " | AGENTE DESTRUÃƒÂDO"
        
        return log_msg
    
    def train_models(self, shared_memory=None, env=None):
        """MÃƒÂ©todo vazio - compatibilidade com GUI"""
        pass

# ============================================
# 4. AGENTE ML PARA GRUPOS (HOMOGÃƒÅ NEO/HETEROGÃƒÅ NEO) - CORRIGIDO ANTI-LOOP
# ============================================

class AgentB:
    """Agente com ML para grupos homogÃƒÂªneo/heterogÃƒÂªneo"""
    def __init__(self, agent_id, start_pos=(0, 0), inference_weights=None, model_choice=None):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.treasures_collected = 0
        self.bombs_defused = 0
        self.steps_taken = 0
        self.last_action = None
        
        # âœ… CORREÃ‡ÃƒO: DireÃ§Ã£o preferencial baseada no ID (diversificar exploraÃ§Ã£o)
        self.preferred_direction = agent_id % 4  # 0=Norte, 1=Sul, 2=Leste, 3=Oeste
        
        # Inicializar modelos de ML
        self.models = {
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'NaiveBayes': GaussianNB(),
            'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        # Pesos dos modelos (cada agente tem todos os modelos, mas com prioridades)
        self.model_weights = inference_weights or {'KNN': 1/3, 'NaiveBayes': 1/3, 'RandomForest': 1/3}
        self.preferred_model = model_choice
        
        # Dados para treinamento
        self.training_data = {'features': [], 'labels': []}
        self.models_trained = False
        
        # Motor de inferÃƒÂªncia
        self.inference_engine = InferenceEngineB(inference_weights)
        
        # HistÃƒÂ³rico de aÃƒÂ§ÃƒÂµes
        self.action_history = deque(maxlen=50)
        
        # CORREÃƒâ€¡ÃƒÆ’O CRÃƒÂTICA: HistÃƒÂ³rico de posiÃƒÂ§ÃƒÂµes para detecÃƒÂ§ÃƒÂ£o de loop
        self.position_history = deque(maxlen=20)
        self.stuck_counter = 0
        
        # MemÃƒÂ³ria individual do agente (para baseline)
        self.memory = SharedMemoryB()
        
        # Inicializar com algum conhecimento bÃƒÂ¡sico
        self.initialize_basic_knowledge()
    
    def initialize_basic_knowledge(self):
        """Inicializa conhecimento bÃƒÂ¡sico para evitar previsÃƒÂµes sem dados."""
        self.training_data['features'].append([0, 0])
        self.training_data['labels'].append('L')
        
        # Gerar 11 amostras sintÃƒÂ©ticas (total = 12, acima do limiar de 10)
        for _ in range(11):
            x, y = random.randint(0, 9), random.randint(0, 9)
            self.training_data['features'].append([x, y])
            self.training_data['labels'].append(random.choice(['L', 'L', 'L', 'B']))
    
    def train_models(self, shared_memory=None, env=None):
        """Treina modelos com dados coletados"""
        if len(self.training_data['features']) >= 10:
            X = np.array(self.training_data['features'])
            y = np.array(self.training_data['labels'])
            
            try:
                # Treinar todos os modelos (cada agente contÃƒÂ©m todos)
                for name, model in self.models.items():
                    model.fit(X, y)
                self.models_trained = True
            except Exception:
                pass
    
    def predict_cell(self, cell_position, shared_memory):
        """Preve o tipo de cÃƒÂ©lula usando modelos ou heurÃƒÂ­stica"""
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        x, y = cell_position
        
        # Se cÃƒÂ©lula jÃƒÂ¡ foi explorada, retorna conteÃƒÂºdo conhecido
        if cell_position in memory.explored:
            if cell_position in memory.bombs_found:
                return 'B'
            else:
                return 'L'
        
        # Se modelos estÃƒÂ£o treinados, usar previsÃƒÂ£o com votaÃƒÂ§ÃƒÂ£o ponderada de probabilidades
        if self.models_trained and len(self.training_data['features']) >= 10:
            from collections import defaultdict

            probs = defaultdict(float)

            for name, model in self.models.items():
                weight = float(self.model_weights.get(name, 0.0))
                if weight <= 0:
                    continue
                try:
                    proba = model.predict_proba([[x, y]])[0]
                    classes = list(model.classes_)
                    for cls, p in zip(classes, proba):
                        probs[cls] += weight * p
                except Exception:
                    try:
                        pred = model.predict([[x, y]])[0]
                        probs[pred] += weight * 1.0
                    except Exception:
                        continue

            if probs:
                best = max(probs.items(), key=lambda kv: kv[1])[0]
                return best
        
        # HeurÃƒÂ­stica baseada na posiÃƒÂ§ÃƒÂ£o
        distance_to_center = np.sqrt((x - 5)**2 + (y - 5)**2)
        if distance_to_center < 3:
            return random.choice(['L', 'L', 'L'])  # Centro mais seguro
        else:
            return random.choice(['L', 'L', 'B', 'L'])  # Bordas mais perigosas
    
    def choose_action(self, shared_memory, env):
        """
        Escolhe prÃƒÂ³xima aÃƒÂ§ÃƒÂ£o com estratÃƒÂ©gia melhorada e anti-loop.
        
        CORREÃƒâ€¡Ãƒâ€¢ES CRÃƒÂTICAS:
        1. DetecÃƒÂ§ÃƒÂ£o de padrÃƒÂ£o A-B-A-B (loop entre 2 cÃƒÂ©lulas)
        2. Movimento de escape quando preso
        3. SimplificaÃƒÂ§ÃƒÂ£o da lÃƒÂ³gica de decisÃƒÂ£o
        """
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        if not self.alive:
            return None
        
        # Registrar posiÃƒÂ§ÃƒÂ£o no histÃƒÂ³rico
        self.position_history.append(self.position)
        
        # DETECÃƒâ€¡ÃƒÆ’O DE LOOP: Verificar padrÃƒÂ£o A-B-A-B (oscilaÃƒÂ§ÃƒÂ£o entre 2 posiÃƒÂ§ÃƒÂµes)
        if len(self.position_history) >= 6:
            hist = list(self.position_history)
            if (hist[-1] == hist[-3] == hist[-5] and 
                hist[-2] == hist[-4] == hist[-6]):
                self.stuck_counter += 1
                # ForÃƒÂ§ar movimento de escape
                return self._escape_move(memory, env)
            else:
                self.stuck_counter = max(0, self.stuck_counter - 1)
        
        x, y = self.position
        
        # Treinar modelos periodicamente
        if self.steps_taken % 5 == 0:
            self.train_models()
        
        # Obter vizinhas (agentes NÃƒÆ’O sabem onde estÃƒÂ£o bombas Ã¢â‚¬â€ descobrem ao pisar)
        neighbors = env.get_neighbors(x, y)
        
        # Filtrar vizinhos seguros (nÃƒÂ£o bomba)
        non_bomb_neighbors = [n for n in neighbors 
                             if not memory.is_known_bomb(n)]
        
        if not non_bomb_neighbors:
            return self.position  # Ficar parado se cercado por bombas
        
        # 1) Prioridade mÃƒÂ¡xima: vizinhas seguras NÃƒÆ’O exploradas
        unexplored = [n for n in non_bomb_neighbors 
                      if not memory.cell_knowledge[n]['explored']]
        
        if unexplored:
            # âœ… CORREÃ‡ÃƒO: Ordenar por mÃºltiplos critÃ©rios
            # Prioridade 1: DireÃ§Ã£o preferencial (mais importante)
            # Prioridade 2: DistÃ¢ncia de outros agentes
            unexplored.sort(key=lambda pos: (
                self._direction_score(pos),  # Menor = melhor
                -self._distance_to_other_agents(pos, memory),  # Maior = melhor (negativo)
            ))
            chosen = unexplored[0]
            
            # Verificar se jÃ¡ estÃ¡ reservado por outro agente
            if not memory.is_reserved_by_other(chosen, self.id):
                memory.reserve_cell(chosen, self.id)
                return chosen
            else:
                # Se a primeira escolha estÃ¡ reservada, tentar a prÃ³xima
                for candidate in unexplored[1:]:
                    if not memory.is_reserved_by_other(candidate, self.id):
                        memory.reserve_cell(candidate, self.id)
                        return candidate
                # Se todas estÃ£o reservadas, escolher a primeira mesmo assim
                memory.reserve_cell(unexplored[0], self.id)
                return unexplored[0]
        
        
        # 2) Todos os vizinhos sÃƒÂ£o explorados - usar BFS global para encontrar cÃƒÂ©lula nÃƒÂ£o explorada
        target = self._find_global_unexplored(memory, env)
        if target:
            path = self._find_safe_path(self.position, target, memory, env)
            if path and len(path) > 1:
                next_step = path[1]
                if next_step in non_bomb_neighbors:
                    memory.reserve_cell(next_step, self.id)
                    return next_step
        
        # 3) Tentar backtrack
        back = self._backtrack(env, memory)
        if back and back != self.position:
            memory.reserve_cell(back, self.id)
            return back
        
        # 4) ÃƒÅ¡LTIMO RECURSO: Movimento de escape inteligente
        return self._escape_move(memory, env)
    
    def _escape_move(self, memory, env):
        """
        Movimento de escape quando o agente estÃƒÂ¡ preso.
        Prioriza cÃƒÂ©lulas que:
        - NÃƒÂ£o foram visitadas recentemente
        - Levam a ÃƒÂ¡reas nÃƒÂ£o exploradas
        - EstÃƒÂ£o longe de outros agentes
        - NÃƒÆ’O sÃƒÂ£o a posiÃƒÂ§ÃƒÂ£o anterior (evita oscilaÃƒÂ§ÃƒÂ£o)
        """
        x, y = self.position
        neighbors = env.get_neighbors(x, y)
        
        # Filtrar apenas cÃƒÂ©lulas seguras
        safe_neighbors = [n for n in neighbors 
                         if not memory.is_known_bomb(n)]
        
        if not safe_neighbors:
            return self.position
        
        # Calcular score para cada vizinho
        best_score = -float('inf')
        best_move = safe_neighbors[0]
        
        for neighbor in safe_neighbors:
            score = 0
            
            # Penalizar cÃƒÂ©lulas recentemente visitadas (peso ALTO)
            visit_count = sum(1 for pos in self.position_history if pos == neighbor)
            score -= visit_count * 100
            
            # Bonus por ter vizinhos nÃƒÂ£o explorados
            nn = env.get_neighbors(*neighbor)
            unexplored_count = sum(1 for n in nn 
                                  if not memory.cell_knowledge[n]['explored'] 
                                  and not memory.is_known_bomb(n))
            score += unexplored_count * 50
            
            # Bonus por distÃƒÂ¢ncia de outros agentes
            score += self._distance_to_other_agents(neighbor, memory) * 5
            
            # Penalizar VOLTAR para a posiÃƒÂ§ÃƒÂ£o anterior (quebrar oscilaÃƒÂ§ÃƒÂ£o) - peso MUITO ALTO
            if len(self.position_history) >= 1:
                if neighbor == self.position_history[-1]:
                    score -= 200
            
            # Bonus por estar longe da posiÃƒÂ§ÃƒÂ£o atual (explorar)
            dist_from_current = abs(neighbor[0] - x) + abs(neighbor[1] - y)
            score += dist_from_current * 10
            
            if score > best_score:
                best_score = score
                best_move = neighbor
        
        memory.reserve_cell(best_move, self.id)
        return best_move
    
    def _distance_to_other_agents(self, pos, shared_memory):
        """Calcula distÃƒÂ¢ncia mÃƒÂ­nima para outros agentes"""
        min_dist = float('inf')
        for other_id, other_pos in shared_memory.agent_positions.items():
            if other_id != self.id and other_pos is not None:
                dist = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
                min_dist = min(min_dist, dist)
        return min_dist if min_dist != float('inf') else 100
    

    def _direction_score(self, pos):
        """
        Calcula score baseado na direÃ§Ã£o preferencial do agente.
        Menor score = direÃ§Ã£o preferida
        """
        dx = pos[0] - self.position[0]
        dy = pos[1] - self.position[1]
        
        # Mapear para direÃ§Ã£o (0-3)
        if dx < 0:
            direction = 0  # Norte (cima)
        elif dx > 0:
            direction = 1  # Sul (baixo)
        elif dy > 0:
            direction = 3  # Leste (direita)
        else:
            direction = 2  # Oeste (esquerda)
        
        # Calcular distÃ¢ncia circular da direÃ§Ã£o preferida
        diff = abs(direction - self.preferred_direction)
        circular_diff = min(diff, 4 - diff)
        
        # Adicionar pequeno componente aleatÃ³rio Ãºnico por agente para desempate
        tiebreaker = (hash(pos) + self.id * 17) % 100 / 1000.0
        
        return circular_diff + tiebreaker
    
    def _find_global_unexplored(self, memory, env):
        """Encontra a cÃƒÂ©lula inexplorada mais prÃƒÂ³xima via BFS global"""
        visited = {self.position}
        queue = deque([self.position])
        while queue:
            current = queue.popleft()
            for neighbor in env.get_neighbors(*current):
                if neighbor not in visited and not memory.is_known_bomb(neighbor):
                    if (not memory.cell_knowledge[neighbor]['explored'] 
                        and not memory.is_reserved_by_other(neighbor, self.id)):
                        return neighbor
                    visited.add(neighbor)
                    queue.append(neighbor)
        return None
    
    def _find_safe_path(self, start, goal, memory, env):
        """BFS para encontrar caminho seguro (evitando bombas conhecidas)"""
        if start == goal:
            return [start]
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            current, path = queue.popleft()
            for neighbor in env.get_neighbors(*current):
                if neighbor not in visited and not memory.is_known_bomb(neighbor):
                    new_path = path + [neighbor]
                    if neighbor == goal:
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        return None
    
    def _backtrack(self, env, memory):
        """
        BFS para encontrar caminho atÃƒÂ© cÃƒÂ©lula com vizinhos nÃƒÂ£o explorados.
        Limite de profundidade 20.
        """
        queue = deque([(self.position, [self.position])])
        visited = {self.position}
        while queue:
            current, path = queue.popleft()
            if len(path) > 20:
                continue
            neighbors = [v for v in env.get_neighbors(*current)
                         if v not in memory.bombs_found]
            has_unexplored = any(
                v not in memory.explored and v not in memory.bombs_found
                for v in neighbors
            )
            if has_unexplored and len(path) > 1:
                return path[1]
            for v in neighbors:
                if v not in visited and v not in memory.bombs_found:
                    visited.add(v)
                    queue.append((v, path + [v]))
        return None
    
    def move_to(self, new_position, shared_memory, env):
        """Move agente para nova posiÃƒÂ§ÃƒÂ£o"""
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        if not self.alive:
            return "Agente inativo"
        
        # CORREÃƒâ€¡ÃƒÆ’O: Se new_position ÃƒÂ© None, logar e retornar
        if new_position is None:
            return f"Agente {self.id}: sem movimento vÃƒÂ¡lido"
        
        # Se ficou parado, logar mas nÃƒÂ£o incrementar passos
        if new_position == self.position:
            return f"Agente {self.id}: aguardando em {self.position}"
        
        old_pos = self.position
        self.position = new_position
        self.steps_taken += 1
        self.action_history.append(old_pos)
        
        # Explorar nova cÃƒÂ©lula
        x, y = new_position
        cell_content = env.get_cell(x, y)
        
        # Atualizar memÃƒÂ³ria compartilhada
        if shared_memory is None:
            log_msg = memory.update_explored(new_position, cell_content, self.id, env)
        else:
            log_msg = shared_memory.update_explored(new_position, cell_content, self.id, env)
        
        # Para baseline, atualizar tambÃƒÂ©m memÃƒÂ³ria individual
        if getattr(self, 'is_baseline', False):
            memory.update_explored(new_position, cell_content, self.id, env)
        
        # Atualizar dados de treinamento
        self.training_data['features'].append([x, y])
        self.training_data['labels'].append(cell_content)
        
        # ConsequÃƒÂªncias da aÃƒÂ§ÃƒÂ£o
        if cell_content == 'B':
            if self.bombs_defused > 0:
                # Usa poder para desativar bomba
                self.bombs_defused -= 1
                log_msg += " | BOMBA DESATIVADA!"
                if shared_memory is not None:
                    shared_memory.cell_knowledge[new_position]['safe'] = True
                else:
                    memory.cell_knowledge[new_position]['safe'] = True
            else:
                self.alive = False
                log_msg += " | AGENTE DESTRUÃƒÂDO"
        
        return log_msg

# ============================================
# 5. MOTOR DE INFERÃƒÅ NCIA PARA GRUPOS ML
# ============================================

class InferenceEngineB:
    def __init__(self, weights=None):
        self.weights = weights or {
            'L': 3.0,    # Livre - prioridade alta
            'B': -200.0, # Bomba - evitar completamente
            'U': 5.0,    # Desconhecido - PRIORIZAR exploraÃƒÂ§ÃƒÂ£o
            'E': -1.0    # Explorado - evitar revisitar
        }
    
    def calculate_score(self, cell_type, position, shared_memory, agent):
        """Calcula pontuaÃƒÂ§ÃƒÂ£o considerando vÃƒÂ¡rios fatores."""
        base_score = self.weights.get(cell_type, 0.0)
        
        x, y = position
        
        # GRANDE bonus por exploraÃƒÂ§ÃƒÂ£o de ÃƒÂ¡rea nova
        if not shared_memory.cell_knowledge[position]['explored']:
            base_score += 10.0
            
            # Bonus extra: contar vizinhos tambÃƒÂ©m inexplorados (fronteira rica)
            neighbors = [(x+dx, y+dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]
                        if 0 <= x+dx < shared_memory.env_size and 0 <= y+dy < shared_memory.env_size]
            unexplored_neighbors = sum(1 for n in neighbors 
                                      if not shared_memory.cell_knowledge[n]['explored']
                                      and n not in shared_memory.bombs_found
                                      and not shared_memory.is_reserved_by_other(n, agent.id))
            base_score += unexplored_neighbors * 1.5
        else:
            # Penalidade por revisitar
            base_score -= 3.0
        
        # Penalidade MODERADA por estar perto de bomba
        for bx, by in shared_memory.bombs_found:
            distance = abs(bx - x) + abs(by - y)
            if distance == 1:  # Adjacente a bomba
                base_score -= 5.0
        
        # Penalidade por proximidade a outros agentes
        for other_id, other_pos in shared_memory.agent_positions.items():
            if other_id != agent.id and other_pos is not None:
                other_distance = abs(x - other_pos[0]) + abs(y - other_pos[1])
                if other_distance == 0:
                    base_score -= 10.0  # Mesma cÃƒÂ©lula
                elif other_distance <= 2:
                    base_score -= 3.0   # PrÃƒÂ³ximo
        
        # Penalidade extra se cÃƒÂ©lula estÃƒÂ¡ reservada por outro
        if shared_memory.is_reserved_by_other(position, agent.id):
            base_score -= 50.0
        
        return base_score
    
    def decide_action(self, available_cells, shared_memory, env, agent):
        """Decide aÃƒÂ§ÃƒÂ£o considerando mÃƒÂºltiplos fatores"""
        if shared_memory is None or getattr(agent, 'is_baseline', False):
            memory = agent.memory
        else:
            memory = shared_memory
            
        if not available_cells:
            return None
        
        best_score = -float('inf')
        best_action = None
        
        for pos, predicted_type in available_cells:
            # Verificar se ÃƒÂ© bomba conhecida
            if pos in memory.bombs_found and not memory.cell_knowledge[pos]['safe']:
                score = self.weights['B']
            else:
                score = self.calculate_score(predicted_type, pos, memory, agent)
            
            # Pequena aleatoriedade para evitar deadlocks
            score += random.uniform(-0.1, 0.1)
            
            if score > best_score:
                best_score = score
                best_action = pos
        
        return best_action

# ============================================
# 6. SIMULAÃƒâ€¡ÃƒÆ’O DA ABORDAGEM B
# ============================================

class ApproachBSimulation:
    def __init__(self, num_agents=4, bomb_ratio=0.3, homogeneous=True, max_steps=500):
        self.env = EnvironmentB(bomb_ratio=bomb_ratio)
        self.shared_memory = SharedMemoryB()
        self.num_agents = num_agents
        self.homogeneous = homogeneous
        self.max_steps = max_steps
        self.agents = []
        self.logs = []
        
        # Calcular cÃƒÂ©lulas livres
        total_cells = self.env.size * self.env.size
        bomb_count = sum(1 for row in self.env.grid for cell in row if cell == 'B')
        self.free_cells = total_cells - bomb_count
        
        self.metrics = {
            'explored_percentage': 0,
            'agents_alive': 0,
            'bombs_triggered': 0,
            'steps_taken': 0,
            'execution_time': 0,
            'success': False,
            'bomb_ratio': bomb_ratio,
            'treasures_found': 0,
            'total_treasures': 0,
            'explored_free_cells': 0
        }
        
        self.setup_agents()
    
    def setup_agents(self):
        """Criar agentes baseado no tipo (homogÃƒÂªneo, heterogÃƒÂªneo ou baseline)"""
        # Marcar posiÃƒÂ§ÃƒÂ£o inicial como explorada ANTES de criar agentes
        initial_pos = (0, 0)
        cell_content = self.env.get_cell(*initial_pos)
        self.shared_memory.explored.add(initial_pos)
        self.shared_memory.cell_knowledge[initial_pos]['explored'] = True
        self.shared_memory.cell_knowledge[initial_pos]['type'] = cell_content
        self.shared_memory.cell_knowledge[initial_pos]['safe'] = True
        
        # Criar agentes - AgentB para homogeneo/heterogeneo
        model_choices = []
        if self.homogeneous:
            # DistribuiÃƒÂ§ÃƒÂ£o homogÃƒÂ©nea aproximada
            base = self.num_agents // 3
            remainder = self.num_agents % 3
            counts = {'KNN': base, 'NaiveBayes': base, 'RandomForest': base}
            order = ['KNN', 'NaiveBayes', 'RandomForest']
            for i in range(remainder):
                counts[order[i]] += 1
            for model, cnt in counts.items():
                model_choices.extend([model] * cnt)
        else:
            # HeterogÃƒÂ©neo por agente
            order = ['KNN', 'RandomForest', 'NaiveBayes']
            for i in range(self.num_agents):
                model_choices.append(order[i % len(order)])

        # Ajustar lista
        if len(model_choices) < self.num_agents:
            model_choices.extend(['NaiveBayes'] * (self.num_agents - len(model_choices)))
        elif len(model_choices) > self.num_agents:
            model_choices = model_choices[:self.num_agents]

        # Embaralhar para distribuir papÃƒÂ©is espacialmente apenas no caso homogÃƒÂ©neo
        if self.homogeneous:
            random.shuffle(model_choices)

        for i in range(self.num_agents):
            model_choice = model_choices[i]
            
            # Definir pesos conforme o tipo
            if self.homogeneous:
                weights = {'KNN': 1/3, 'RandomForest': 1/3, 'NaiveBayes': 1/3}
            else:
                if model_choice == 'KNN':
                    weights = {'KNN': 0.6, 'RandomForest': 0.2, 'NaiveBayes': 0.2}
                elif model_choice == 'RandomForest':
                    weights = {'KNN': 0.2, 'RandomForest': 0.6, 'NaiveBayes': 0.2}
                elif model_choice == 'NaiveBayes':
                    weights = {'KNN': 0.2, 'RandomForest': 0.2, 'NaiveBayes': 0.6}
                else:
                    weights = {'KNN': 1/3, 'RandomForest': 1/3, 'NaiveBayes': 1/3}

            agent = AgentB(agent_id=i, start_pos=initial_pos, inference_weights=weights, model_choice=model_choice)
            self.agents.append(agent)

    def run_simulation(self, verbose=False, success_threshold_pct=100.0):
        """Executa simulaÃƒÂ§ÃƒÂ£o completa."""
        start_time = time.time()
        step = 0
        
        if verbose:
            self.logs.append(f"=== INÃƒÂCIO SIMULAÃƒâ€¡ÃƒÆ’O ABORDAGEM B ===")
            self.logs.append(f"Agentes: {self.num_agents} | CÃƒÂ©lulas livres: {self.free_cells}")
            self.logs.append(f"Bombas: {self.metrics['bomb_ratio']*100}%")
            self.logs.append(f"Tipo: {'HomogÃƒÂªneo' if self.homogeneous else 'HeterogÃƒÂªneo'}")
        
        while step < self.max_steps:
            step += 1
            agents_alive = len([a for a in self.agents if a.alive])
            
            if agents_alive == 0:
                break
            
            step_logs = []
            
            # Limpar reservas no inÃƒÂ­cio de cada passo
            self.shared_memory.clear_reservations()
            
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                # Cada agente decide DEPOIS do anterior ter atualizado a memÃƒÂ³ria
                next_pos = agent.choose_action(self.shared_memory, self.env)
                if next_pos:
                    log_msg = agent.move_to(next_pos, self.shared_memory, self.env)
                    if verbose and ("BOMBA" in log_msg or "aguardando" not in log_msg):
                        step_logs.append(log_msg)
            
            # Calcular percentagem explorada
            explored_free_cells = 0
            for i in range(self.env.size):
                for j in range(self.env.size):
                    pos = (i, j)
                    if pos in self.shared_memory.explored and self.env.grid[i, j] != 'B':
                        explored_free_cells += 1
            
            explored_pct = (explored_free_cells / self.free_cells * 100) if self.free_cells > 0 else 0
            
            # Atualizar mÃƒÂ©tricas
            self.metrics['explored_percentage'] = explored_pct
            self.metrics['explored_free_cells'] = explored_free_cells
            
            if verbose and step % 50 == 0:
                self.logs.append(f"Passo {step}: {explored_pct:.1f}% explorado, {agents_alive} agentes vivos")
            
            # Verificar critÃƒÂ©rio de sucesso
            if explored_pct >= success_threshold_pct and agents_alive > 0:
                self.metrics['success'] = True
                if verbose:
                    self.logs.append(f"Ã¢Å“â€¦ SUCESSO! {explored_pct:.1f}% explorado, {agents_alive} agentes vivos")
                break
        
        # Calcular mÃƒÂ©tricas finais
        end_time = time.time()
        self.metrics['execution_time'] = end_time - start_time
        self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
        self.metrics['steps_taken'] = step
        
        if verbose:
            self.logs.append(f"\n=== FIM DA SIMULAÃƒâ€¡ÃƒÆ’O ===")
            self.logs.append(f"Tempo: {self.metrics['execution_time']:.2f}s")
            self.logs.append(f"Passos: {self.metrics['steps_taken']}")
            self.logs.append(f"Explorado: {self.metrics['explored_percentage']:.1f}%")
            self.logs.append(f"Agentes vivos: {self.metrics['agents_alive']}")
            self.logs.append(f"Sucesso: {'SIM' if self.metrics['success'] else 'NÃƒÆ’O'}")
        
        return self.metrics
    
    def get_explored_percentage(self):
        """Calcula percentagem de cÃƒÂ©lulas livres exploradas"""
        explored_free_cells = 0
        for x in range(10):
            for y in range(10):
                pos = (x, y)
                if self.env.grid[x, y] != 'B' and self.shared_memory.cell_knowledge[pos]['explored']:
                    explored_free_cells += 1
        
        return (explored_free_cells / self.free_cells) * 100 if self.free_cells > 0 else 0
    
    def print_logs(self):
        """Exibe logs da simulaÃƒÂ§ÃƒÂ£o"""
        for log in self.logs:
            print(log)

# ============================================
# 7. BASELINE B: N AGENTES BFS COLABORATIVOS - CORRIGIDO
# ============================================

class BaselineB_BFS:
    """
    Baseline B: N agentes BFS colaborativos
    
    REGRAS CRÃƒÂTICAS:
    Ã¢Å“â€¦ N agentes permitidos
    Ã¢Å“â€¦ Todos executam o MESMO algoritmo (BFS puro)
    Ã¢Å“â€¦ ColaboraÃƒÂ§ÃƒÂ£o APENAS via memÃƒÂ³ria compartilhada
    Ã¢Å“â€¦ SEM aprendizagem
    Ã¢Å“â€¦ SEM motor de inferÃƒÂªncia
    Ã¢Å“â€¦ SEM pesos
    Ã¢Å“â€¦ SEM decisÃƒÂ£o inteligente
    """
    def __init__(self, num_agents=4, bomb_ratio=0.3, max_steps=500):
        self.env = EnvironmentB(bomb_ratio=bomb_ratio)
        self.shared_memory = SharedMemoryB()
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.agents = []
        self.logs = []
        
        # Calcular cÃƒÂ©lulas livres
        total_cells = self.env.size * self.env.size
        bomb_count = sum(1 for row in self.env.grid for cell in row if cell == 'B')
        self.free_cells = total_cells - bomb_count
        
        self.metrics = {
            'explored_percentage': 0,
            'agents_alive': 0,
            'bombs_triggered': 0,
            'steps_taken': 0,
            'execution_time': 0,
            'success': False,
            'bomb_ratio': bomb_ratio,
            'treasures_found': 0,
            'total_treasures': 0,
            'explored_free_cells': 0
        }
        
        # Criar agentes BFS
        self.setup_agents()
    
    def setup_agents(self):
        """
        Criar N agentes BFS
        TODOS executam o MESMO algoritmo
        DiferenÃƒÂ§a: apenas ID e direÃƒÂ§ÃƒÂ£o preferencial (para dividir espaÃƒÂ§o)
        """
        # Marcar (0,0) como explorada
        initial_pos = (0, 0)
        cell_content = self.env.get_cell(*initial_pos)
        self.shared_memory.explored.add(initial_pos)
        self.shared_memory.cell_knowledge[initial_pos]['explored'] = True
        self.shared_memory.cell_knowledge[initial_pos]['type'] = cell_content
        self.shared_memory.cell_knowledge[initial_pos]['safe'] = True
        
        # Criar N agentes BFS idÃƒÂªnticos
        for i in range(self.num_agents):
            agent = AgentBFS(agent_id=i, start_pos=initial_pos)
            self.agents.append(agent)
    
    def run(self):
        """Executa simulaÃƒÂ§ÃƒÂ£o com N agentes BFS"""
        start_time = time.time()
        step = 0
        
        # Loop principal
        while step < self.max_steps:
            step += 1
            agents_alive = len([a for a in self.agents if a.alive])
            
            if agents_alive == 0:
                break
            
            # âœ… CORREÃ‡ÃƒO: Limpar reservas no inÃ­cio de cada passo
            self.shared_memory.clear_reservations()
            
            # âœ… CORREÃ‡ÃƒO: Escolha SEQUENCIAL de aÃ§Ãµes (nÃ£o simultÃ¢nea)
            actions = {}
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                # Cada agente escolhe SUA aÃ§Ã£o vendo as reservas dos anteriores
                next_pos = agent.choose_action(self.shared_memory, self.env)
                actions[agent.id] = next_pos
            
            # Depois TODOS se movem simultaneamente
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                if agent.id in actions and actions[agent.id]:
                    log_msg = agent.move_to(actions[agent.id], self.shared_memory, self.env)
                    if verbose and log_msg:
                        print(f"[{time.strftime('%H:%M:%S')}] {log_msg}")
            
            # Calcular percentagem explorada
            explored_free_cells = 0
            for i in range(self.env.size):
                for j in range(self.env.size):
                    pos = (i, j)
                    if (pos in self.shared_memory.explored and 
                        self.env.grid[i, j] != 'B'):
                        explored_free_cells += 1
            
            explored_pct = (explored_free_cells / self.free_cells * 100) if self.free_cells > 0 else 0
            
            # Atualizar mÃƒÂ©tricas
            self.metrics['explored_percentage'] = explored_pct
            self.metrics['explored_free_cells'] = explored_free_cells
            
            # Verificar sucesso
            if explored_pct >= 100.0 and agents_alive > 0:
                self.metrics['success'] = True
                break
        
        # Calcular mÃƒÂ©tricas finais
        end_time = time.time()
        self.metrics['execution_time'] = end_time - start_time
        self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
        self.metrics['steps_taken'] = step
        self.metrics['bombs_triggered'] = sum(1 for a in self.agents if not a.alive)
        
        return self.metrics


# ============================================
# 8. FUNÃƒâ€¡Ãƒâ€¢ES DE ANÃƒÂLISE E COMPARAÃƒâ€¡ÃƒÆ’O
# ============================================

def run_multiple_simulations_b(num_simulations=5, num_agents=4, homogeneous=True):
    """Executa mÃƒÂºltiplas simulaÃƒÂ§ÃƒÂµes para estatÃƒÂ­sticas"""
    results = []
    
    for i in range(num_simulations):
        sim = ApproachBSimulation(
            num_agents=num_agents,
            bomb_ratio=0.3,
            homogeneous=homogeneous,
            max_steps=300
        )
        
        metrics = sim.run_simulation(verbose=False)
        results.append(metrics)
    
    # Calcular mÃƒÂ©dias
    avg_explored = np.mean([r['explored_percentage'] for r in results])
    avg_time = np.mean([r['execution_time'] for r in results])
    success_rate = np.mean([1 if r['success'] else 0 for r in results])
    avg_survivors = np.mean([r['agents_alive'] for r in results])
    
    return {
        'type': 'HomogÃƒÂªneo' if homogeneous else 'HeterogÃƒÂªneo',
        'num_agents': num_agents,
        'avg_explored': avg_explored,
        'avg_time': avg_time,
        'success_rate': success_rate,
        'avg_survivors': avg_survivors,
        'results': results
    }

def compare_approaches_b():
    """Compara abordagens homogÃƒÂªnea, heterogÃƒÂªnea e baseline BFS"""
    print("Comparando Abordagem B...")
    
    results = []
    agent_counts = [2, 4, 6, 8]
    
    for num_agents in agent_counts:
        print(f"\nTestando com {num_agents} agentes...")
        
        # HomogÃƒÂªneo
        homo_results = run_multiple_simulations_b(
            num_simulations=5, num_agents=num_agents, homogeneous=True
        )
        
        # HeterogÃƒÂªneo
        hetero_results = run_multiple_simulations_b(
            num_simulations=5, num_agents=num_agents, homogeneous=False
        )
        
        # Baseline BFS com N agentes
        bfs_results = []
        for _ in range(5):
            bfs = BaselineB_BFS(num_agents=num_agents)
            bfs_metrics = bfs.run()
            bfs_results.append(bfs_metrics)
        
        bfs_avg_explored = np.mean([r['explored_percentage'] for r in bfs_results])
        bfs_avg_time = np.mean([r['execution_time'] for r in bfs_results])
        bfs_success_rate = np.mean([1 if r['success'] else 0 for r in bfs_results])
        
        results.append({
            'num_agents': num_agents,
            'homogeneous': homo_results,
            'heterogeneous': hetero_results,
            'bfs': {
                'avg_explored': bfs_avg_explored,
                'avg_time': bfs_avg_time,
                'success_rate': bfs_success_rate,
                'results': bfs_results
            }
        })
        
        print(f"  HomogÃƒÂªneo: {homo_results['success_rate']:.0%} sucesso")
        print(f"  HeterogÃƒÂªneo: {hetero_results['success_rate']:.0%} sucesso")
        print(f"  BFS ({num_agents} agentes): {bfs_success_rate:.0%} sucesso")
    
    return results


def test_approach_b(bomb_ratios=None, num_agents=4, num_simulations=5, max_steps=3000,
                    success_threshold_pct=70.0, homogeneous=True):
    """
    Testa a Abordagem B com diferentes bomb_ratios.
    """
    if bomb_ratios is None:
        bomb_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = {}
    print("=" * 70)
    print("APPROACH B - TESTE POR BOMB RATIO (backtrack + ML)")
    print("=" * 70)
    print(f"Agentes: {num_agents} | SimulaÃƒÂ§ÃƒÂµes: {num_simulations} | Max passos: {max_steps}")
    print(f"Sucesso: >= {success_threshold_pct}% cÃƒÂ©lulas livres exploradas")
    print("=" * 70)
    for bomb_ratio in bomb_ratios:
        print(f"\n--- {bomb_ratio*100:.0f}% BOMBAS ---")
        sim_results = []
        for sim_num in range(num_simulations):
            sim = ApproachBSimulation(
                num_agents=num_agents,
                bomb_ratio=bomb_ratio,
                homogeneous=homogeneous,
                max_steps=max_steps
            )
            metrics = sim.run_simulation(verbose=False, success_threshold_pct=success_threshold_pct)
            sim_results.append(metrics)
            status = "Ã¢Å“â€¦" if metrics['success'] else "Ã¢ÂÅ’"
            print(f"  Sim {sim_num+1}: {metrics['explored_percentage']:.1f}% | "
                  f"Vivos: {metrics['agents_alive']}/{num_agents} | "
                  f"Passos: {metrics['steps_taken']} {status}")
        avg_explored = np.mean([r['explored_percentage'] for r in sim_results])
        std_explored = np.std([r['explored_percentage'] for r in sim_results])
        avg_alive = np.mean([r['agents_alive'] for r in sim_results])
        success_rate = np.mean([1 if r['success'] else 0 for r in sim_results])
        results[bomb_ratio] = {
            'avg_explored': avg_explored,
            'std_explored': std_explored,
            'avg_alive': avg_alive,
            'success_rate': success_rate,
            'simulations': sim_results
        }
        print(f"  MÃƒÂ©dia explorado: {avg_explored:.1f}% (Ã‚Â±{std_explored:.1f}%) | "
              f"Taxa sucesso: {success_rate*100:.0f}%")
    return results


# ============================================
# 9. EXECUÃƒâ€¡ÃƒÆ’O PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("PROJETO IA - ABORDAGEM B: EXPLORAÃƒâ€¡ÃƒÆ’O COMPLETA")
    print("="*60)
    
    # Teste rÃƒÂ¡pido
    print("\n1. Teste rÃƒÂ¡pido - 4 agentes heterogÃƒÂªneos:")
    sim = ApproachBSimulation(num_agents=4, homogeneous=False, max_steps=300)
    metrics = sim.run_simulation(verbose=True)
    sim.print_logs()
    
    # Teste Baseline BFS com 4 agentes
    print("\n\n2. Teste Baseline BFS - 4 agentes:")
    baseline = BaselineB_BFS(num_agents=4)
    baseline_metrics = baseline.run()
    print(f"Baseline BFS (4 agentes): {'Sucesso' if baseline_metrics['success'] else 'Falha'}")
    print(f"Explorado: {baseline_metrics['explored_percentage']:.1f}%")
    print(f"Passos: {baseline_metrics['steps_taken']}")
    print(f"Tempo: {baseline_metrics['execution_time']:.2f}s")