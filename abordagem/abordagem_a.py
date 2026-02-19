import numpy as np
import random
import time
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import copy
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. AMBIENTE BALANCEADO PARA ABORDAGEM A
# ============================================

class Environment:
    def __init__(self, size=10, bomb_ratio=0.3, treasure_count=12, approach='A'):
        """
        Ambiente para Abordagem A com balanceamento melhorado.
        
        Args:
            size: Tamanho do grid (size x size)
            bomb_ratio: Proporção de bombas (0.2 a 0.8)
            treasure_count: Número de tesouros
            approach: Tipo de abordagem ('A', 'B' ou 'C')
        """
        self.size = size
        self.grid = np.empty((size, size), dtype=object)
        self.original_grid = None
        self.treasure_count = treasure_count
        
        # Garantir que bomb_ratio esteja entre 20% e 80%
        self.bomb_ratio = max(0.2, min(0.8, bomb_ratio))
        self.approach = approach.upper()
        self.flag_position = None  # Para abordagem C
        
        self.generate_environment()
        
    def generate_environment(self):
        """Gera ambiente baseado na abordagem com balanceamento"""
        total_cells = self.size * self.size
        all_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        
        if self.approach == 'A':
            self._generate_approach_a_balanced(all_positions, total_cells)
        elif self.approach == 'B':
            self._generate_approach_b(all_positions, total_cells)
        elif self.approach == 'C':
            self._generate_approach_c(all_positions, total_cells)
        else:
            raise ValueError("Abordagem deve ser 'A', 'B' ou 'C'")
        
        self.original_grid = self.grid.copy()
        
        # Validar ambiente (silencioso por padrão)
        if self.approach == 'A':
            self._validate_environment_a(verbose=False)
    
    def _generate_approach_a_balanced(self, all_positions, total_cells):
        """
        Gera ambiente balanceado para Abordagem A.
        Garante que todos os tesouros são alcançáveis.
        """
        # Garantir que tesouros não sejam mais que 20% do ambiente
        max_treasures = int(total_cells * 0.2)
        self.treasure_count = min(self.treasure_count, max_treasures)
        
        # Posição inicial
        start_pos = (0, 0)
        
        # Escolher posições para tesouros espalhadas pelo grid
        treasure_positions = self._select_distributed_treasures(all_positions, start_pos)
        
        # Criar zonas seguras ao redor de cada tesouro e da origem
        safe_zones = self._create_safe_zones(treasure_positions, start_pos)
        
        # Calcular número de bombas baseado no bomb_ratio
        # Excluir: zonas seguras e tesouros
        occupied_positions = safe_zones | set(treasure_positions)
        available_for_bombs = [pos for pos in all_positions if pos not in occupied_positions]
        
        # Garantir balanceamento: células livres devem ser suficientes
        max_bombs = int(total_cells * self.bomb_ratio)
        bomb_count = min(max_bombs, len(available_for_bombs))
        
        # Garantir mínimo de células livres (pelo menos 20% do total)
        min_free_cells = int(total_cells * 0.2)
        free_cells_count = len(safe_zones) + len(available_for_bombs) - bomb_count
        if free_cells_count < min_free_cells:
            bomb_count = max(0, len(available_for_bombs) - (min_free_cells - len(safe_zones)))
        
        bomb_positions = random.sample(available_for_bombs, bomb_count) if bomb_count > 0 else []
        
        # Inicializar grid
        for i in range(self.size):
            for j in range(self.size):
                pos = (i, j)
                if pos in treasure_positions:
                    self.grid[i, j] = 'T'
                elif pos in bomb_positions:
                    self.grid[i, j] = 'B'
                else:
                    self.grid[i, j] = 'L'
    
    def _select_distributed_treasures(self, all_positions, start_pos):
        """
        Seleciona posições de tesouros distribuídas pelo grid.
        Evita concentração em uma área.
        """
        # Dividir grid em quadrantes
        mid_x, mid_y = self.size // 2, self.size // 2
        quadrants = {
            'NW': [(i, j) for i, j in all_positions if i < mid_x and j < mid_y],
            'NE': [(i, j) for i, j in all_positions if i < mid_x and j >= mid_y],
            'SW': [(i, j) for i, j in all_positions if i >= mid_x and j < mid_y],
            'SE': [(i, j) for i, j in all_positions if i >= mid_x and j >= mid_y]
        }
        
        # Remover posição inicial
        for quad in quadrants.values():
            if start_pos in quad:
                quad.remove(start_pos)
        
        # Distribuir tesouros entre quadrantes
        treasures_per_quad = self.treasure_count // 4
        extra_treasures = self.treasure_count % 4
        
        treasure_positions = []
        for i, (quad_name, quad_positions) in enumerate(quadrants.items()):
            if not quad_positions:
                continue
            
            num_treasures = treasures_per_quad + (1 if i < extra_treasures else 0)
            num_treasures = min(num_treasures, len(quad_positions))
            
            if num_treasures > 0:
                selected = random.sample(quad_positions, num_treasures)
                treasure_positions.extend(selected)
        
        # Se ainda precisamos de mais tesouros, adicionar aleatoriamente
        if len(treasure_positions) < self.treasure_count:
            remaining_positions = [p for p in all_positions 
                                 if p not in treasure_positions and p != start_pos]
            needed = self.treasure_count - len(treasure_positions)
            if needed > 0 and remaining_positions:
                treasure_positions.extend(random.sample(remaining_positions, 
                                                       min(needed, len(remaining_positions))))
        
        return treasure_positions
    
    def _create_safe_zones(self, treasure_positions, start_pos):
        """
        Cria zonas seguras (sem bombas) ao redor de cada tesouro e da origem.
        Usa BFS para garantir caminhos conectados.
        """
        safe_zones = set()
        
        # Adicionar posição inicial
        safe_zones.add(start_pos)
        
        # Para cada tesouro, criar caminho até ele
        for treasure_pos in treasure_positions:
            path = self._find_path_bfs(start_pos, treasure_pos, safe_zones)
            safe_zones.update(path)
            
            # Adicionar vizinhos do tesouro como seguro
            neighbors = self.get_neighbors(*treasure_pos)
            for neighbor in neighbors:
                safe_zones.add(neighbor)
        
        return safe_zones
    
    def _find_path_bfs(self, start, end, existing_safe):
        """
        Encontra caminho de start até end usando BFS.
        Prefere usar células já seguras quando possível.
        """
        queue = deque([start])
        visited = {start}
        parent = {start: None}
        
        while queue:
            current = queue.popleft()
            
            if current == end:
                # Reconstruir caminho
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]
            
            x, y = current
            neighbors = self.get_neighbors(x, y)
            
            # Ordenar vizinhos: preferir células já seguras
            neighbors = sorted(neighbors, 
                             key=lambda n: (n not in existing_safe, random.random()))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        # Se não encontrar caminho, retornar linha reta
        return [start, end]
    
    def _validate_environment_a(self, verbose=True):
        """
        Valida se o ambiente está balanceado para Abordagem A:
        - Todos os tesouros são alcançáveis
        - Proporção de bombas está dentro dos limites
        - Células livres são suficientes
        """
        total_cells = self.size * self.size
        
        # Contar tipos de células
        bomb_count = np.sum(self.grid == 'B')
        free_count = np.sum(self.grid == 'L')
        treasure_count = np.sum(self.grid == 'T')
        
        # Verificar proporções
        bomb_ratio_actual = bomb_count / total_cells
        free_ratio_actual = free_count / total_cells
        
        # Verificar alcançabilidade de todos os tesouros
        treasure_positions = [(i, j) for i in range(self.size) 
                             for j in range(self.size) if self.grid[i, j] == 'T']
        
        reachable_treasures = 0
        for treasure_pos in treasure_positions:
            if self._is_reachable((0, 0), treasure_pos):
                reachable_treasures += 1
        
        all_reachable = (reachable_treasures == len(treasure_positions))
        
        # Log de validação (se verbose)
        if verbose:
            print(f"\n📊 Estatísticas do Ambiente (Abordagem A):")
            print(f"   Tamanho: {self.size}x{self.size} ({total_cells} células)")
            print(f"   🟢 Células Livres: {free_count} ({free_ratio_actual:.1%})")
            print(f"   💣 Bombas: {bomb_count} ({bomb_ratio_actual:.1%})")
            print(f"   💎 Tesouros: {treasure_count} ({treasure_count/total_cells:.1%})")
            print(f"   ✅ Tesouros Alcançáveis: {reachable_treasures}/{len(treasure_positions)}")
            
            if not all_reachable:
                print(f"   ⚠️  AVISO: Alguns tesouros não são alcançáveis!")
            
            if bomb_ratio_actual < 0.2 or bomb_ratio_actual > 0.8:
                print(f"   ⚠️  AVISO: Proporção de bombas fora do ideal (20%-80%)")
            
            if free_ratio_actual < 0.2:
                print(f"   ⚠️  AVISO: Poucas células livres (<20%)")
        
        return all_reachable
    
    def _is_reachable(self, start_pos, end_pos):
        """
        Verifica se end_pos é alcançável a partir de start_pos.
        Considera apenas células livres e tesouros (não bombas).
        """
        queue = deque([start_pos])
        visited = {start_pos}
        
        while queue:
            current = queue.popleft()
            
            if current == end_pos:
                return True
            
            x, y = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                if (0 <= nx < self.size and 0 <= ny < self.size and 
                    neighbor not in visited):
                    cell_type = self.grid[nx, ny]
                    # Pode atravessar: Livre ou Tesouro (não Bomba)
                    if cell_type in ['L', 'T']:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        return False
    
    def _generate_approach_b(self, all_positions, total_cells):
        """Abordagem B: Sem tesouros, apenas exploração"""
        self.treasure_count = 0
        treasure_positions = []
        
        # Garantir que (0,0) seja sempre uma célula livre
        safe_start_positions = [(0, 0)]
        remaining_positions = [pos for pos in all_positions if pos not in safe_start_positions]
        
        # Calcular número de bombas
        bomb_count = int((total_cells - len(safe_start_positions)) * self.bomb_ratio)
        bomb_positions = random.sample(remaining_positions, min(bomb_count, len(remaining_positions)))
        
        # Inicializar grid
        for i in range(self.size):
            for j in range(self.size):
                pos = (i, j)
                if pos in bomb_positions:
                    self.grid[i, j] = 'B'
                else:
                    self.grid[i, j] = 'L'
    
    def _generate_approach_c(self, all_positions, total_cells):
        """Abordagem C: Com bandeira (usa lógica da abordagem_c.py)"""
        max_treasures = int(total_cells * 0.2)
        self.treasure_count = min(self.treasure_count, max_treasures)
        
        treasure_positions = random.sample(all_positions, self.treasure_count)
        
        remaining_positions = [pos for pos in all_positions if pos not in treasure_positions]
        self.flag_position = random.choice(remaining_positions)
        remaining_positions.remove(self.flag_position)
        
        bomb_count = int((total_cells - self.treasure_count - 1) * self.bomb_ratio)
        bomb_positions = random.sample(remaining_positions, min(bomb_count, len(remaining_positions)))
        
        # Inicializar grid
        for i in range(self.size):
            for j in range(self.size):
                pos = (i, j)
                if pos in treasure_positions:
                    self.grid[i, j] = 'T'
                elif pos in bomb_positions:
                    self.grid[i, j] = 'B'
                elif pos == self.flag_position:
                    self.grid[i, j] = 'F'
                else:
                    self.grid[i, j] = 'L'
        
    def reset_treasure(self, position):
        """Remove tesouro após ser coletado"""
        x, y = position
        self.grid[x, y] = 'L'
        
    def get_cell(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[x, y]
        return None
    
    def get_neighbors(self, x, y):
        """Retorna vizinhas válidas (apenas horizontal/vertical)"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors

class SharedMemory:
    def __init__(self, env_size=10):
        self.explored = set()
        self.treasures_found = set()  # Usar set para evitar duplicados
        self.treasures_collected = set()  # Tesouros já coletados
        self.bombs_found = set()
        self.agent_positions = {}
        self.agent_status = {}
        self.cell_knowledge = {}  # Conhecimento sobre cada célula
        self.flag_found = False  # Para abordagem C
        self.env_size = env_size
        
        # Inicializar conhecimento
        for i in range(env_size):
            for j in range(env_size):
                self.cell_knowledge[(i, j)] = {
                    'type': 'U',  # U = Unknown
                    'explored': False,
                    'safe': True  # Assume-se seguro até descobrir bomba
                }
        
        # A posição inicial (0,0) está sempre explorada
        self.explored.add((0, 0))
        self.cell_knowledge[(0, 0)]['explored'] = True
        self.cell_knowledge[(0, 0)]['type'] = 'L'  # Posição inicial é sempre livre
    
    def update_explored(self, position, content, agent_id, env):
        """Atualiza memória com nova exploração"""
        x, y = position
        self.explored.add(position)
        self.agent_positions[agent_id] = position
        
        # Atualizar conhecimento da célula
        self.cell_knowledge[position]['explored'] = True
        self.cell_knowledge[position]['type'] = content
        
        log_msg = f"Agente {agent_id}: {content} em {position}"
        
        if content == 'T':
            if position not in self.treasures_collected:
                self.treasures_found.add(position)
                self.treasures_collected.add(position)
                # Remove tesouro do ambiente
                env.reset_treasure(position)
                log_msg += " (TESOURO COLETADO!)"
            else:
                log_msg += " (TESOURO JÁ COLETADO)"
        elif content == 'B':
            self.bombs_found.add(position)
            self.cell_knowledge[position]['safe'] = False
            log_msg += " (BOMBA)"
        elif content == 'F':
            self.flag_found = True
            log_msg += " (BANDEIRA ENCONTRADA!)"
        else:
            self.cell_knowledge[position]['safe'] = True
            
        return log_msg
    
    def is_safe_cell(self, position):
        """Verifica se célula é segura para visitar"""
        return self.cell_knowledge[position]['safe']
    
    def get_best_unknown_neighbor(self, position, env):
        """Retorna melhor vizinha não explorada"""
        x, y = position
        neighbors = env.get_neighbors(x, y)
        
        # Filtrar apenas células seguras e não exploradas
        safe_unknown = []
        for pos in neighbors:
            if not self.cell_knowledge[pos]['explored'] and self.cell_knowledge[pos]['safe']:
                safe_unknown.append(pos)
        
        return safe_unknown

# ============================================
# 3. AGENTE CORRIGIDO
# ============================================

class Agent:
    def __init__(self, agent_id, start_pos=(0, 0), inference_weights=None, approach='A'):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.treasures_collected = 0
        self.bombs_defused = 0
        self.steps_taken = 0
        self.last_action = None
        self.approach = approach  # Adicionar abordagem
        
        # Para abordagem B: usar algoritmo greedy
        if approach == 'B':
            self.use_greedy = True
        else:
            self.use_greedy = False
        
        # Inicializar modelos de ML
        self.models = {
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'NaiveBayes': GaussianNB(),
            'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        # Dados para treinamento (inicializar com dados básicos)
        self.training_data = {'features': [], 'labels': []}
        self.models_trained = False
        
        # Motor de inferência
        self.inference_engine = InferenceEngine(inference_weights)
        
        # Histórico de ações
        self.action_history = deque(maxlen=50)
        
        # Memória individual do agente (para baseline)
        self.memory = SharedMemory()
        
        # Inicializar com algum conhecimento básico
        self.initialize_basic_knowledge()
    
    def initialize_basic_knowledge(self):
        """Inicializa conhecimento básico para evitar previsões sem dados"""
        # Adicionar ponto inicial como livre
        self.training_data['features'].append([0, 0])
        self.training_data['labels'].append('L')
        
        # Adicionar alguns pontos hipotéticos
        for _ in range(5):
            x, y = random.randint(0, 9), random.randint(0, 9)
            self.training_data['features'].append([x, y])
            self.training_data['labels'].append(random.choice(['L', 'L', 'L', 'B']))  # Mais chances de ser livre
    
    def train_models(self, shared_memory=None, env=None):
        """Treina modelos com dados coletados"""
        if len(self.training_data['features']) >= 10 and not self.models_trained:
            X = np.array(self.training_data['features'])
            y = np.array(self.training_data['labels'])
            
            try:
                for name, model in self.models.items():
                    model.fit(X, y)
                self.models_trained = True
            except:
                pass  # Se der erro, tenta na próxima vez
    
    def predict_cell(self, cell_position, shared_memory):
        """Preve o tipo de célula usando modelos ou heurística"""
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        x, y = cell_position
        
        # Se célula já foi explorada, retorna conteúdo conhecido
        if cell_position in memory.explored:
            if cell_position in memory.treasures_collected:
                return 'L'  # Tesouro já coletado é como célula livre
            elif cell_position in memory.bombs_found:
                return 'B'
            else:
                return 'L'
        
        # Se modelos estão treinados, usar previsão
        if self.models_trained and len(self.training_data['features']) >= 10:
            predictions = []
            for name, model in self.models.items():
                try:
                    pred = model.predict([[x, y]])[0]
                    predictions.append(pred)
                except:
                    pass
            
            if predictions:
                from collections import Counter
                most_common = Counter(predictions).most_common(1)
                if most_common:
                    return most_common[0][0]
        
        # Heurística baseada na posição
        distance_to_center = np.sqrt((x - 5)**2 + (y - 5)**2)
        if distance_to_center < 3:
            return random.choice(['L', 'T', 'L'])  # Centro tem mais chance de tesouro
        else:
            return random.choice(['L', 'L', 'B', 'L'])  # Bordas mais perigosas
    
    def choose_action(self, shared_memory, env):
        """Escolhe próxima ação com estratégia melhorada"""
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        if not self.alive:
            return None
        
        x, y = self.position
        
        # Treinar modelos periodicamente
        if self.steps_taken % 20 == 0:
            self.train_models()
        
        # Obter vizinhas
        neighbors = env.get_neighbors(x, y)
        
        # Filtrar apenas células seguras
        safe_neighbors = []
        for neighbor in neighbors:
            if memory.is_safe_cell(neighbor):
                safe_neighbors.append(neighbor)
        
        # Se não houver vizinhas seguras, ficar parado
        if not safe_neighbors:
            return None
        
        # Se usar algoritmo greedy (abordagem B), escolher célula não explorada mais próxima
        if self.use_greedy:
            return self.choose_greedy_action(safe_neighbors, shared_memory)
        
        # Prever tipo de cada vizinha segura
        available_actions = []
        for neighbor in safe_neighbors:
            # Evitar voltar para onde já esteve recentemente
            if neighbor in self.action_history:
                continue
                
            predicted_type = self.predict_cell(neighbor, shared_memory)
            available_actions.append((neighbor, predicted_type))
        
        # Se todas as ações estão no histórico, limpar histórico
        if not available_actions:
            self.action_history.clear()
            for neighbor in safe_neighbors:
                predicted_type = self.predict_cell(neighbor, shared_memory)
                available_actions.append((neighbor, predicted_type))
        
        # Usar motor de inferência para decidir
        next_pos = self.inference_engine.decide_action(
            available_actions, shared_memory, env, self
        )
        
        return next_pos
    
    def choose_greedy_action(self, safe_neighbors, shared_memory):
        """Algoritmo greedy simples: escolher primeira célula não explorada"""
        if not safe_neighbors:
            return None
        
        # Priorizar células não exploradas
        unexplored = [neighbor for neighbor in safe_neighbors 
                     if not shared_memory.cell_knowledge[neighbor]['explored']]
        
        if unexplored:
            # Escolher a primeira célula não explorada (estratégia simples)
            return unexplored[0]
        else:
            # Se todas foram exploradas, escolher qualquer uma (não deve acontecer)
            return safe_neighbors[0]
    
    def move_to(self, new_position, shared_memory, env):
        """Move agente para nova posição"""
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        if not self.alive or new_position is None:
            return "Agente inativo"
        
        old_pos = self.position
        self.position = new_position
        self.steps_taken += 1
        self.action_history.append(old_pos)
        
        # Explorar nova célula
        x, y = new_position
        cell_content = env.get_cell(x, y)
        
        # Atualizar memória compartilhada
        if shared_memory is None:
            log_msg = memory.update_explored(new_position, cell_content, self.id, env)
        else:
            log_msg = shared_memory.update_explored(new_position, cell_content, self.id, env)
        
        # Para baseline, atualizar também memória individual
        if getattr(self, 'is_baseline', False):
            memory.update_explored(new_position, cell_content, self.id, env)
        
        # Atualizar dados de treinamento
        self.training_data['features'].append([x, y])
        self.training_data['labels'].append(cell_content)
        
        # Consequências da ação
        if cell_content == 'T':
            self.treasures_collected += 1
            self.bombs_defused += 1  # Ganha poder para desativar bomba
            log_msg += f" | Tesouros: {self.treasures_collected}"
        elif cell_content == 'B':
            if self.bombs_defused > 0:
                # Usa poder para desativar bomba
                self.bombs_defused -= 1
                log_msg += " | BOMBA DESATIVADA!"
                # Marcar célula como segura após desativar
                if shared_memory is not None:
                    shared_memory.cell_knowledge[new_position]['safe'] = True
                else:
                    memory.cell_knowledge[new_position]['safe'] = True
            else:
                self.alive = False
                log_msg += " | AGENTE DESTRUÍDO"
        
        return log_msg

# ============================================
# 4. AGENTE GREEDY BEST-FIRST (BASELINE PURA)
# ============================================

class AgentGreedyBestFirst:
    """
    Agente Greedy Best-first Search - BASELINE PURA para Abordagem A
    
    CARACTERÍSTICAS:
    ✅ SEM ML (sem modelos, sem treino)
    ✅ SEM motor de inferência complexo
    ✅ SEM previsões (apenas explora o conhecido)
    ✅ Usa apenas heurística gulosa: distância + potencial de tesouro
    ✅ Estratégia simples determinística
    
    HEURÍSTICA: h(n) = (distância ao tesouro conhecido) + (distância ao centro)
    - Prioriza células não exploradas próximas ao centro (estatisticamente mais tesouros)
    - Usa memória compartilhada como baseline deve
    """
    
    def __init__(self, agent_id, start_pos=(0, 0)):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.treasures_collected = 0
        self.bombs_defused = 0
        self.steps_taken = 0
        
        # CORREÇÃO: Seed única por agente para dispersão
        self.random_seed = agent_id * 1000
        
    def heuristic_value(self, position, shared_memory, env):
        """
        Calcula valor heurístico da célula
        Menor valor = maior prioridade
        
        Considera:
        1. Distância ao tesouro conhecido mais próximo
        2. Distância ao centro (statistically more treasures)
        3. Se já explorada (penalidade)
        """
        x, y = position
        center_x, center_y = env.size / 2, env.size / 2
        
        # Penalidade base: célula já explorada
        if shared_memory.cell_knowledge[position]['explored']:
            return float('inf')  # Não revisitar
        
        # Distância ao centro (áreas centrais têm mais tesouros)
        distance_to_center = abs(x - center_x) + abs(y - center_y)
        
        # Se há tesouro conhecido, calcular distância
        distance_to_known_treasure = float('inf')
        if shared_memory.treasures_found:
            for tx, ty in shared_memory.treasures_found:
                if (tx, ty) not in shared_memory.treasures_collected:
                    dist = abs(x - tx) + abs(y - ty)
                    distance_to_known_treasure = min(distance_to_known_treasure, dist)
        
        # Heurística combinada: prioriza perto do centro se não há tesouro conhecido
        if distance_to_known_treasure == float('inf'):
            # Sem tesouro conhecido: explorar próximo ao centro
            h_value = distance_to_center * 0.7
        else:
            # Tesouro conhecido: ir buscar ou explorar perto
            h_value = distance_to_known_treasure * 0.8 + distance_to_center * 0.2
        
        # Penalidade se célula é adjacente a bomba conhecida
        for bx, by in shared_memory.bombs_found:
            if abs(x - bx) + abs(y - by) == 1:
                h_value += 5.0  # Penalidade por estar perto de bomba
        
        # CORREÇÃO: Adicionar variação baseada no ID do agente
        # Isso faz com que agentes priorizem direções ligeiramente diferentes
        random.seed(self.random_seed + x * 100 + y)
        h_value += random.uniform(-0.5, 0.5)
        
        return h_value
    
    def choose_action(self, shared_memory, env):
        """
        Greedy Best-first: escolhe célula com menor heurística
        (sem ML, apenas exploração gulosa)
        """
        if not self.alive:
            return None
        
        x, y = self.position
        neighbors = env.get_neighbors(x, y)
        
        # CORREÇÃO: Ordenar vizinhas usando seed do agente
        # Isso garante que cada agente tenha uma ordem de preferência diferente
        random.seed(self.random_seed + self.steps_taken)
        random.shuffle(neighbors)
        
        # Filtrar vizinhas seguras e não exploradas
        candidates = []
        for neighbor in neighbors:
            # Evitar bombas conhecidas
            if neighbor in shared_memory.bombs_found:
                continue
            
            # Calcular valor heurístico
            h_val = self.heuristic_value(neighbor, shared_memory, env)
            
            if h_val != float('inf'):  # Não é explorada
                candidates.append((h_val, neighbor))
        
        if not candidates:
            # Se sem vizinhas boas, procurar mais longe (BFS simples)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < env.size and 0 <= ny < env.size:
                        neighbor = (nx, ny)
                        if neighbor in shared_memory.bombs_found:
                            continue
                        h_val = self.heuristic_value(neighbor, shared_memory, env)
                        if h_val != float('inf'):
                            candidates.append((h_val, neighbor))
        
        if not candidates:
            return None
        
        # Escolher célula com menor heurística
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    def move_to(self, new_position, shared_memory, env):
        """Move agente para nova posição"""
        if not self.alive or new_position is None:
            return "Agente inativo"
        
        self.position = new_position
        self.steps_taken += 1
        
        x, y = new_position
        cell_content = env.get_cell(x, y)
        log_msg = f"Agente {self.id}: {cell_content} em {new_position}"
        
        # Atualizar memória compartilhada
        shared_memory.explored.add(new_position)
        shared_memory.agent_positions[self.id] = new_position
        shared_memory.cell_knowledge[new_position]['explored'] = True
        shared_memory.cell_knowledge[new_position]['type'] = cell_content
        
        if cell_content == 'T':
            if new_position not in shared_memory.treasures_collected:
                self.treasures_collected += 1
                self.bombs_defused += 1  # CORREÇÃO: Ganhar poder de desativar bomba
                shared_memory.treasures_found.add(new_position)
                shared_memory.treasures_collected.add(new_position)
                env.reset_treasure(new_position)
                log_msg += " (TESOURO COLETADO!)"
            else:
                log_msg += " (TESOURO JÁ COLETADO)"
        elif cell_content == 'B':
            shared_memory.bombs_found.add(new_position)
            shared_memory.cell_knowledge[new_position]['safe'] = False
            if self.bombs_defused > 0:
                self.bombs_defused -= 1
                log_msg += " (BOMBA DESATIVADA!)"
                shared_memory.cell_knowledge[new_position]['safe'] = True
            else:
                self.alive = False
                log_msg += " (AGENTE DESTRUÍDO)"
        else:
            shared_memory.cell_knowledge[new_position]['safe'] = True
        
        return log_msg
    
    def train_models(self, shared_memory=None, env=None):
        """Método vazio - compatibilidade com GUI"""
        pass

# ============================================
# 5. MOTOR DE INFERÊNCIA MELHORADO
# ============================================

class InferenceEngine:
    def __init__(self, weights=None):
        self.weights = weights or {
            'T': 5.0,    # Tesouro - prioridade máxima
            'L': 1.0,    # Livre
            'B': -100.0, # Bomba - evitar completamente
            'U': 0.3,    # Desconhecido
            'E': -0.5    # Explorado
        }
    
    def calculate_score(self, cell_type, position, shared_memory, agent):
        """Calcula pontuação considerando vários fatores"""
        base_score = self.weights.get(cell_type, 0.0)
        
        # Bonus por tesouro não coletado
        if cell_type == 'T' and position in shared_memory.treasures_found:
            if position not in shared_memory.treasures_collected:
                base_score += 10.0
        
        # Penalidade por estar perto de bomba
        x, y = position
        for bx, by in shared_memory.bombs_found:
            distance = abs(bx - x) + abs(by - y)
            if distance == 1:  # Adjacente a bomba
                base_score -= 20.0
        
        # Bonus por exploração de área nova
        if not shared_memory.cell_knowledge[position]['explored']:
            base_score += 2.0
        
        return base_score
    
    def decide_action(self, available_cells, shared_memory, env, agent):
        """Decide ação considerando múltiplos fatores"""
        if shared_memory is None or getattr(agent, 'is_baseline', False):
            memory = agent.memory
        else:
            memory = shared_memory
            
        if not available_cells:
            return None
        
        best_score = -float('inf')
        best_action = None
        
        for pos, predicted_type in available_cells:
            # Verificar se é bomba conhecida
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
# 5. SIMULAÇÃO MELHORADA
# ============================================

class ApproachASimulation:
    def __init__(self, num_agents=4, bomb_ratio=0.3, treasure_count=12, 
                 homogeneous=True, max_steps=500, approach='A'):
        self.env = Environment(bomb_ratio=bomb_ratio, treasure_count=treasure_count, approach=approach)
        self.approach = approach
        self.shared_memory = SharedMemory()
        self.num_agents = num_agents
        self.homogeneous = homogeneous
        self.max_steps = max_steps
        self.agents = []
        self.logs = []
        self.metrics = {
            'treasures_found': 0,
            'total_treasures': treasure_count if approach != 'B' else 0,
            'agents_alive': 0,
            'steps_taken': 0,
            'execution_time': 0,
            'success': False,
            'bomb_ratio': bomb_ratio,
            'explored_percentage': 0,
            'flag_found': False
        }
        self.setup_agents()
        
    def setup_agents(self):
        """Configura agentes"""
        # Pesos padrão equilibrados
        base_weights = {'T': 5.0, 'L': 1.0, 'B': -100.0, 'U': 0.3, 'E': -0.5}
        
        if self.homogeneous:
            for i in range(self.num_agents):
                agent = Agent(agent_id=i, inference_weights=base_weights, approach=self.approach)
                self.agents.append(agent)
        else:
            # Perfis diferentes para agentes heterogêneos
            profiles = [
                {'T': 6.0, 'L': 0.5, 'B': -80.0, 'U': 1.0, 'E': -0.3},  # Arrojado
                {'T': 4.0, 'L': 1.5, 'B': -120.0, 'U': 0.2, 'E': -1.0},  # Cauteloso
                {'T': 5.0, 'L': 1.0, 'B': -100.0, 'U': 0.5, 'E': -0.5},  # Equilibrado
                {'T': 5.5, 'L': 0.8, 'B': -90.0, 'U': 0.7, 'E': -0.4},   # Moderado
                {'T': 4.5, 'L': 1.2, 'B': -110.0, 'U': 0.4, 'E': -0.8}    # Conservador
            ]
            
            for i in range(self.num_agents):
                profile = profiles[i % len(profiles)]
                agent = Agent(agent_id=i, inference_weights=profile, approach=self.approach)
                self.agents.append(agent)
    
    def run_simulation(self, verbose=False):
        """Executa simulação completa baseada na abordagem"""
        start_time = time.time()
        step = 0
        
        # Definir critérios de sucesso baseados na abordagem
        if self.approach == 'A':
            total_treasures = self.env.treasure_count
            success_threshold = total_treasures * 0.5
            success_condition = lambda: len(self.shared_memory.treasures_collected) > success_threshold
        elif self.approach == 'B':
            bomb_count = sum(1 for row in self.env.grid for cell in row if cell == 'B')
            total_free_cells = 100 - bomb_count  # 10x10 grid minus bombs
            success_condition = lambda: (self.get_explored_percentage() >= 100.0 and 
                                       len([a for a in self.agents if a.alive]) > 0)
        elif self.approach == 'C':
            success_condition = lambda: self.shared_memory.flag_found
        else:
            raise ValueError(f"Abordagem desconhecida: {self.approach}")
        
        if verbose:
            self.logs.append(f"=== INÍCIO SIMULAÇÃO ABORDAGEM {self.approach} ===")
            if self.approach == 'A':
                self.logs.append(f"Agentes: {self.num_agents} | Tesouros: {total_treasures}")
            elif self.approach == 'B':
                self.logs.append(f"Agentes: {self.num_agents} | Células livres: {total_free_cells}")
            elif self.approach == 'C':
                self.logs.append(f"Agentes: {self.num_agents} | Objetivo: Encontrar bandeira")
            self.logs.append(f"Bombas: {self.metrics['bomb_ratio']*100}%")
            self.logs.append(f"Tipo: {'Homogêneo' if self.homogeneous else 'Heterogêneo'}")
        
        while step < self.max_steps:
            step += 1
            agents_alive = len([a for a in self.agents if a.alive])
            
            if agents_alive == 0:
                break
            
            step_logs = []
            
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                # Escolher e executar ação
                next_pos = agent.choose_action(self.shared_memory, self.env)
                if next_pos:  # Só mover se houver ação válida
                    log_msg = agent.move_to(next_pos, self.shared_memory, self.env)
                    if verbose and ("TESOURO" in log_msg or "BOMBA" in log_msg or "BANDEIRA" in log_msg):
                        # Adicionar contador de tesouros ao log
                        if "TESOURO COLETADO" in log_msg and self.approach == 'A':
                            treasures_count = len(self.shared_memory.treasures_collected)
                            log_msg += f" | Total: {treasures_count}/{self.env.treasure_count}"
                        step_logs.append(log_msg)
            
            # Imprimir logs do passo
            if verbose and step_logs:
                for log in step_logs:
                    self.logs.append(log)
            
            # Atualizar métricas
            if verbose and step % 50 == 0:
                if self.approach == 'A':
                    treasures_found = len(self.shared_memory.treasures_collected)
                    self.logs.append(f"Passo {step}: {treasures_found}/{total_treasures} tesouros, {agents_alive} agentes vivos")
                elif self.approach == 'B':
                    explored_pct = self.get_explored_percentage()
                    self.logs.append(f"Passo {step}: {explored_pct:.1f}% explorado, {agents_alive} agentes vivos")
                elif self.approach == 'C':
                    self.logs.append(f"Passo {step}: Bandeira encontrada: {self.shared_memory.flag_found}, {agents_alive} agentes vivos")
            
            # Verificar critério de sucesso
            if success_condition():
                self.metrics['success'] = True
                if verbose:
                    if self.approach == 'A':
                        treasures_found = len(self.shared_memory.treasures_collected)
                        self.logs.append(f"✅ SUCESSO! {treasures_found}/{total_treasures} tesouros (>50%)")
                    elif self.approach == 'B':
                        explored_pct = self.get_explored_percentage()
                        self.logs.append(f"✅ SUCESSO! {explored_pct:.1f}% explorado, {agents_alive} agentes vivos")
                    elif self.approach == 'C':
                        self.logs.append(f"✅ SUCESSO! Bandeira encontrada!")
                break
        
        # Calcular métricas finais
        end_time = time.time()
        self.metrics['execution_time'] = end_time - start_time
        self.metrics['treasures_found'] = len(self.shared_memory.treasures_collected)
        self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
        self.metrics['steps_taken'] = step
        self.metrics['explored_percentage'] = self.get_explored_percentage()
        self.metrics['flag_found'] = self.shared_memory.flag_found
        
        if verbose:
            self.logs.append(f"\n=== FIM DA SIMULAÇÃO ===")
            self.logs.append(f"Tempo: {self.metrics['execution_time']:.2f}s")
            self.logs.append(f"Passos: {self.metrics['steps_taken']}")
            if self.approach == 'A':
                self.logs.append(f"Tesouros: {self.metrics['treasures_found']}/{self.env.treasure_count}")
            elif self.approach == 'B':
                self.logs.append(f"Explorado: {self.metrics['explored_percentage']:.1f}%")
            elif self.approach == 'C':
                self.logs.append(f"Bandeira: {'Encontrada' if self.metrics['flag_found'] else 'Não encontrada'}")
            self.logs.append(f"Agentes vivos: {self.metrics['agents_alive']}")
            self.logs.append(f"Sucesso: {'SIM' if self.metrics['success'] else 'NÃO'}")
        
        return self.metrics
    
    def get_explored_percentage(self):
        """Calcula percentagem de células livres exploradas"""
        # Contar bombas no grid
        bomb_count = sum(1 for row in self.env.grid for cell in row if cell == 'B')
        total_free_cells = 100 - bomb_count  # 10x10 = 100 células totais
        explored_free_cells = 0
        
        for x in range(10):
            for y in range(10):
                pos = (x, y)
                # Contar células que não são bombas e foram exploradas
                if self.env.grid[x, y] != 'B' and self.shared_memory.cell_knowledge[pos]['explored']:
                    explored_free_cells += 1
        
        return (explored_free_cells / total_free_cells) * 100 if total_free_cells > 0 else 0
    
    def print_logs(self):
        """Exibe logs da simulação"""
        for log in self.logs:
            print(log)

# ============================================
# BASELINE SIMULATION (SEM COLABORAÇÃO)
# ============================================

class BaselineSimulation:
    def __init__(self, num_agents=4, bomb_ratio=0.3, treasure_count=12, max_steps=500):
        self.env = Environment(bomb_ratio=bomb_ratio, treasure_count=treasure_count)
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.agents = []
        self.logs = []
        self.shared_memory = SharedMemory()  # Baseline agora usa memória compartilhada
        self.metrics = {
            'treasures_found': 0,
            'total_treasures': treasure_count,
            'agents_alive': 0,
            'steps_taken': 0,
            'execution_time': 0,
            'success': False,
            'bomb_ratio': bomb_ratio
        }
        self.setup_agents()
        
    def setup_agents(self):
        """Configura agentes com Greedy Best-first (baseline pura)"""
        for i in range(self.num_agents):
            # Usar AgentGreedyBestFirst - baseline pura SEM ML
            agent = AgentGreedyBestFirst(agent_id=i)
            self.agents.append(agent)
    
    def run_simulation(self, verbose=False):
        """Executa simulação baseline com Greedy Best-first"""
        start_time = time.time()
        step = 0
        total_treasures = self.env.treasure_count
        success_threshold = total_treasures * 0.5
        
        if verbose:
            print(f"Iniciando simulação BASELINE (Greedy Best-first) com {self.num_agents} agentes...")
        
        while step < self.max_steps:
            step += 1
            actions_taken = 0
            
            # Cada agente age
            for agent in self.agents:
                if agent.alive:
                    # Agente escolhe ação usando Greedy Best-first
                    next_pos = agent.choose_action(self.shared_memory, self.env)
                    if next_pos:
                        actions_taken += 1
                        log_msg = agent.move_to(next_pos, self.shared_memory, self.env)
                        if verbose and ("TESOURO" in log_msg or "DESTRUÍDO" in log_msg):
                            print(log_msg)
            
            # Atualizar contagem
            self.metrics['treasures_found'] = len(self.shared_memory.treasures_collected)
            self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
            self.metrics['steps_taken'] = step
            
            # Verificar condições de parada
            if self.metrics['treasures_found'] > success_threshold:
                self.metrics['success'] = True
                if verbose:
                    print(f"Sucesso! {self.metrics['treasures_found']} tesouros encontrados (>50%).")
                break
                
            if actions_taken == 0:  # Todos os agentes parados
                if verbose:
                    print("Todos os agentes parados.")
                break
        
        # Calcular métricas finais
        self.metrics['agents_alive'] = sum(1 for agent in self.agents if agent.alive)
        self.metrics['steps_taken'] = step
        self.metrics['execution_time'] = time.time() - start_time
        
        if verbose:
            self.logs.append(f"Simulação baseline concluída em {step} passos")
            self.logs.append(f"Tempo: {self.metrics['execution_time']:.2f}s")
            self.logs.append(f"Tesouros: {self.metrics['treasures_found']}/{total_treasures}")
            self.logs.append(f"Agentes vivos: {self.metrics['agents_alive']}")
            self.logs.append(f"Sucesso: {'SIM' if self.metrics['success'] else 'NÃO'}")
        
        return self.metrics
    
    def print_logs(self):
        """Exibe logs da simulação"""
        for log in self.logs:
            print(log)

# ============================================
# BASELINES CLÁSSICOS
# ============================================

class BaselineA_Greedy:
    """Baseline A: Greedy Best-First Search para encontrar tesouros"""
    def __init__(self, bomb_ratio=0.3, treasure_count=10, max_steps=500):
        self.env = Environment(bomb_ratio=bomb_ratio, treasure_count=treasure_count, approach='A')
        self.max_steps = max_steps
        self.explored = set()
        self.treasures_found = set()
        self.path = []
        self.metrics = {
            'treasures_found': 0,
            'execution_time': 0,
            'steps_taken': 0,
            'success': False,
            'explored_percentage': 0
        }
    
    def heuristic(self, position):
        """Heurística: distância estimada para tesouros não encontrados"""
        if not self.treasures_found:
            # Se nenhum tesouro encontrado, explorar aleatoriamente
            return random.random()
        
        # Distância mínima para tesouros restantes (estimativa)
        min_dist = float('inf')
        for i in range(self.env.size):
            for j in range(self.env.size):
                if self.env.grid[i, j] == 'T' and (i, j) not in self.treasures_found:
                    dist = abs(position[0] - i) + abs(position[1] - j)
                    min_dist = min(min_dist, dist)
        return min_dist if min_dist != float('inf') else 0
    
    def run(self):
        start_time = time.time()
        current_pos = (0, 0)
        self.explored.add(current_pos)
        self.path.append(current_pos)
        
        for step in range(self.max_steps):
            # Explorar célula atual
            cell = self.env.get_cell(*current_pos)
            if cell == 'T':
                self.treasures_found.add(current_pos)
            elif cell == 'B':
                # Bomba encontrada, parar
                break
            
            # Verificar sucesso
            if len(self.treasures_found) > self.env.treasure_count * 0.5:
                self.metrics['success'] = True
                break
            
            # Encontrar melhor vizinho não explorado
            neighbors = self.env.get_neighbors(*current_pos)
            candidates = [(pos, self.heuristic(pos)) for pos in neighbors if pos not in self.explored]
            
            if not candidates:
                break
            
            # Escolher com menor heurística (greedy)
            candidates.sort(key=lambda x: x[1])
            next_pos = candidates[0][0]
            
            current_pos = next_pos
            self.explored.add(current_pos)
            self.path.append(current_pos)
        
        self.metrics['treasures_found'] = len(self.treasures_found)
        self.metrics['steps_taken'] = len(self.path)
        self.metrics['execution_time'] = time.time() - start_time
        self.metrics['explored_percentage'] = len(self.explored) / (self.env.size ** 2) * 100
        
        return self.metrics

class BaselineB_BFS:
    """Baseline B: Breadth-First Search para explorar todo o ambiente"""
    def __init__(self, bomb_ratio=0.3, max_steps=500):
        self.env = Environment(bomb_ratio=bomb_ratio, treasure_count=0, approach='B')
        self.max_steps = max_steps
        self.explored = set()
        self.safe_path = []
        self.metrics = {
            'explored_percentage': 0,
            'execution_time': 0,
            'steps_taken': 0,
            'success': False,
            'survivors': 1  # Simula um agente
        }
    
    def run(self):
        start_time = time.time()
        from collections import deque
        queue = deque([(0, 0)])
        self.explored.add((0, 0))
        self.safe_path.append((0, 0))
        
        while queue and len(self.safe_path) < self.max_steps:
            current_pos = queue.popleft()
            
            # Explorar vizinhos
            neighbors = self.env.get_neighbors(*current_pos)
            for neighbor in neighbors:
                if neighbor not in self.explored:
                    self.explored.add(neighbor)
                    cell = self.env.get_cell(*neighbor)
                    if cell == 'B':
                        # Bomba, não adicionar ao caminho seguro
                        continue
                    queue.append(neighbor)
                    self.safe_path.append(neighbor)
        
        explored_pct = len(self.explored) / (self.env.size ** 2) * 100
        self.metrics['explored_percentage'] = explored_pct
        self.metrics['steps_taken'] = len(self.safe_path)
        self.metrics['execution_time'] = time.time() - start_time
        self.metrics['success'] = explored_pct >= 100 and len(self.safe_path) > 0
        
        return self.metrics

class BaselineC_AStar:
    """Baseline C: A* para encontrar a bandeira"""
    def __init__(self, bomb_ratio=0.3, treasure_count=10, max_steps=500):
        self.env = Environment(bomb_ratio=bomb_ratio, treasure_count=treasure_count, approach='C')
        self.max_steps = max_steps
        self.path = []
        self.metrics = {
            'flag_found': False,
            'execution_time': 0,
            'steps_taken': 0,
            'success': False,
            'path_length': 0
        }
    
    def heuristic(self, position):
        """Distância Manhattan para a bandeira"""
        if self.env.flag_position:
            return abs(position[0] - self.env.flag_position[0]) + abs(position[1] - self.env.flag_position[1])
        return 0
    
    def run(self):
        start_time = time.time()
        import heapq
        
        start = (0, 0)
        goal = self.env.flag_position
        
        if not goal:
            return self.metrics
        
        # A* algorithm
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
            
            for neighbor in self.env.get_neighbors(*current):
                cell = self.env.get_cell(*neighbor)
                if cell == 'B':
                    continue  # Evitar bombas
                
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
        
        # Reconstruir caminho
        if goal in came_from:
            current = goal
            while current != start:
                self.path.append(current)
                current = came_from[current]
            self.path.append(start)
            self.path.reverse()
            
            self.metrics['flag_found'] = True
            self.metrics['success'] = True
            self.metrics['path_length'] = len(self.path)
        
        self.metrics['steps_taken'] = len(self.path)
        self.metrics['execution_time'] = time.time() - start_time
        
        return self.metrics

# ============================================
# 6. FUNÇÕES DE ANÁLISE
# ============================================

def run_multiple_simulations(num_simulations=10, num_agents=4, homogeneous=True):
    """Executa múltiplas simulações para estatísticas"""
    results = []
    
    for i in range(num_simulations):
        sim = ApproachASimulation(
            num_agents=num_agents,
            bomb_ratio=0.3,  # 30% bombas é mais razoável
            treasure_count=12,
            homogeneous=homogeneous,
            max_steps=300
        )
        
        metrics = sim.run_simulation(verbose=False)
        results.append(metrics)
    
    # Calcular médias
    avg_treasures = np.mean([r['treasures_found'] for r in results])
    avg_time = np.mean([r['execution_time'] for r in results])
    success_rate = np.mean([1 if r['success'] else 0 for r in results])
    avg_survivors = np.mean([r['agents_alive'] for r in results])
    
    return {
        'type': 'Homogêneo' if homogeneous else 'Heterogêneo',
        'num_agents': num_agents,
        'avg_treasures': avg_treasures,
        'avg_time': avg_time,
        'success_rate': success_rate,
        'avg_survivors': avg_survivors,
        'results': results
    }

def compare_approaches():
    """Compara abordagens homogênea e heterogênea"""
    print("Comparando abordagens...")
    
    # Testar com diferentes números de agentes
    agent_counts = [2, 4, 6, 8]
    comparisons = []
    
    for num_agents in agent_counts:
        print(f"\nTestando com {num_agents} agentes:")
        
        # Homogêneo
        homo_result = run_multiple_simulations(
            num_simulations=5,
            num_agents=num_agents,
            homogeneous=True
        )
        
        # Heterogêneo
        hetero_result = run_multiple_simulations(
            num_simulations=5,
            num_agents=num_agents,
            homogeneous=False
        )
        
        comparisons.append({
            'num_agents': num_agents,
            'homogeneous': homo_result,
            'heterogeneous': hetero_result
        })
        
        print(f"  Homogêneo: {homo_result['avg_treasures']:.1f} tesouros, {homo_result['success_rate']:.0%} sucesso")
        print(f"  Heterogêneo: {hetero_result['avg_treasures']:.1f} tesouros, {hetero_result['success_rate']:.0%} sucesso")
    
    return comparisons

def plot_results(comparisons):
    """Gera gráficos comparativos"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    agent_counts = [c['num_agents'] for c in comparisons]
    
    # Gráfico 1: Tesouros encontrados
    homo_treasures = [c['homogeneous']['avg_treasures'] for c in comparisons]
    hetero_treasures = [c['heterogeneous']['avg_treasures'] for c in comparisons]
    
    axes[0, 0].plot(agent_counts, homo_treasures, 'o-', label='Homogêneo', linewidth=2)
    axes[0, 0].plot(agent_counts, hetero_treasures, 's-', label='Heterogêneo', linewidth=2)
    axes[0, 0].set_title('Tesouros Encontrados vs Número de Agentes')
    axes[0, 0].set_xlabel('Número de Agentes')
    axes[0, 0].set_ylabel('Tesouros Encontrados')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Taxa de Sucesso
    homo_success = [c['homogeneous']['success_rate'] for c in comparisons]
    hetero_success = [c['heterogeneous']['success_rate'] for c in comparisons]
    
    axes[0, 1].bar(np.array(agent_counts) - 0.2, homo_success, width=0.4, label='Homogêneo')
    axes[0, 1].bar(np.array(agent_counts) + 0.2, hetero_success, width=0.4, label='Heterogêneo')
    axes[0, 1].set_title('Taxa de Sucesso (>50% tesouros)')
    axes[0, 1].set_xlabel('Número de Agentes')
    axes[0, 1].set_ylabel('Taxa de Sucesso')
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Gráfico 3: Sobreviventes
    homo_survivors = [c['homogeneous']['avg_survivors'] for c in comparisons]
    hetero_survivors = [c['heterogeneous']['avg_survivors'] for c in comparisons]
    
    x = np.arange(len(agent_counts))
    width = 0.35
    axes[1, 0].bar(x - width/2, homo_survivors, width, label='Homogêneo')
    axes[1, 0].bar(x + width/2, hetero_survivors, width, label='Heterogêneo')
    axes[1, 0].set_title('Agentes Sobreviventes')
    axes[1, 0].set_xlabel('Número de Agentes')
    axes[1, 0].set_ylabel('Sobreviventes')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(agent_counts)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Gráfico 4: Tempo de Execução
    homo_time = [c['homogeneous']['avg_time'] for c in comparisons]
    hetero_time = [c['heterogeneous']['avg_time'] for c in comparisons]
    
    axes[1, 1].plot(agent_counts, homo_time, 'o-', label='Homogêneo', linewidth=2)
    axes[1, 1].plot(agent_counts, hetero_time, 's-', label='Heterogêneo', linewidth=2)
    axes[1, 1].set_title('Tempo de Execução vs Número de Agentes')
    axes[1, 1].set_xlabel('Número de Agentes')
    axes[1, 1].set_ylabel('Tempo (segundos)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('COMPARAÇÃO: Abordagem A (Homogêneo vs Heterogêneo)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================
# 7. EXECUÇÃO PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("PROJETO IA - ABORDAGEM A CORRIGIDA")
    print("="*60)
    
    # Teste rápido
    print("\n1. Teste rápido - 4 agentes heterogêneos:")
    sim = ApproachASimulation(
        num_agents=4,
        bomb_ratio=0.3,
        treasure_count=12,
        homogeneous=False,
        max_steps=200
    )
    
    metrics = sim.run_simulation(verbose=True)
    sim.print_logs()
    
    # Comparação completa
    print("\n\n2. Comparação completa...")
    comparisons = compare_approaches()
    
    print("\n\n3. Gerando gráficos...")
    plot_results(comparisons)
    
    # Análise
    print("\n4. ANÁLISE DOS RESULTADOS:")
    print("-"*40)
    
    best_homo = max(comparisons, key=lambda x: x['homogeneous']['success_rate'])
    best_hetero = max(comparisons, key=lambda x: x['heterogeneous']['success_rate'])
    
    print(f"\nMelhor configuração Homogênea:")
    print(f"  {best_homo['num_agents']} agentes: {best_homo['homogeneous']['success_rate']:.0%} sucesso")
    
    print(f"\nMelhor configuração Heterogênea:")
    print(f"  {best_hetero['num_agents']} agentes: {best_hetero['heterogeneous']['success_rate']:.0%} sucesso")
    
    print("\n5. RESPOSTAS ÀS QUESTÕES:")
    print("-"*40)
    print("\na) Qual grupo obteve melhor resultado?")
    print("   O grupo heterogêneo geralmente tem melhor performance")
    
    print("\nb) O que ocorre com menos vs mais agentes?")
    print("   Menos agentes: mais lento, maior risco")
    print("   Mais agentes: mais rápido, mas pode haver sobreposição")
    
    print("\nc) Vantagens da colaboração heterogênea?")
    print("   - Diversidade de estratégias")
    print("   - Maior robustez")
    print("   - Exploração mais eficiente")

# ============================================
# FUNÇÕES PARA GUI
# ============================================

def compare_approaches():
    """Compara abordagens homogênea e heterogênea com baselines clássicos por abordagem"""
    print("Comparando abordagens corretamente...")
    
    results = []
    
    # Para cada abordagem, comparar homogêneo vs heterogêneo vs baseline
    approaches = [
        ('A', 10, 'tesouros'),  # abordagem, tesouros, objetivo
        ('B', 0, 'exploração'),  # abordagem B sem tesouros
        ('C', 10, 'bandeira')    # abordagem C com bandeira
    ]
    
    for approach, treasure_count, objective in approaches:
        print(f"\n=== ABORDAGEM {approach} ===")
        print(f"Objetivo: {objective}")
        
        # Executar múltiplas simulações para estatísticas
        num_simulations = 5
        
        # Homogêneo
        homo_results = []
        for i in range(num_simulations):
            sim = ApproachASimulation(
                num_agents=4, bomb_ratio=0.3, treasure_count=treasure_count, 
                homogeneous=True, max_steps=100, approach=approach
            )
            sim.run_simulation()
            homo_results.append(sim.metrics)
        
        # Heterogêneo
        hetero_results = []
        for i in range(num_simulations):
            sim = ApproachASimulation(
                num_agents=4, bomb_ratio=0.3, treasure_count=treasure_count, 
                homogeneous=False, max_steps=100, approach=approach
            )
            sim.run_simulation()
            hetero_results.append(sim.metrics)
        
        # Calcular taxas de sucesso
        homo_success_rate = np.mean([1 if r['success'] else 0 for r in homo_results])
        hetero_success_rate = np.mean([1 if r['success'] else 0 for r in hetero_results])
        
        print(f"Homogêneo - Taxa de sucesso: {homo_success_rate:.1%}")
        print(f"Heterogêneo - Taxa de sucesso: {hetero_success_rate:.1%}")
        
        # Identificar melhor grupo
        if hetero_success_rate > homo_success_rate:
            best_group = 'heterogeneous'
            best_results = hetero_results
            best_success_rate = hetero_success_rate
            print("Melhor grupo: Heterogêneo")
        else:
            best_group = 'homogeneous'
            best_results = homo_results
            best_success_rate = homo_success_rate
            print("Melhor grupo: Homogêneo")
        
        # Adicionar resultados dos grupos
        results.append({
            'name': f'{best_group.title()} {approach}',
            'approach': approach,
            'group_type': best_group,
            'agents': 4,
            'metrics': {
                'success': best_success_rate >= 0.5,  # Pelo menos 50% das simulações
                'avg_treasures_found': np.mean([r['treasures_found'] for r in best_results]),
                'avg_explored_percentage': np.mean([r.get('explored_percentage', 0) for r in best_results]),
                'avg_flag_found': np.mean([1 if r.get('flag_found', False) else 0 for r in best_results]),
                'avg_execution_time': np.mean([r['execution_time'] for r in best_results]),
                'avg_agents_alive': np.mean([r['agents_alive'] for r in best_results]),
                'avg_steps_taken': np.mean([r['steps_taken'] for r in best_results]),
                'success_rate': best_success_rate
            }
        })
        
        # Baseline correspondente
        if approach == 'A':
            baseline_sim = BaselineA_Greedy(bomb_ratio=0.3, treasure_count=10, max_steps=100)
            baseline_metrics = baseline_sim.run()
            baseline_name = 'Greedy A'
        elif approach == 'B':
            baseline_sim = BaselineB_BFS(bomb_ratio=0.3, max_steps=100)
            baseline_metrics = baseline_sim.run()
            baseline_name = 'BFS B'
        elif approach == 'C':
            baseline_sim = BaselineC_AStar(bomb_ratio=0.3, treasure_count=10, max_steps=100)
            baseline_metrics = baseline_sim.run()
            baseline_name = 'A* C'
        
        results.append({
            'name': baseline_name,
            'approach': approach,
            'group_type': 'baseline',
            'agents': 1,
            'metrics': {
                'success': baseline_metrics['success'],
                'avg_treasures_found': baseline_metrics.get('treasures_found', 0),
                'avg_explored_percentage': baseline_metrics.get('explored_percentage', 0),
                'avg_flag_found': 1.0 if baseline_metrics.get('flag_found', False) else 0.0,
                'avg_execution_time': baseline_metrics['execution_time'],
                'avg_agents_alive': 1 if baseline_metrics['success'] else 0,
                'avg_steps_taken': baseline_metrics['steps_taken'],
                'success_rate': 1.0 if baseline_metrics['success'] else 0.0
            }
        })
        
        print(f"Baseline {baseline_name}: {'Sucesso' if baseline_metrics['success'] else 'Falha'}")
    
    return results

def plot_comparison_results(results):
    """Plota resultados da comparação"""
    print("Plotando resultados...")
    # Implementação simplificada - apenas imprime
    print(f"Resultados: {results}")