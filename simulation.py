import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import time
from collections import deque
import heapq
from datetime import datetime

# Cores para agentes na visualização
AGENT_COLORS = ['🔴', '🔵', '🟡', '🟢', '🟣', '🟠', '⚪', '🟤', '🟥', '🟦']

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
        self.shared_knowledge = {}  # {(x,y): 'L'/'B'/'T'/'F'}
        self.collect_states = False
        
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
        """Decide próximo movimento usando ensemble de modelos com conhecimento compartilhado"""
        if not self.alive:
            return None
        
        # ⭐ NOVO: Coletar todos os movimentos possíveis, incluindo células exploradas
        # se não houver células não exploradas disponíveis
        possible_moves = []
        unexplored_moves = []
        
        for nx, ny in environment.get_neighbors(*self.position):
            if not environment.explored[nx, ny]:
                unexplored_moves.append((nx, ny))
            else:
                # Permitir revisitar células exploradas como último recurso
                possible_moves.append((nx, ny))
        
        # ⭐ Priorizar células não exploradas
        if unexplored_moves:
            possible_moves = unexplored_moves
        elif not possible_moves:
            # Nenhum movimento possível
            return None
        
        # Se modelos não estão treinados, escolher aleatoriamente priorizando segurança
        if not any(m.is_trained for m in self.models.values()):
            # Priorizar movimento para células que já sabemos ser seguras (shared_knowledge)
            known_safe = []
            for move in possible_moves:
                if move in environment.shared_knowledge:
                    cell_info = environment.shared_knowledge[move]
                    if cell_info in ['L', 'T', 'F']:
                        known_safe.append(move)
                    elif cell_info == 'B' and move in self.deactivated_bombs:
                        known_safe.append(move)  # Bomba desativada por mim é segura
            if known_safe:
                return known_safe[np.random.randint(len(known_safe))]
            # ⭐ NOVO: Se não há células conhecidas como seguras, escolher entre não exploradas
            if unexplored_moves:
                return unexplored_moves[np.random.randint(len(unexplored_moves))]
            return possible_moves[np.random.randint(len(possible_moves))]
        
        # Avaliar cada movimento possível
        move_scores = []
        for move in possible_moves:
            # Bonus de segurança se conhecimento compartilhado indica célula segura
            safety_bonus = 0
            if move in environment.shared_knowledge:
                cell_info = environment.shared_knowledge[move]
                if cell_info in ['L', 'T', 'F']:
                    safety_bonus = 2.0  # Muito seguro (já explorado por outro agente)
                elif cell_info == 'B':
                    # Verifica se ESTE agente desativou esta bomba
                    if move in self.deactivated_bombs:
                        safety_bonus = 2.0  # Seguro para mim (desativei)
                    else:
                        safety_bonus = -2.0  # Perigoso para outros (bomba ativa)
            
            # ⭐ NOVO: Penalizar células já exploradas para priorizar exploração
            if environment.explored[move[0], move[1]]:
                safety_bonus -= 1.0  # Penalidade por revisitar
            
            obs = self.observe(environment, move[0], move[1]).reshape(1, -1)
            
            # Combinar predições dos modelos
            ensemble_score = safety_bonus  # Começar com bonus de segurança
            total_weight = 0
            
            for model_name, model in self.models.items():
                if model.is_trained:
                    proba = model.predict_proba(obs)
                    if len(proba) > 0:
                        # Probabilidade de ser seguro (classe 1)
                        safe_prob = proba[0][1] if proba.shape[1] > 1 else proba[0][0]
                        ensemble_score += self.weights[model_name] * safe_prob
                        total_weight += self.weights[model_name]
            
            if total_weight > 0:
                ensemble_score /= total_weight
            
            move_scores.append(ensemble_score)
        
        # Escolher movimento com maior score
        best_idx = np.argmax(move_scores)
        return possible_moves[best_idx]

    
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
        
        # ⭐ NOVO: Se já estava explorada, verificar se é uma bomba não desativada por este agente
        if already_explored:
            # Se é uma bomba que este agente não desativou, ele morre
            if result == 'B' and target_pos not in self.deactivated_bombs:
                self.alive = False
                self.bombs_activated += 1
                self.update_knowledge(environment, *target_pos, result)
                return result
            # Caso contrário, apenas atualizar conhecimento
            self.update_knowledge(environment, *target_pos, result)
            return result
        
        # Processar resultado (apenas para células recém-exploradas)
        if result == 'B':
            if self.has_treasure_protection:
                # Proteção do tesouro desativa a bomba PARA ESTE AGENTE
                self.has_treasure_protection = False
                self.bombs_activated += 1
                self.bombs_deactivated += 1
                self.deactivated_bombs.add(target_pos)  # Registra bomba desativada individualmente
                # NÃO sobrescreve shared_knowledge - outros agentes ainda veem como 'B'
            else:
                # Agente é destruído
                self.alive = False
                self.bombs_activated += 1
        elif result == 'T':
            self.treasures_found += 1
            self.has_treasure_protection = True
            environment.shared_knowledge[target_pos] = 'T_FOUND'
        
        # Atualizar conhecimento próprio
        self.update_knowledge(environment, *target_pos, result)
        
        return result

class ClassicAlgorithm:
    """Implementação de algoritmos clássicos de busca"""
    
    @staticmethod
    def greedy_best_first(environment, start_pos):
        """Busca Gulosa para Abordagem A"""
        visited = set()
        visited.add(start_pos)
        current = start_pos
        path = [start_pos]
        treasures_found = 0
        bombs_hit = 0
        steps = 0
        
        total_treasures = environment.count_treasures()
        
        while treasures_found <= total_treasures * 0.5 and steps < 100:
            # Heurística: distância até tesouros não descobertos
            best_move = None
            best_score = float('inf')
            
            for nx, ny in environment.get_neighbors(*current):
                if (nx, ny) not in visited:
                    # Estimar valor da célula
                    score = np.random.random()  # Simplificado
                    if score < best_score:
                        best_score = score
                        best_move = (nx, ny)
            
            if best_move is None:
                break
            
            current = best_move
            visited.add(current)
            path.append(current)
            steps += 1
            
            result = environment.mark_explored(*current)
            if result == 'T':
                treasures_found += 1
            elif result == 'B':
                bombs_hit += 1
        
        return {
            'path': path,
            'steps': steps,
            'treasures_found': treasures_found,
            'bombs_activated': bombs_hit,
            'exploration_percentage': environment.get_exploration_percentage()
        }
    
    @staticmethod
    def bfs(environment, start_pos):
        """Busca em Largura para Abordagem B"""
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)
        path = []
        bombs_hit = 0
        steps = 0
        
        while queue and environment.get_exploration_percentage() < 100:
            current = queue.popleft()
            path.append(current)
            steps += 1
            
            result = environment.mark_explored(*current)
            if result == 'B':
                bombs_hit += 1
            
            for nx, ny in environment.get_neighbors(*current):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return {
            'path': path,
            'steps': steps,
            'bombs_activated': bombs_hit,
            'exploration_percentage': environment.get_exploration_percentage()
        }
    
    @staticmethod
    def a_star(environment, start_pos):
        """A* para Abordagem C"""
        # Encontrar posição da bandeira
        flag_pos = None
        for i in range(environment.size):
            for j in range(environment.size):
                if environment.grid[i, j] == 'F':
                    flag_pos = (i, j)
                    break
            if flag_pos:
                break
        
        if flag_pos is None:
            return {'path': [], 'steps': 0, 'found_flag': False}
        
        def heuristic(pos):
            return abs(pos[0] - flag_pos[0]) + abs(pos[1] - flag_pos[1])
        
        open_set = [(heuristic(start_pos), 0, start_pos)]
        came_from = {}
        g_score = {start_pos: 0}
        visited = set()
        bombs_hit = 0
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            visited.add(current)
            result = environment.mark_explored(*current)
            
            if result == 'B':
                bombs_hit += 1
            
            if current == flag_pos:
                # Reconstruir caminho
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_pos)
                path.reverse()
                
                return {
                    'path': path,
                    'steps': len(path),
                    'found_flag': True,
                    'bombs_activated': bombs_hit,
                    'exploration_percentage': environment.get_exploration_percentage()
                }
            
            for neighbor in environment.get_neighbors(*current):
                tentative_g = current_g + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current
        
        return {'path': [], 'steps': 0, 'found_flag': False, 'bombs_activated': bombs_hit}


class Simulation:
    def __init__(self, approach='A', num_agents=2, bomb_ratio=0.5, 
                 group_type='homogeneous'):
        self.approach = approach
        self.num_agents = num_agents
        self.bomb_ratio = bomb_ratio
        self.group_type = group_type
        
        # Criar ambiente
        treasure_count = 10 if approach == 'A' else 0
        self.environment = Environment(bomb_ratio, treasure_count, approach, None)
        
        # ⭐ NOVO: Marcar posição inicial como explorada
        self.environment.mark_explored(0, 0)
        
        # Estado da simulação
        self.collect_states = False
        self.states = []
        
        # Criar agentes
        self.agents = []
        self._create_agents()
        
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
        """Cria agentes homogêneos ou heterogêneos - todos começam em (0,0)"""
        # Todos os agentes começam na posição (0, 0)
        start_position = (0, 0)
        
        for i in range(self.num_agents):
            if self.group_type == 'homogeneous':
                # Todos com mesmos pesos
                weights = {'knn': 1/3, 'naive_bayes': 1/3, 'random_forest': 1/3}
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


def run_experiment(approach, num_agents, bomb_ratio, group_type, repetitions=30):
    results = []
    for rep in range(repetitions):
        sim = Simulation(approach, num_agents, bomb_ratio, group_type)
        metrics = sim.run()
        metrics['repetition'] = rep
        results.append(metrics)
    return pd.DataFrame(results)


def run_baseline(approach, bomb_ratio):
    """Executa algoritmo clássico baseline"""
    treasure_count = 10 if approach == 'A' else 0
    env = Environment(bomb_ratio, treasure_count, approach, None)
    
    start_time = time.time()
    
    if approach == 'A':
        result = ClassicAlgorithm.greedy_best_first(env, (0, 0))
    elif approach == 'B':
        result = ClassicAlgorithm.bfs(env, (0, 0))
    else:  # C
        result = ClassicAlgorithm.a_star(env, (0, 0))
    
    end_time = time.time()
    
    metrics = {
        'approach': approach,
        'group_type': 'baseline',
        'execution_time': end_time - start_time,
        'bomb_ratio': bomb_ratio,
        **result
    }
    
    return metrics


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
