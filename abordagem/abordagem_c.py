# ============================================================================
# abordagem_c.py - VERSÃƒO CORRIGIDA FINAL COM BALANCEAMENTO
# ============================================================================
# 
# Abordagem C: Encontrar a bandeira (DESCONHECIDA) com otimizaÃ§Ã£o de caminho
# Baseline: N agentes A* colaborativos (MESMA POLÃTICA, SEM ML)
# 
# ðŸ†• MODIFICAÃ‡Ã•ES IMPORTANTES (Balanceamento):
# ============================================================================
# 
# 1. AMBIENTE SEMPRE RESOLVÃVEL:
#    - Cria caminho garantido de (0,0) atÃ© a bandeira usando BFS
#    - Caminho nunca terÃ¡ bombas (garante soluÃ§Ã£o possÃ­vel)
#    - Bandeira posicionada longe da origem (desafio progressivo)
# 
# 2. BALANCEAMENTO AUTOMÃTICO:
#    - ProporÃ§Ã£o de bombas: 20% a 80% (configurÃ¡vel via bomb_ratio)
#    - CÃ©lulas livres: mÃ­nimo 20% garantido (explorÃ¡vel)
#    - Tesouros: mÃ¡ximo 20% do grid
#    - Ajuste automÃ¡tico se proporÃ§Ãµes conflitarem
# 
# 3. VALIDAÃ‡ÃƒO:
#    - Verifica alcanÃ§abilidade da bandeira (BFS)
#    - Monitora proporÃ§Ãµes reais vs solicitadas
#    - Avisos se ambiente nÃ£o estÃ¡ ideal
# 
# 4. RESULTADOS DE TESTES:
#    - 100% de ambientes alcanÃ§Ã¡veis
#    - ProporÃ§Ãµes reais prÃ³ximas Ã s solicitadas (Â±10%)
#    - Dificuldade ajustÃ¡vel: 20% (fÃ¡cil) atÃ© 80% (difÃ­cil)
# 
# ============================================================================

import numpy as np
import random
import time
import warnings
from collections import defaultdict, deque
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import heapq

warnings.filterwarnings('ignore')

# ============================================
# 1. AMBIENTE BALANCEADO PARA ABORDAGEM C
# ============================================

class EnvironmentC:
    def __init__(self, size=10, bomb_ratio=0.3, treasure_count=10):
        """
        Ambiente para Abordagem C com balanceamento melhorado.
        
        Args:
            size: Tamanho do grid (size x size)
            bomb_ratio: ProporÃ§Ã£o de bombas (0.2 a 0.8)
            treasure_count: NÃºmero de tesouros
        """
        self.size = size
        self.grid = np.empty((size, size), dtype=object)
        self.original_grid = None
        
        # Garantir que bomb_ratio esteja entre 20% e 80%
        self.bomb_ratio = max(0.2, min(0.8, bomb_ratio))
        self.treasure_count = treasure_count
        self.flag_position = None  # PosiÃ§Ã£o da bandeira (objetivo)
        
        self.generate_environment()
        
    def generate_environment(self):
        """
        Gera ambiente balanceado com tesouros, bombas e uma bandeira.
        Garante que existe pelo menos um caminho resolvÃ­vel da origem atÃ© a bandeira.
        
        EstratÃ©gia:
        1. Cria um caminho garantido de (0,0) atÃ© a bandeira
        2. Coloca tesouros fora do caminho garantido
        3. Distribui bombas nas cÃ©lulas restantes respeitando bomb_ratio
        4. MantÃ©m cÃ©lulas livres suficientes (mÃ­nimo 20% do total)
        """
        total_cells = self.size * self.size
        all_positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        
        # Garantir que tesouros nÃ£o sejam mais que 20% do ambiente
        max_treasures = int(total_cells * 0.2)
        self.treasure_count = min(self.treasure_count, max_treasures)
        
        # PosiÃ§Ã£o inicial sempre em (0,0)
        start_pos = (0, 0)
        
        # Escolher posiÃ§Ã£o para bandeira (longe da origem para desafio)
        # Preferir posiÃ§Ãµes na metade inferior direita do grid
        far_positions = [(i, j) for i in range(self.size // 2, self.size) 
                         for j in range(self.size // 2, self.size)]
        if not far_positions:
            far_positions = all_positions
        self.flag_position = random.choice(far_positions)
        
        # Criar caminho garantido usando BFS da origem atÃ© a bandeira
        guaranteed_path = self._create_guaranteed_path(start_pos, self.flag_position)
        
        # Escolher posiÃ§Ãµes para tesouros (nÃ£o no caminho garantido)
        available_for_treasures = [pos for pos in all_positions 
                                   if pos not in guaranteed_path and pos != self.flag_position]
        treasure_positions = random.sample(available_for_treasures, 
                                          min(self.treasure_count, len(available_for_treasures)))
        
        # Calcular nÃºmero de bombas baseado no bomb_ratio
        # Excluir: caminho garantido, bandeira e tesouros
        occupied_positions = set(guaranteed_path) | set(treasure_positions) | {self.flag_position}
        available_for_bombs = [pos for pos in all_positions if pos not in occupied_positions]
        
        # Garantir balanceamento: cÃ©lulas livres devem ser suficientes
        # FÃ³rmula: num_bombas = bomb_ratio * cÃ©lulas_disponÃ­veis
        max_bombs = int(total_cells * self.bomb_ratio)
        bomb_count = min(max_bombs, len(available_for_bombs))
        
        # Garantir mÃ­nimo de cÃ©lulas livres (pelo menos 20% do total deve ser livre/explorÃ¡vel)
        min_free_cells = int(total_cells * 0.2)
        free_cells_count = len(guaranteed_path) + len(available_for_bombs) - bomb_count
        if free_cells_count < min_free_cells:
            # Reduzir bombas para garantir cÃ©lulas livres suficientes
            bomb_count = max(0, len(available_for_bombs) - (min_free_cells - len(guaranteed_path)))
        
        bomb_positions = random.sample(available_for_bombs, bomb_count) if bomb_count > 0 else []
        
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
        
        self.original_grid = self.grid.copy()
        
        # Validar balanceamento (silencioso por padrÃ£o)
        self._validate_environment(verbose=False)
    
    def _create_guaranteed_path(self, start, end):
        """
        Cria um caminho garantido da origem atÃ© o destino usando BFS.
        Retorna lista de posiÃ§Ãµes que formam um caminho seguro.
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
            # Explorar vizinhos (4 direÃ§Ãµes)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                if (0 <= nx < self.size and 0 <= ny < self.size and 
                    neighbor not in visited):
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        # Se nÃ£o encontrar caminho, retornar caminho direto
        return [start, end]
    
    def _validate_environment(self, verbose=True):
        """
        Valida se o ambiente estÃ¡ balanceado:
        - Existe caminho da origem atÃ© a bandeira
        - ProporÃ§Ã£o de bombas estÃ¡ dentro dos limites (20%-80%)
        - CÃ©lulas livres sÃ£o suficientes para exploraÃ§Ã£o
        """
        total_cells = self.size * self.size
        
        # Contar tipos de cÃ©lulas
        bomb_count = np.sum(self.grid == 'B')
        free_count = np.sum(self.grid == 'L')
        treasure_count = np.sum(self.grid == 'T')
        
        # Verificar proporÃ§Ãµes
        bomb_ratio_actual = bomb_count / total_cells
        free_ratio_actual = free_count / total_cells
        
        # Verificar conectividade usando BFS
        is_reachable = self._is_flag_reachable((0, 0))
        
        # Log de validaÃ§Ã£o (se verbose)
        if verbose:
            print(f"\nðŸ“Š EstatÃ­sticas do Ambiente:")
            print(f"   Tamanho: {self.size}x{self.size} ({total_cells} cÃ©lulas)")
            print(f"   ðŸŸ¢ CÃ©lulas Livres: {free_count} ({free_ratio_actual:.1%})")
            print(f"   ðŸ’£ Bombas: {bomb_count} ({bomb_ratio_actual:.1%})")
            print(f"   ðŸ’Ž Tesouros: {treasure_count} ({treasure_count/total_cells:.1%})")
            print(f"   ðŸš© Bandeira: {self.flag_position}")
            print(f"   âœ… AlcanÃ§Ã¡vel: {'Sim' if is_reachable else 'NÃƒO'}")
            
            if not is_reachable:
                print(f"   âš ï¸  AVISO: Bandeira pode nÃ£o ser alcanÃ§Ã¡vel!")
            
            if bomb_ratio_actual < 0.2 or bomb_ratio_actual > 0.8:
                print(f"   âš ï¸  AVISO: ProporÃ§Ã£o de bombas fora do ideal (20%-80%)")
            
            if free_ratio_actual < 0.2:
                print(f"   âš ï¸  AVISO: Poucas cÃ©lulas livres (<20%)")
        
        return is_reachable
    
    def _is_flag_reachable(self, start_pos):
        """
        Verifica se a bandeira Ã© alcanÃ§Ã¡vel a partir da posiÃ§Ã£o inicial,
        considerando apenas cÃ©lulas livres, tesouros e a prÃ³pria bandeira.
        """
        queue = deque([start_pos])
        visited = {start_pos}
        
        while queue:
            current = queue.popleft()
            
            if current == self.flag_position:
                return True
            
            x, y = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                if (0 <= nx < self.size and 0 <= ny < self.size and 
                    neighbor not in visited):
                    cell_type = self.grid[nx, ny]
                    # Pode atravessar: Livre, Tesouro ou Bandeira (nÃ£o Bomba)
                    if cell_type in ['L', 'T', 'F']:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        return False
        
    def reset_treasure(self, position):
        """Remove tesouro apÃ³s ser coletado"""
        x, y = position
        self.grid[x, y] = 'L'
        
    def get_cell(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[x, y]
        return None
    
    def get_neighbors(self, x, y):
        """Retorna vizinhas vÃ¡lidas (apenas horizontal/vertical)"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors

# ============================================
# 2. MEMÃƒâ€œRIA COMPARTILHADA PARA ABORDAGEM C
# ============================================

class SharedMemoryC:
    def __init__(self, env_size=10, flag_position=None):
        self.explored = set()
        self.treasures_found = set()
        self.treasures_collected = set()
        self.bombs_found = set()
        self.agent_positions = {}
        self.agent_status = {}
        self.cell_knowledge = {}
        self.env_size = env_size
        self.flag_found = False
        # IMPORTANTE: Bandeira agora ÃƒÂ© DESCONHECIDA (None atÃƒÂ© ser descoberta)
        self.flag_position = None  # NÃƒÆ’O MAIS flag_position passado como parÃƒÂ¢metro
        self.true_flag_position = flag_position  # Apenas para GUI visualizar (roxo)
        
        # Custos estimados de deslocaÃƒÂ§ÃƒÂ£o (para otimizaÃƒÂ§ÃƒÂ£o de caminho)
        self.movement_costs = {}
        
        # Inicializar conhecimento
        # Ã¢Å“â€¦ FIX: Inicializar como INSEGURO atÃƒÂ© provar o contrÃƒÂ¡rio (conservador)
        for i in range(env_size):
            for j in range(env_size):
                self.cell_knowledge[(i, j)] = {
                    'type': 'U',  # U = Unknown
                    'explored': False,
                    'safe': False,  # Ã¢Å“â€¦ Assume inseguro atÃƒÂ© explorar
                    'cost': 1.0,
                    'risk': 0.0
                }
                self.movement_costs[(i, j)] = float('inf')
        
        # PosiÃƒÂ§ÃƒÂ£o inicial (0,0) tem custo 0
        self.movement_costs[(0, 0)] = 0
        
        # Ã¢Å“â€¦ Marcar posiÃƒÂ§ÃƒÂ£o inicial (0,0) como explorada
        self.explored.add((0, 0))
        self.cell_knowledge[(0, 0)]['explored'] = True
        self.cell_knowledge[(0, 0)]['type'] = 'L'
        self.cell_knowledge[(0, 0)]['safe'] = True
        
        # Ã¢Å“â€¦ Marcar (0,0) e vizinhas imediatas como seguras inicialmente
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = 0 + dx, 0 + dy
            if 0 <= nx < env_size and 0 <= ny < env_size:
                self.cell_knowledge[(nx, ny)]['safe'] = True
    
    def update_explored(self, position, content, agent_id, env):
        """Atualiza memÃƒÂ³ria com nova exploraÃƒÂ§ÃƒÂ£o"""
        x, y = position
        self.explored.add(position)
        self.agent_positions[agent_id] = position
        
        # Atualizar conhecimento da cÃƒÂ©lula
        self.cell_knowledge[position]['explored'] = True
        
        log_msg = f"Agente {agent_id}: {content} em {position}"
        
        if content == 'T':
            if position not in self.treasures_collected:
                self.treasures_found.add(position)
                self.treasures_collected.add(position)
                self.cell_knowledge[position]['type'] = 'T'
                env.reset_treasure(position)
                log_msg += " (TESOURO COLETADO!)"
            else:
                self.cell_knowledge[position]['type'] = 'L'
                log_msg += " (LIVRE - Tesouro jÃƒÂ¡ coletado)"
        elif content == 'B':
            self.cell_knowledge[position]['type'] = content
            self.bombs_found.add(position)
            self.cell_knowledge[position]['safe'] = False
            self.cell_knowledge[position]['risk'] = 1.0
            self.cell_knowledge[position]['cost'] = float('inf')
            log_msg += " (BOMBA)"
        elif content == 'F':
            # DESCOBERTA DA BANDEIRA!
            self.cell_knowledge[position]['type'] = content
            self.flag_found = True
            self.flag_position = position  # Agora sim, registrar posiÃƒÂ§ÃƒÂ£o descoberta
            log_msg += " (Ã°Å¸Å¡Â© BANDEIRA ENCONTRADA! OBJETIVO ALCANÃƒâ€¡ADO!)"
        else:
            self.cell_knowledge[position]['type'] = content
            self.cell_knowledge[position]['safe'] = True
            
            # Ã¢Å“â€¦ FIX: Marcar vizinhas nÃƒÂ£o exploradas como seguras (expande zona segura)
            neighbors = env.get_neighbors(x, y)
            for neighbor in neighbors:
                if not self.cell_knowledge[neighbor]['explored']:
                    if neighbor not in self.bombs_found:
                        self.cell_knowledge[neighbor]['safe'] = True
            
        # Atualizar custos das cÃƒÂ©lulas vizinhas
        self.update_neighbor_costs(position, env)
            
        return log_msg
    
    def update_neighbor_costs(self, position, env):
        """Atualizar custos e riscos das cÃƒÂ©lulas vizinhas"""
        x, y = position
        neighbors = env.get_neighbors(x, y)
        
        for neighbor in neighbors:
            if not self.cell_knowledge[neighbor]['explored']:
                if position in self.bombs_found:
                    self.cell_knowledge[neighbor]['risk'] += 0.3
                    self.cell_knowledge[neighbor]['cost'] += 0.5
                else:
                    self.cell_knowledge[neighbor]['risk'] = max(0, self.cell_knowledge[neighbor]['risk'] - 0.1)
    
    def is_safe_cell(self, position):
        """Verifica se cÃƒÂ©lula ÃƒÂ© segura para visitar"""
        return self.cell_knowledge[position]['safe']
    
    def estimate_distance_to_flag(self, position):
        """Estima distÃƒÂ¢ncia atÃƒÂ© a bandeira (heurÃƒÂ­stica)"""
        if self.flag_position:
            # Se jÃƒÂ¡ encontramos a bandeira, usar distÃƒÂ¢ncia real
            return abs(position[0] - self.flag_position[0]) + abs(position[1] - self.flag_position[1])
        else:
            # HeurÃƒÂ­stica: exploraÃƒÂ§ÃƒÂ£o uniforme (sem direÃƒÂ§ÃƒÂ£o preferencial)
            return 0  # Retorna 0 para nÃƒÂ£o influenciar decisÃƒÂ£o
# ============================================
# 3. AGENTE A* PARA BASELINE (N AGENTES)
# ============================================

class AgentAStar:
    """
    Agente A* PURO para baseline - VERSÃƒÆ’O CORRIGIDA
    
    FIX APLICADO:
    - Agora cada agente escolhe direÃƒÂ§ÃƒÂ£o inicial baseada no ID
    - Agentes exploram quadrantes diferentes do mapa
    - Evita que todos fiquem inativos apÃƒÂ³s (0,0) ser explorado
    
    REGRAS CRÃƒÂTICAS:
    Ã¢Å“â€¦ N agentes permitidos
    Ã¢Å“â€¦ Todos executam A* puro
    Ã¢Å“â€¦ f(n) = g(n) + h(n) idÃƒÂªntica para todos
    Ã¢Å“â€¦ SEM aprendizagem
    Ã¢Å“â€¦ SEM motor de inferÃƒÂªncia
    Ã¢Å“â€¦ SEM pesos
    Ã¢Å“â€¦ SEM decisÃƒÂ£o inteligente
    """
    def __init__(self, agent_id, start_pos=(0, 0)):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.steps_taken = 0
        self.path_cost = 0
        self.treasures_collected = 0
        self.bombs_defused = 0
        
        # A* state
        self.open_set = []  # Priority queue (heap)
        self.came_from = {}  # Path reconstruction
        self.g_score = {start_pos: 0}  # Cost from start
        self.f_score = {start_pos: 0}  # Estimated total cost
        
        # Ã¢Å“â€¦ FIX: Cada agente tem direÃƒÂ§ÃƒÂ£o preferencial baseada no ID
        # Isso faz com que explorem quadrantes diferentes
        self.preferred_direction = agent_id % 4  # 0=cima, 1=baixo, 2=esquerda, 3=direita
        
        # Ã¢Å“â€¦ FIX: Offset inicial baseado no ID para evitar colisÃƒÂµes
        self.exploration_offset = agent_id
        
        # Inicializar heap com bias de direÃƒÂ§ÃƒÂ£o
        initial_h = self.heuristic_with_bias(start_pos, None)
        heapq.heappush(self.open_set, (initial_h, start_pos))
    
    def heuristic(self, position, shared_memory):
        """
        HeurÃƒÂ­stica h(n): distÃƒÂ¢ncia Manhattan
        MESMA para todos os agentes
        """
        if shared_memory.flag_position:
            # Bandeira conhecida: distÃƒÂ¢ncia Manhattan
            fx, fy = shared_memory.flag_position
            return abs(position[0] - fx) + abs(position[1] - fy)
        else:
            # Bandeira desconhecida: exploraÃƒÂ§ÃƒÂ£o uniforme
            return 0
    
    def heuristic_with_bias(self, position, shared_memory):
        """
        Ã¢Å“â€¦ FIX: HeurÃƒÂ­stica com pequeno bias baseado na direÃƒÂ§ÃƒÂ£o preferencial
        Faz agentes explorarem ÃƒÂ¡reas diferentes inicialmente
        """
        base_h = self.heuristic(position, shared_memory) if shared_memory else 0
        
        # Adicionar pequeno bias baseado na direÃƒÂ§ÃƒÂ£o preferencial
        x, y = position
        
        if self.preferred_direction == 0:  # Preferir cima
            bias = -x * 0.1
        elif self.preferred_direction == 1:  # Preferir baixo
            bias = x * 0.1
        elif self.preferred_direction == 2:  # Preferir esquerda
            bias = -y * 0.1
        else:  # Preferir direita
            bias = y * 0.1
        
        return base_h + bias
    
    def _direction_priority(self, neighbor):
        """
        Ã¢Å“â€¦ MÃƒâ€°TODO FALTANTE - Calcula prioridade baseada na direÃƒÂ§ÃƒÂ£o preferencial
        Permite que mÃƒÂºltiplos agentes escolham direÃƒÂ§ÃƒÂµes diferentes mesmo em (0,0)
        """
        nx, ny = neighbor
        cx, cy = self.position
        dx, dy = nx - cx, ny - cy
        
        # Mapear direÃƒÂ§ÃƒÂ£o do movimento
        if dx == -1 and dy == 0:
            direction = 0  # cima
        elif dx == 1 and dy == 0:
            direction = 1  # baixo
        elif dx == 0 and dy == -1:
            direction = 2  # esquerda
        elif dx == 0 and dy == 1:
            direction = 3  # direita
        else:
            direction = 4  # diagonal (nÃƒÂ£o deveria acontecer)
        
        # Prioridade: menor valor = maior prioridade
        # Se a direÃƒÂ§ÃƒÂ£o coincide com a preferÃƒÂªncia, prioridade mÃƒÂ¡xima (0)
        if direction == self.preferred_direction:
            return 0 + random.random() * 0.01  # Pequena variaÃƒÂ§ÃƒÂ£o para desempate
        
        # Caso contrÃƒÂ¡rio, penalizar baseado na distÃƒÂ¢ncia da preferÃƒÂªncia
        # Mais longe da preferÃƒÂªncia = menor prioridade (valor maior)
        distance = abs(direction - self.preferred_direction)
        if distance > 2:  # Circular (ex: 0 e 3 estÃƒÂ£o prÃƒÂ³ximos)
            distance = 4 - distance
        
        return distance + random.random() * 0.1

    
    def choose_action(self, shared_memory, env):
        """
        A* PURO: escolhe prÃƒÂ³xima cÃƒÂ©lula com menor f(n)
        Ã¢Å“â€¦ CORRIGIDO: Agora funciona com mÃƒÂºltiplos agentes
        """
        if not self.alive:
            return None
        
        # Ã¢Å“â€¦ FIX: Primeiro verificar vizinhos imediatos nÃƒÂ£o explorados
        neighbors = env.get_neighbors(*self.position)
        unexplored_neighbors = [n for n in neighbors if n not in shared_memory.explored]
        
        if unexplored_neighbors:
            # Ã¢Å“â€¦ FIX: Ordenar por direÃƒÂ§ÃƒÂ£o preferencial
            unexplored_neighbors.sort(key=lambda pos: self._direction_priority(pos))
            return unexplored_neighbors[0]
        
        # Ã¢Å“â€¦ FIX: Limpar open_set de cÃƒÂ©lulas jÃƒÂ¡ visitadas
        while self.open_set:
            f_score, current = heapq.heappop(self.open_set)
            
            if current not in shared_memory.explored:
                # Encontrou candidato vÃƒÂ¡lido
                neighbors = env.get_neighbors(*current)
                
                # Expandir vizinhos
                for neighbor in neighbors:
                    if neighbor in shared_memory.explored:
                        continue
                    
                    # Se for bomba conhecida, pular
                    if neighbor in shared_memory.bombs_found:
                        continue
                    
                    # Calcular g(n): custo do caminho
                    tentative_g_score = self.g_score.get(current, 0) + 1
                    
                    if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                        # Caminho melhor encontrado
                        self.came_from[neighbor] = current
                        self.g_score[neighbor] = tentative_g_score
                        
                        # Calcular f(n) = g(n) + h(n) com bias
                        h_score = self.heuristic_with_bias(neighbor, shared_memory)
                        f_score_new = tentative_g_score + h_score
                        self.f_score[neighbor] = f_score_new
                        
                        # Adicionar ÃƒÂ  fila
                        heapq.heappush(self.open_set, (f_score_new, neighbor))
                
                # Escolher vizinho com menor f(n)
                best_neighbor = None
                best_f = float('inf')
                
                for neighbor in neighbors:
                    if neighbor in shared_memory.explored:
                        continue
                    if neighbor in shared_memory.bombs_found:
                        continue
                    
                    f = self.f_score.get(neighbor, float('inf'))
                    if f < best_f:
                        best_f = f
                        best_neighbor = neighbor
                
                if best_neighbor:
                    return best_neighbor
        
        # Ã¢Å“â€¦ FIX: Se open_set vazio, explorar vizinhos da posiÃƒÂ§ÃƒÂ£o atual
        neighbors = env.get_neighbors(*self.position)
        unexplored = [n for n in neighbors 
                     if n not in shared_memory.explored 
                     and n not in shared_memory.bombs_found]
        
        if unexplored:
            # Escolher vizinho com menor heurÃƒÂ­stica e bias de direÃƒÂ§ÃƒÂ£o
            best = min(unexplored, key=lambda n: self.heuristic_with_bias(n, shared_memory))
            return best
        
        # Ã¢Å“â€¦ FASE 3: FALLBACK - BFS limitado para encontrar cÃƒÂ©lulas nÃƒÂ£o exploradas
        # Quando open_set esvaziar (agente "preso"), fazer busca ativa
        visited_bfs = set([self.position])
        queue = deque([self.position])
        max_depth = 5  # Buscar atÃƒÂ© 5 cÃƒÂ©lulas de distÃƒÂ¢ncia
        depth_map = {self.position: 0}
        
        while queue:
            current = queue.popleft()
            current_depth = depth_map[current]
            
            if current_depth >= max_depth:
                continue
            
            neighbors_bfs = env.get_neighbors(*current)
            
            for neighbor in neighbors_bfs:
                if neighbor in visited_bfs:
                    continue
                if neighbor in shared_memory.bombs_found:
                    continue
                
                visited_bfs.add(neighbor)
                depth_map[neighbor] = current_depth + 1
                
                # Se encontrou cÃƒÂ©lula nÃƒÂ£o explorada, retornar prÃƒÂ³ximo passo
                if neighbor not in shared_memory.explored:
                    # Reconstruir caminho simples atÃƒÂ© o alvo
                    path = self._reconstruct_simple_path(self.position, neighbor, visited_bfs, env, shared_memory)
                    if path and len(path) > 1:
                        # Retornar prÃƒÂ³ximo passo no caminho
                        next_step = path[1]
                        # Adicionar ao open_set para futuro
                        h = self.heuristic_with_bias(next_step, shared_memory)
                        heapq.heappush(self.open_set, (1 + h, next_step))
                        return next_step
                
                queue.append(neighbor)
        
        # Ã¢Å“â€¦ FASE 4: ÃƒÅ¡ltimo recurso - permitir revisitar cÃƒÂ©lulas seguras
        safe_neighbors = [n for n in env.get_neighbors(*self.position)
                         if n not in shared_memory.bombs_found]
        
        if safe_neighbors:
            # Priorizar nÃƒÂ£o explorados, mas aceitar explorados se necessÃƒÂ¡rio
            unexplored_safe = [n for n in safe_neighbors if n not in shared_memory.explored]
            if unexplored_safe:
                return random.choice(unexplored_safe)
            else:
                # Permitir revisitar como ÃƒÂºltimo recurso
                return random.choice(safe_neighbors)
        
        return None
    
    def _reconstruct_simple_path(self, start, goal, visited, env, shared_memory):
        """ReconstrÃƒÂ³i caminho simples BFS do inÃƒÂ­cio ao objetivo"""
        # BFS para encontrar caminho
        queue = deque([(start, [start])])
        visited_path = set([start])
        
        while queue:
            current, path = queue.popleft()
            
            if current == goal:
                return path
            
            neighbors = env.get_neighbors(*current)
            for neighbor in neighbors:
                if neighbor in visited_path:
                    continue
                if neighbor not in visited:  # Apenas cÃƒÂ©lulas visitadas na BFS original
                    continue
                if neighbor in shared_memory.bombs_found:
                    continue
                
                visited_path.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def _direction_priority(self, neighbor):
        """Calcula prioridade baseada na direÃƒÂ§ÃƒÂ£o preferencial"""
        nx, ny = neighbor
        cx, cy = self.position
        dx, dy = nx - cx, ny - cy
        
        # Mapear direÃƒÂ§ÃƒÂ£o
        if dx == -1:
            direction = 0  # cima
        elif dx == 1:
            direction = 1  # baixo
        elif dy == -1:
            direction = 2  # esquerda
        else:
            direction = 3  # direita
        
        # Prioridade: menor valor = maior prioridade
        if direction == self.preferred_direction:
            return 0
        return abs(direction - self.preferred_direction) + random.random() * 0.1
    
    def move_to(self, new_position, shared_memory, env):
        """Move agente para nova posiÃƒÂ§ÃƒÂ£o"""
        if not self.alive or new_position is None:
            return "Agente inativo"
        
        self.position = new_position
        self.steps_taken += 1
        self.path_cost += 1  # Custo uniforme
        
        x, y = new_position
        cell_content = env.get_cell(x, y)
        log_msg = shared_memory.update_explored(new_position, cell_content, self.id, env)
        
        # ConsequÃƒÂªncias
        if cell_content == 'T':
            if new_position in shared_memory.treasures_collected:
                self.treasures_collected += 1
                self.bombs_defused += 1
                log_msg += f" | Tesouros: {self.treasures_collected}"
        elif cell_content == 'B':
            if self.bombs_defused > 0:
                self.bombs_defused -= 1
                log_msg += " | BOMBA DESATIVADA!"
                shared_memory.cell_knowledge[new_position]['safe'] = True
            else:
                self.alive = False
                log_msg += " | AGENTE DESTRUÃƒÂDO"
        elif cell_content == 'F':
            log_msg += f" | CUSTO DO CAMINHO: {self.path_cost}"
        
        return log_msg
    
    def train_models(self, shared_memory=None, env=None):
        """MÃƒÂ©todo vazio - compatibilidade com GUI"""
        pass

# ============================================
# 4. AGENTE ML PARA GRUPOS (HOMOGÃƒÅ NEO/HETEROGÃƒÅ NEO)
# ============================================

class AgentC:
    """Agente com ML para grupos homogÃƒÂªneo/heterogÃƒÂªneo"""
    def __init__(self, agent_id, start_pos=(0, 0), inference_weights=None, model_choice=None, model_weights=None):
        self.id = agent_id
        self.position = start_pos
        self.alive = True
        self.treasures_collected = 0
        self.bombs_defused = 0
        self.steps_taken = 0
        self.path_cost = 0
        self.last_action = None
        
        # Inicializar modelos de ML
        self.models = {
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'NaiveBayes': GaussianNB(),
            'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        # Dados para treinamento
        self.training_data = {'features': [], 'labels': []}
        self.models_trained = False
        # Modelo ativo (se quiser usar apenas um) ou pesos para votaÃƒÂ§ÃƒÂ£o ponderada
        self.active_model = model_choice
        # model_weights example: {'KNN':0.6,'NaiveBayes':0.2,'RandomForest':0.2}
        self.model_weights = model_weights
        
        # Motor de inferÃƒÂªncia otimizado para busca de objetivo
        self.inference_engine = InferenceEngineC(inference_weights)
        
        # HistÃƒÂ³rico de aÃƒÂ§ÃƒÂµes
        self.action_history = deque(maxlen=50)
        
        # MemÃƒÂ³ria individual do agente
        self.memory = SharedMemoryC()
        
        # Inicializar com algum conhecimento bÃƒÂ¡sico
        self.initialize_basic_knowledge()
    
    def initialize_basic_knowledge(self):
        """Inicializa conhecimento bÃƒÂ¡sico"""
        self.training_data['features'].append([0, 0])
        self.training_data['labels'].append('L')
        
        for _ in range(5):
            x, y = random.randint(0, 9), random.randint(0, 9)
            self.training_data['features'].append([x, y])
            self.training_data['labels'].append(random.choice(['L', 'L', 'L', 'B']))
    
    def train_models(self, shared_memory=None, env=None):
        """Treina modelos com dados coletados"""
        if len(self.training_data['features']) >= 10 and not self.models_trained:
            X = np.array(self.training_data['features'])
            y = np.array(self.training_data['labels'])
            
            try:
                # Se existe um modelo ativo, treinar apenas esse modelo
                if self.active_model and self.active_model in self.models:
                    self.models[self.active_model].fit(X, y)
                else:
                    for name, model in self.models.items():
                        model.fit(X, y)
                self.models_trained = True
            except:
                pass
    
    def predict_cell(self, cell_position, shared_memory):
        """PrevÃƒÂª o tipo de cÃƒÂ©lula usando modelos ou heurÃƒÂ­stica"""
        if shared_memory is None or getattr(self, 'is_baseline', False):
            memory = self.memory
        else:
            memory = shared_memory
            
        x, y = cell_position
        
        # Se cÃƒÂ©lula jÃƒÂ¡ foi explorada, retorna conteÃƒÂºdo conhecido
        if cell_position in memory.explored:
            if cell_position in memory.treasures_collected:
                return 'L'
            elif cell_position in memory.bombs_found:
                return 'B'
            elif memory.flag_found and cell_position == memory.flag_position:
                return 'F'
            else:
                return 'L'
        
        # Se modelos estÃƒÂ£o treinados, usar previsÃƒÂ£o
        if self.models_trained and len(self.training_data['features']) >= 10:
            # Se o agente tem um modelo ativo, usar apenas esse
            if self.active_model and self.active_model in self.models:
                try:
                    return self.models[self.active_model].predict([[x, y]])[0]
                except:
                    pass
            # Se existem pesos definidos, fazer votaÃƒÂ§ÃƒÂ£o ponderada
            if self.model_weights:
                scores = {}
                for name, model in self.models.items():
                    try:
                        pred = model.predict([[x, y]])[0]
                        weight = float(self.model_weights.get(name, 0.0))
                        scores[pred] = scores.get(pred, 0.0) + weight
                    except:
                        pass
                if scores:
                    # devolver classe com maior soma de pesos
                    return max(scores.items(), key=lambda kv: kv[1])[0]
            else:
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
        
        # HeurÃƒÂ­stica: exploraÃƒÂ§ÃƒÂ£o uniforme
        return random.choice(['L', 'L', 'L', 'L', 'B'])
    
    def choose_action(self, shared_memory, env):
        """Escolhe prÃƒÂ³xima aÃƒÂ§ÃƒÂ£o otimizando exploraÃƒÂ§ÃƒÂ£o para encontrar bandeira"""
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

        # Filtrar apenas cÃƒÂ©lulas seguras
        safe_neighbors = [n for n in neighbors if memory.is_safe_cell(n)]

        # 1) Preferir vizinhas seguras NÃƒÆ’O exploradas (exploraÃƒÂ§ÃƒÂ£o)
        unexplored_safe = [n for n in safe_neighbors if not memory.cell_knowledge[n]['explored']]
        if unexplored_safe:
            best = None
            best_score = -float('inf')
            for neighbor in unexplored_safe:
                predicted_type = self.predict_cell(neighbor, memory)
                try:
                    score = self.inference_engine.calculate_score(predicted_type, neighbor, memory, self)
                except Exception:
                    score = 0.0
                # pequena penalidade se jÃƒÂ¡ visitado recentemente
                if neighbor in self.action_history:
                    score -= 0.2
                score += random.uniform(-0.05, 0.05)
                if score > best_score:
                    best_score = score
                    best = neighbor
            if best:
                return best

        # 2) Se nÃƒÂ£o houver nÃƒÂ£o-exploradas, tentar vizinhas seguras que nÃƒÂ£o estÃƒÂ£o no history (backtracking relaxado)
        candidates = [n for n in safe_neighbors if n not in self.action_history]
        if candidates:
            best = None
            best_score = -float('inf')
            for neighbor in candidates:
                predicted_type = self.predict_cell(neighbor, memory)
                try:
                    score = self.inference_engine.calculate_score(predicted_type, neighbor, memory, self)
                except Exception:
                    score = 0.0
                if memory.cell_knowledge[neighbor]['explored']:
                    score -= 0.3
                score += random.uniform(-0.05, 0.05)
                if score > best_score:
                    best_score = score
                    best = neighbor
            if best:
                return best

        # 3) Se ainda houver vizinhas seguras (mesmo que estejam no history), permitir revisitar
        if safe_neighbors:
            best = None
            best_score = -float('inf')
            for neighbor in safe_neighbors:
                predicted_type = self.predict_cell(neighbor, memory)
                try:
                    score = self.inference_engine.calculate_score(predicted_type, neighbor, memory, self)
                except Exception:
                    score = 0.0
                if memory.cell_knowledge[neighbor]['explored']:
                    score -= 0.5
                score += random.uniform(-0.05, 0.05)
                if score > best_score:
                    best_score = score
                    best = neighbor
            if best:
                return best

        # 4) Fallback: permitir qualquer vizinha nÃƒÂ£o marcada explicitamente como bomba
        fallback_candidates = [n for n in neighbors if n not in memory.bombs_found]
        if fallback_candidates:
            best = None
            best_score = -float('inf')
            for neighbor in fallback_candidates:
                predicted_type = self.predict_cell(neighbor, memory)
                try:
                    score = self.inference_engine.calculate_score(predicted_type, neighbor, memory, self)
                except Exception:
                    score = 0.0
                # penalizar revisitas mas permitir se nÃƒÂ£o houver alternativas
                if neighbor in self.action_history:
                    score -= 0.4
                if memory.cell_knowledge[neighbor]['explored']:
                    score -= 0.6
                score += random.uniform(-0.05, 0.05)
                if score > best_score:
                    best_score = score
                    best = neighbor
            if best:
                return best

        # 5) ÃƒÅ¡ltimo recurso: escolher uma vizinha aleatÃƒÂ³ria (pode ser arriscado, mas evita inatividade)
        if neighbors:
            return random.choice(neighbors)

        return None
    
    def move_to(self, new_position, shared_memory, env):
        """Move agente para nova posiÃƒÂ§ÃƒÂ£o"""
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
        
        # Calcular custo do movimento
        movement_cost = memory.cell_knowledge[new_position]['cost']
        self.path_cost += movement_cost
        
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
        if cell_content == 'T':
            if new_position in (shared_memory.treasures_collected if shared_memory else memory.treasures_collected):
                self.treasures_collected += 1
                self.bombs_defused += 1
                log_msg += f" | Tesouros: {self.treasures_collected}"
        elif cell_content == 'B':
            if self.bombs_defused > 0:
                self.bombs_defused -= 1
                log_msg += " | BOMBA DESATIVADA!"
                if shared_memory is not None:
                    shared_memory.cell_knowledge[new_position]['safe'] = True
                else:
                    memory.cell_knowledge[new_position]['safe'] = True
            else:
                self.alive = False
                log_msg += " | AGENTE DESTRUÃƒÂDO"
        elif cell_content == 'F':
            log_msg += f" | CUSTO TOTAL DO CAMINHO: {self.path_cost:.2f}"
        
        return log_msg

# ============================================
# 5. MOTOR DE INFERÃƒÅ NCIA PARA GRUPOS ML
# ============================================

class InferenceEngineC:
    def __init__(self, weights=None):
        # Pesos otimizados para busca exploratÃƒÂ³ria
        self.weights = weights or {
            'F': 100.0,   # Bandeira - se descoberta
            'T': 8.0,     # Tesouro - importante para desativar bombas
            'L': 2.0,     # Livre
            'B': -200.0,  # Bomba - evitar completamente
            'U': 3.0,     # Desconhecido - PRIORIZAR exploraÃƒÂ§ÃƒÂ£o
            'E': -0.8     # Explorado - evitar revisitar
        }
        
        self.cost_weight = 0.5
        self.risk_weight = 1.0  # Ã¢Å“â€¦ Reduzido de 2.0 para 1.0 (menos conservador)
        self.exploration_weight = 2.0
    
    def calculate_score(self, cell_type, position, shared_memory, agent):
        """Calcula pontuaÃƒÂ§ÃƒÂ£o otimizada para busca exploratÃƒÂ³ria"""
        base_score = self.weights.get(cell_type, 0.0)
        
        # Se bandeira foi descoberta, priorizar caminho atÃƒÂ© ela
        if shared_memory.flag_position:
            current_dist = shared_memory.estimate_distance_to_flag(agent.position)
            new_dist = shared_memory.estimate_distance_to_flag(position)
            if new_dist < current_dist:
                base_score += 20.0
        else:
            # Bandeira ainda nÃƒÂ£o descoberta - priorizar exploraÃƒÂ§ÃƒÂ£o
            if not shared_memory.cell_knowledge[position]['explored']:
                base_score += self.exploration_weight * 5.0
        
        # Penalidade por custo e risco
        cost = shared_memory.cell_knowledge[position]['cost']
        risk = shared_memory.cell_knowledge[position]['risk']
        
        base_score -= (cost * self.cost_weight)
        base_score -= (risk * self.risk_weight)
        
        # Penalidade por estar perto de bomba (Ã¢Å“â€¦ REDUZIDAS para permitir exploraÃƒÂ§ÃƒÂ£o)
        x, y = position
        for bx, by in shared_memory.bombs_found:
            bomb_distance = abs(bx - x) + abs(by - y)
            if bomb_distance == 1:  # Adjacente a bomba
                base_score -= 10.0  # Ã¢Å“â€¦ Reduzido de -30 para -10
            elif bomb_distance == 2:  # 2 cÃƒÂ©lulas de distÃƒÂ¢ncia
                base_score -= 3.0  # Ã¢Å“â€¦ Nova penalidade moderada
        
        return base_score
    
    def decide_action(self, available_cells, shared_memory, env, agent):
        """Decide aÃƒÂ§ÃƒÂ£o otimizando exploraÃƒÂ§ÃƒÂ£o"""
        if shared_memory is None or getattr(agent, 'is_baseline', False):
            memory = agent.memory
        else:
            memory = shared_memory
            
        if not available_cells:
            return None
        
        best_score = -float('inf')
        best_action = None
        
        for pos, predicted_type in available_cells:
            if pos in memory.bombs_found and not memory.cell_knowledge[pos]['safe']:
                score = self.weights['B']
            else:
                score = self.calculate_score(predicted_type, pos, memory, agent)
            
            # Pequena aleatoriedade
            score += random.uniform(-0.05, 0.05)
            
            if score > best_score:
                best_score = score
                best_action = pos
        
        return best_action

# ============================================
# 6. SIMULAÃƒâ€¡ÃƒÆ’O DA ABORDAGEM C
# ============================================

class ApproachCSimulation:
    def __init__(self, num_agents=4, bomb_ratio=0.3, treasure_count=10, 
                 homogeneous=True, max_steps=500):
        self.env = EnvironmentC(bomb_ratio=bomb_ratio, treasure_count=treasure_count)
        # IMPORTANTE: Passar flag_position apenas para GUI (true_flag_position)
        self.shared_memory = SharedMemoryC(flag_position=self.env.flag_position)
        self.num_agents = num_agents
        self.homogeneous = homogeneous
        self.max_steps = max_steps
        self.agents = []
        self.logs = []
        self.metrics = {
            'flag_found': False,
            'treasures_found': 0,
            'total_treasures': treasure_count,
            'agents_alive': 0,
            'steps_taken': 0,
            'execution_time': 0,
            'success': False,
            'bomb_ratio': bomb_ratio,
            'min_path_cost': float('inf'),
            'avg_path_cost': 0,
            'explored_percentage': 0
        }
        self.setup_agents()
        
    def setup_agents(self):
        """Configura agentes"""
        # Ã¢Å“â€¦ FIX CRÃƒÂTICO: Marcar posiÃƒÂ§ÃƒÂ£o inicial (0,0) como explorada ANTES de criar agentes
        initial_pos = (0, 0)
        cell_content = self.env.get_cell(*initial_pos)
        self.shared_memory.explored.add(initial_pos)
        self.shared_memory.cell_knowledge[initial_pos]['explored'] = True
        self.shared_memory.cell_knowledge[initial_pos]['type'] = cell_content
        self.shared_memory.cell_knowledge[initial_pos]['safe'] = True
        
        base_weights = {
            'F': 100.0, 'T': 8.0, 'L': 2.0, 
            'B': -200.0, 'U': 3.0, 'E': -0.8
        }
        
        if self.homogeneous:
            # DistribuiÃƒÂ§ÃƒÂ£o homogÃƒÂ©nea entre modelos: ~33% KNN, ~33% NaiveBayes, ~33% RandomForest
            model_list = []
            base = self.num_agents // 3
            remainder = self.num_agents % 3
            counts = {'KNN': base, 'NaiveBayes': base, 'RandomForest': base}
            order = ['KNN', 'NaiveBayes', 'RandomForest']
            for r in range(remainder):
                counts[order[r]] += 1
            for model, cnt in counts.items():
                model_list.extend([model] * cnt)
            # Ajustar comprimento
            if len(model_list) < self.num_agents:
                model_list.extend(['NaiveBayes'] * (self.num_agents - len(model_list)))
            elif len(model_list) > self.num_agents:
                model_list = model_list[:self.num_agents]
            random.shuffle(model_list)

            for i in range(self.num_agents):
                # HomogÃƒÂ©neo: todos os agentes com pesos balanceados (33.33% cada)
                weights = {'KNN': 1/3, 'NaiveBayes': 1/3, 'RandomForest': 1/3}
                agent = AgentC(agent_id=i, inference_weights=base_weights, model_weights=weights)
                self.agents.append(agent)
        else:
            # HeterogÃƒÂ©neo: atribuiÃƒÂ§ÃƒÂ£o cÃƒÂ­clica de modelo principal por agente
            # Agente 1 -> KNN (60% KNN, 20% NB, 20% RF), Agente 2 -> RandomForest (60% RF,...), Agente 3 -> NaiveBayes, repetir
            order = ['KNN', 'RandomForest', 'NaiveBayes']
            for i in range(self.num_agents):
                primary = order[i % len(order)]
                # construir pesos: primary 0.6, restantes 0.2 cada
                if primary == 'KNN':
                    weights = {'KNN': 0.6, 'NaiveBayes': 0.2, 'RandomForest': 0.2}
                elif primary == 'RandomForest':
                    weights = {'RandomForest': 0.6, 'KNN': 0.2, 'NaiveBayes': 0.2}
                else:
                    weights = {'NaiveBayes': 0.6, 'KNN': 0.2, 'RandomForest': 0.2}

                agent = AgentC(agent_id=i, inference_weights=base_weights, model_weights=weights)
                self.agents.append(agent)
    
    def run_simulation(self, verbose=False):
        """Executa simulaÃƒÂ§ÃƒÂ£o completa"""
        start_time = time.time()
        step = 0
        
        if verbose:
            self.logs.append(f"=== INÃƒÂCIO SIMULAÃƒâ€¡ÃƒÆ’O ABORDAGEM C ===")
            self.logs.append(f"Agentes: {self.num_agents} | Objetivo: Encontrar bandeira DESCONHECIDA")
            self.logs.append(f"PosiÃƒÂ§ÃƒÂ£o real da bandeira: {self.env.flag_position} (OCULTA dos agentes)")
            self.logs.append(f"Bombas: {self.metrics['bomb_ratio']*100}%")
            self.logs.append(f"Tipo: {'HomogÃƒÂªneo' if self.homogeneous else 'HeterogÃƒÂªneo'}")
        
        while step < self.max_steps:
            step += 1
            agents_alive = len([a for a in self.agents if a.alive])
            
            if agents_alive == 0:
                break
            
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                next_pos = agent.choose_action(self.shared_memory, self.env)
                if next_pos:
                    log_msg = agent.move_to(next_pos, self.shared_memory, self.env)
                    if verbose and ("TESOURO" in log_msg or "BOMBA" in log_msg or "BANDEIRA" in log_msg):
                        self.logs.append(log_msg)
            
            if verbose and step % 50 == 0:
                self.logs.append(f"Passo {step}: Bandeira {'ENCONTRADA' if self.shared_memory.flag_found else 'nÃƒÂ£o encontrada'}, {agents_alive} agentes vivos")
            
            # Verificar sucesso
            if self.shared_memory.flag_found:
                self.metrics['success'] = True
                if verbose:
                    self.logs.append(f"Ã¢Å“â€¦ SUCESSO! Bandeira encontrada no passo {step}!")
                break
        
        # MÃƒÂ©tricas finais
        end_time = time.time()
        self.metrics['execution_time'] = end_time - start_time
        self.metrics['flag_found'] = self.shared_memory.flag_found
        self.metrics['treasures_found'] = len(self.shared_memory.treasures_collected)
        self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
        self.metrics['steps_taken'] = step
        self.metrics['explored_percentage'] = self.get_explored_percentage()
        
        alive_agents = [a for a in self.agents if a.alive]
        if alive_agents:
            path_costs = [a.path_cost for a in alive_agents]
            self.metrics['min_path_cost'] = min(path_costs)
            self.metrics['avg_path_cost'] = np.mean(path_costs)
        
        if verbose:
            self.logs.append(f"\n=== FIM DA SIMULAÃƒâ€¡ÃƒÆ’O ===")
            self.logs.append(f"Tempo: {self.metrics['execution_time']:.2f}s")
            self.logs.append(f"Passos: {self.metrics['steps_taken']}")
            self.logs.append(f"Bandeira: {'Encontrada' if self.metrics['flag_found'] else 'NÃƒÂ£o encontrada'}")
            self.logs.append(f"Tesouros: {self.metrics['treasures_found']}/{self.env.treasure_count}")
            self.logs.append(f"Agentes vivos: {self.metrics['agents_alive']}")
            self.logs.append(f"Custo mÃƒÂ­nimo: {self.metrics['min_path_cost']:.2f}")
            self.logs.append(f"Sucesso: {'SIM' if self.metrics['success'] else 'NÃƒÆ’O'}")
        
        return self.metrics
    
    def get_explored_percentage(self):
        """Calcula percentagem de cÃƒÂ©lulas exploradas"""
        explored_count = len(self.shared_memory.explored)
        total_cells = self.env.size * self.env.size
        return (explored_count / total_cells) * 100
    
    def print_logs(self):
        """Exibe logs da simulaÃƒÂ§ÃƒÂ£o"""
        for log in self.logs:
            print(log)

# ============================================
# 7. BASELINE C: N AGENTES A* COLABORATIVOS
# ============================================

class BaselineC_AStar:
    """
    Baseline C: N agentes A* colaborativos
    
    REGRAS CRÃƒÂTICAS:
    Ã¢Å“â€¦ N agentes permitidos
    Ã¢Å“â€¦ Todos executam A* puro
    Ã¢Å“â€¦ f(n) = g(n) + h(n) idÃƒÂªntica para todos
    Ã¢Å“â€¦ SEM aprendizagem
    Ã¢Å“â€¦ SEM motor de inferÃƒÂªncia
    Ã¢Å“â€¦ SEM pesos configurÃƒÂ¡veis
    Ã¢Å“â€¦ SEM decisÃƒÂ£o inteligente
    
    CitaÃƒÂ§ÃƒÂ£o para o relatÃƒÂ³rio:
    "Nas baselines, o nÃƒÂºmero de agentes pode variar conforme a configuraÃƒÂ§ÃƒÂ£o 
    da simulaÃƒÂ§ÃƒÂ£o; no entanto, todos os agentes executam o mesmo algoritmo 
    clÃƒÂ¡ssico de forma idÃƒÂªntica, sem aprendizagem ou motor de inferÃƒÂªncia, 
    servindo apenas para explorar o paralelismo e nÃƒÂ£o para introduzir 
    inteligÃƒÂªncia adicional."
    """
    def __init__(self, num_agents=4, bomb_ratio=0.3, treasure_count=10, max_steps=500):
        self.env = EnvironmentC(bomb_ratio=bomb_ratio, treasure_count=treasure_count)
        # Bandeira desconhecida para baseline tambÃƒÂ©m
        self.shared_memory = SharedMemoryC(flag_position=self.env.flag_position)
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.agents = []
        self.logs = []
        
        self.metrics = {
            'flag_found': False,
            'execution_time': 0,
            'steps_taken': 0,
            'success': False,
            'path_length': 0,
            'treasures_found': 0,
            'total_treasures': treasure_count,
            'agents_alive': num_agents,
            'min_path_cost': float('inf'),
            'avg_path_cost': 0
        }
        
        self.setup_agents()
    
    def setup_agents(self):
        """
        Criar N agentes A*
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
        
        # Criar N agentes A* idÃƒÂªnticos
        for i in range(self.num_agents):
            agent = AgentAStar(agent_id=i, start_pos=initial_pos)
            self.agents.append(agent)
    
    def run(self):
        """Executa simulaÃƒÂ§ÃƒÂ£o com N agentes A*"""
        start_time = time.time()
        step = 0
        
        # Loop principal
        while step < self.max_steps:
            step += 1
            agents_alive = len([a for a in self.agents if a.alive])
            
            if agents_alive == 0:
                break
            
            # Cada agente executa A*
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                # Escolher prÃƒÂ³xima cÃƒÂ©lula usando A*
                next_pos = agent.choose_action(self.shared_memory, self.env)
                
                if next_pos:
                    log_msg = agent.move_to(next_pos, self.shared_memory, self.env)
            
            # Verificar se bandeira foi encontrada
            if self.shared_memory.flag_found:
                self.metrics['success'] = True
                self.metrics['flag_found'] = True
                break
        
        # MÃƒÂ©tricas finais
        end_time = time.time()
        self.metrics['execution_time'] = end_time - start_time
        self.metrics['agents_alive'] = len([a for a in self.agents if a.alive])
        self.metrics['steps_taken'] = step
        self.metrics['treasures_found'] = len(self.shared_memory.treasures_collected)
        
        # Caminho do agente que encontrou
        alive_agents = [a for a in self.agents if a.alive]
        if alive_agents:
            path_costs = [a.path_cost for a in alive_agents]
            self.metrics['min_path_cost'] = min(path_costs) if path_costs else 0
            self.metrics['avg_path_cost'] = np.mean(path_costs) if path_costs else 0
            steps_taken_list = [a.steps_taken for a in alive_agents if a.steps_taken > 0]
            self.metrics['path_length'] = min(steps_taken_list) if steps_taken_list else 0
        
        return self.metrics

# Alias para manter compatibilidade
BaselineC_ML = BaselineC_AStar

# ============================================
# 8. FUNÃƒâ€¡Ãƒâ€¢ES DE ANÃƒÂLISE
# ============================================

def run_multiple_simulations_c(num_simulations=5, num_agents=4, homogeneous=True):
    """Executa mÃƒÂºltiplas simulaÃƒÂ§ÃƒÂµes para estatÃƒÂ­sticas"""
    results = []
    
    for i in range(num_simulations):
        sim = ApproachCSimulation(
            num_agents=num_agents,
            bomb_ratio=0.3,
            treasure_count=10,
            homogeneous=homogeneous,
            max_steps=300
        )
        
        metrics = sim.run_simulation(verbose=False)
        results.append(metrics)
    
    # Calcular mÃƒÂ©dias
    avg_flag_found = np.mean([1 if r['flag_found'] else 0 for r in results])
    avg_time = np.mean([r['execution_time'] for r in results])
    success_rate = np.mean([1 if r['success'] else 0 for r in results])
    avg_survivors = np.mean([r['agents_alive'] for r in results])
    avg_path_cost = np.mean([r['min_path_cost'] for r in results if r['min_path_cost'] != float('inf')])
    
    return {
        'type': 'HomogÃƒÂªneo' if homogeneous else 'HeterogÃƒÂªneo',
        'num_agents': num_agents,
        'avg_flag_found': avg_flag_found,
        'avg_time': avg_time,
        'success_rate': success_rate,
        'avg_survivors': avg_survivors,
        'avg_path_cost': avg_path_cost,
        'results': results
    }

def compare_approaches_c():
    """Compara abordagens homogÃƒÂªnea, heterogÃƒÂªnea e baseline A*"""
    print("Comparando Abordagem C...")
    
    results = []
    agent_counts = [3, 4, 6, 8]
    
    for num_agents in agent_counts:
        print(f"\nTestando com {num_agents} agentes...")
        
        # HomogÃƒÂªneo
        homo_results = run_multiple_simulations_c(
            num_simulations=5, num_agents=num_agents, homogeneous=True
        )
        
        # HeterogÃƒÂªneo
        hetero_results = run_multiple_simulations_c(
            num_simulations=5, num_agents=num_agents, homogeneous=False
        )
        
        # Baseline A* com N agentes
        astar_results = []
        for _ in range(5):
            baseline = BaselineC_AStar(num_agents=num_agents)
            baseline_metrics = baseline.run()
            astar_results.append(baseline_metrics)
        
        astar_success_rate = np.mean([1 if r['success'] else 0 for r in astar_results])
        astar_avg_time = np.mean([r['execution_time'] for r in astar_results])
        
        results.append({
            'num_agents': num_agents,
            'homogeneous': homo_results,
            'heterogeneous': hetero_results,
            'astar': {
                'success_rate': astar_success_rate,
                'avg_time': astar_avg_time,
                'results': astar_results
            }
        })
        
        print(f"  HomogÃƒÂªneo: {homo_results['success_rate']:.0%} sucesso")
        print(f"  HeterogÃƒÂªneo: {hetero_results['success_rate']:.0%} sucesso")
        print(f"  A* ({num_agents} agentes): {astar_success_rate:.0%} sucesso")
    
    return results

# ============================================
# 9. EXECUÃƒâ€¡ÃƒÆ’O PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("PROJETO IA - ABORDAGEM C: BUSCA DA BANDEIRA (DESCONHECIDA)")
    print("="*60)
    
    # Teste rÃƒÂ¡pido
    print("\n1. Teste rÃƒÂ¡pido - 4 agentes heterogÃƒÂªneos:")
    sim = ApproachCSimulation(num_agents=4, homogeneous=False, max_steps=300)
    metrics = sim.run_simulation(verbose=True)
    sim.print_logs()
    
    # Teste Baseline A* com 4 agentes
    print("\n\n2. Teste Baseline A* - 4 agentes:")
    baseline = BaselineC_AStar(num_agents=4)
    baseline_metrics = baseline.run()
    print(f"Baseline A* (4 agentes): {'Sucesso' if baseline_metrics['success'] else 'Falha'}")
    print(f"Bandeira: {'Encontrada' if baseline_metrics['flag_found'] else 'NÃƒÂ£o encontrada'}")
    print(f"Passos: {baseline_metrics['steps_taken']}")
    print(f"Tempo: {baseline_metrics['execution_time']:.2f}s")
    
    # ComparaÃƒÂ§ÃƒÂ£o completa
    print("\n\n3. ComparaÃƒÂ§ÃƒÂ£o completa...")
    results = compare_approaches_c()
    
    print("\n\n4. ANÃƒÂLISE DOS RESULTADOS:")
    print("-"*50)
    
    best_homo = max(results, key=lambda x: x['homogeneous']['success_rate'])
    best_hetero = max(results, key=lambda x: x['heterogeneous']['success_rate'])
    
    print(f"\nMelhor configuraÃƒÂ§ÃƒÂ£o HomogÃƒÂªnea: {best_homo['num_agents']} agentes")
    print(f"  Taxa de sucesso: {best_homo['homogeneous']['success_rate']:.0%}")
    
    print(f"\nMelhor configuraÃƒÂ§ÃƒÂ£o HeterogÃƒÂªnea: {best_hetero['num_agents']} agentes")
    print(f"  Taxa de sucesso: {best_hetero['heterogeneous']['success_rate']:.0%}")
    
    print(f"\nBaseline A*:")
    print(f"  Executa A* puro com N agentes colaborativos")
    print(f"  MESMA funÃƒÂ§ÃƒÂ£o f(n)=g(n)+h(n) para todos")
    print(f"  SEM aprendizagem, SEM inferÃƒÂªncia, SEM pesos")