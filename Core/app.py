import streamlit as st
import time
import random
import math
import sys
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Adicionar a raiz do projeto ao caminho Python para imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Adicionar todos os subdiretórios ao caminho Python
for subdir in ['Agentes', 'Comunicacao', 'Abordagem', 'Core']:
    subdir_path = os.path.join(project_root, subdir)
    if subdir_path not in sys.path:
        sys.path.insert(0, subdir_path)

# Importar nossa lógica Python
from data_types import Cell, Agent, MLModelType, Approach, Point, SimulationStats, ModelRanking
from communication_hub import CommunicationHub
from approach_a import evaluateApproachA
from approach_b import evaluateApproachB
from approach_c import evaluateApproachC
from decision_tree_model import decisionTreePredictNextMove
from knn_model import knnPredictNextMove
from naive_bayes_model import naiveBayesPredictNextMove

# Constantes
GRID_SIZE = 10
TOTAL_CELLS = GRID_SIZE * GRID_SIZE
AGENT_COLORS = ['🔴', '🔵', '🟡', '🟢', '🟣', '🟠', '⚪', '🟤', '🟥', '🟦']

class AppState:
    def __init__(self):
        self.grid: List[List[Cell]] = []
        self.agents: List[Agent] = []
        self.hub = CommunicationHub()
        self.logs: List[Dict] = []
        self.stats: Optional[SimulationStats] = None
        self.historical_stats: List[SimulationStats] = []
        self.global_rankings: Dict[str, Dict[str, int]] = {
            'A': {'KNN': 0, 'DecisionTree': 0, 'NaiveBayes': 0},
            'B': {'KNN': 0, 'DecisionTree': 0, 'NaiveBayes': 0},
            'C': {'KNN': 0, 'DecisionTree': 0, 'NaiveBayes': 0}
        }
        self.is_running = False
        self.mission_finished = False
        self.start_time = 0
        self.logged_destroyed_ids = set()
        self.step_count = 0

def initialize_environment(state: AppState, num_agents: int, bomb_ratio: float, current_approach: Approach):
    state.hub.reset()
    state.logged_destroyed_ids.clear()
    state.mission_finished = False
    state.step_count = 0

    flat_grid = ['L'] * TOTAL_CELLS
    num_bombs = int(TOTAL_CELLS * (bomb_ratio / 100))
    num_treasures = 10

    # Colocar bombas
    placed = 0
    while placed < num_bombs:
        idx = random.randint(0, TOTAL_CELLS - 1)
        if flat_grid[idx] == 'L' and idx != 0:
            flat_grid[idx] = 'B'
            placed += 1

    # Colocar tesouros para Abordagem A
    if current_approach == Approach.A:
        placed = 0
        while placed < num_treasures:
            idx = random.randint(0, TOTAL_CELLS - 1)
            if flat_grid[idx] == 'L' and idx != 0:
                flat_grid[idx] = 'T'
                placed += 1

    # Colocar bandeira para Abordagem C
    if current_approach == Approach.C:
        while True:
            idx = random.randint(0, TOTAL_CELLS - 1)
            if flat_grid[idx] == 'L' and idx != 0:
                flat_grid[idx] = 'F'
                break

    # Criar grelha
    state.grid = []
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            cell_type = flat_grid[i * GRID_SIZE + j]
            row.append(Cell(x=j, y=i, type=cell_type, isExplored=False))
        state.grid.append(row)

    # Explorar posição inicial
    state.grid[0][0].isExplored = True
    state.hub.registerExploration(state.grid[0][0])

    # Criar agentes
    models = [MLModelType.KNN, MLModelType.DECISION_TREE, MLModelType.NAIVE_BAYES]
    state.agents = []
    for i in range(num_agents):
        state.agents.append(Agent(
            id=i,
            model=models[i % len(models)],
            x=0, y=0,
            isAlive=True,
            hasShield=False,
            path=[{'x': 0, 'y': 0}],
            color=AGENT_COLORS[i % len(AGENT_COLORS)]
        ))

    state.logs = []
    state.stats = None

    mission_desc = {
        Approach.A: 'Foco em T.',
        Approach.B: 'Foco em exploração total.',
        Approach.C: 'Localize a F.'
    }[current_approach]

    add_log(state, f"Protocolo {current_approach.value} iniciado. {mission_desc}", 'info')
    add_log(state, "Ambiente inicializado com sucesso.", 'success')

def add_log(state: AppState, message: str, type_: str = 'info'):
    state.logs.insert(0, {
        'timestamp': time.time(),
        'message': message,
        'type': type_
    })
    if len(state.logs) > 200:
        state.logs = state.logs[:200]

def step_simulation(state: AppState, current_approach: Approach, num_agents: int):
    if state.mission_finished:
        return

    state.step_count += 1

    any_alive = False
    flag_found_by_any_agent = False
    batch_logs = []
    current_winner_model = None

    for agent in state.agents:
        if not agent.isAlive or (current_approach == Approach.C and flag_found_by_any_agent):
            continue
        any_alive = True

        current_pos = Point(x=agent.x, y=agent.y)
        valid_moves = [
            Point(x=agent.x + 1, y=agent.y), Point(x=agent.x - 1, y=agent.y),
            Point(x=agent.x, y=agent.y + 1), Point(x=agent.x, y=agent.y - 1)
        ]
        valid_moves = [m for m in valid_moves if 0 <= m.x < GRID_SIZE and 0 <= m.y < GRID_SIZE and not state.hub.isKnownBomb(m.x, m.y)]

        unexplored = [m for m in valid_moves if not state.hub.isExplored(m.x, m.y)]
        pool = unexplored if unexplored else valid_moves
        if not pool:
            continue

        # Escolher movimento baseado no modelo
        if agent.model == MLModelType.KNN:
            move = knnPredictNextMove(current_pos, pool, state.grid, state.hub)
        elif agent.model == MLModelType.DECISION_TREE:
            move = decisionTreePredictNextMove(current_pos, pool, state.grid, state.hub)
        else:  # NAIVE_BAYES
            move = naiveBayesPredictNextMove(current_pos, pool, state.grid, state.hub)

        is_new = not state.hub.isExplored(move.x, move.y)
        cell = state.grid[move.y][move.x]
        agent.x, agent.y = move.x, move.y
        agent.path.append({'x': move.x, 'y': move.y})

        if is_new:
            if cell.type == 'B':
                if agent.hasShield:
                    agent.hasShield = False
                    batch_logs.append({
                        'timestamp': time.time(),
                        'message': f"Agente {agent.id} neutralizou obstáculo em [{move.x},{move.y}]",
                        'type': 'warning'
                    })
                else:
                    agent.isAlive = False
                    if agent.id not in state.logged_destroyed_ids:
                        batch_logs.append({
                            'timestamp': time.time(),
                            'message': f"Agente {agent.id} [{agent.model.value}] neutralizado por B",
                            'type': 'error'
                        })
                        state.logged_destroyed_ids.add(agent.id)
            elif cell.type == 'T':
                agent.hasShield = True
                batch_logs.append({
                    'timestamp': time.time(),
                    'message': f"Agente {agent.id} coletou T (Escudo de proteção)",
                    'type': 'success'
                })
            elif cell.type == 'F' and current_approach == Approach.C:
                flag_found_by_any_agent = True
                current_winner_model = agent.model
                batch_logs.append({
                    'timestamp': time.time(),
                    'message': f"ALERTA: Agente {agent.id} localizou a F!",
                    'type': 'success'
                })

            state.hub.registerExploration(cell)
            state.grid[move.y][move.x].isExplored = True

    for log_entry in reversed(batch_logs):
        state.logs.insert(0, log_entry)

    flat = [cell for row in state.grid for cell in row]
    explored_count = sum(1 for c in flat if c.isExplored)

    success = False
    should_end = False

    if current_approach == Approach.C:
        success = flag_found_by_any_agent
        should_end = success or not any_alive
    elif current_approach == Approach.B:
        success = evaluateApproachB(state.grid, state.agents)
        should_end = success or not any_alive or (explored_count == TOTAL_CELLS)
    else:  # Abordagem A
        success = evaluateApproachA(state.grid)
        should_end = not any_alive or (explored_count == TOTAL_CELLS)

    if should_end:
        state.mission_finished = True
        duration = time.time() - state.start_time
        state.stats = SimulationStats(
            cellsExplored=round((explored_count / TOTAL_CELLS) * 100),
            agentsAlive=len([a for a in state.agents if a.isAlive]),
            totalAgents=num_agents,
            executionTime=duration,
            success=success,
            approach=current_approach
        )

        if success:
            final_winner = current_winner_model or max(
                [(model, sum(1 for a in state.agents if a.model == model and a.isAlive))
                 for model in [MLModelType.KNN, MLModelType.DECISION_TREE, MLModelType.NAIVE_BAYES]],
                key=lambda x: x[1]
            )[0]
            state.global_rankings[current_approach.value][final_winner.value] += 1

        state.historical_stats.append(state.stats)
        state.is_running = False


    if not should_end and state.step_count > 100:
        state.mission_finished = True
        duration = time.time() - state.start_time
        state.stats = SimulationStats(
            cellsExplored=round((explored_count / TOTAL_CELLS) * 100),
            agentsAlive=len([a for a in state.agents if a.isAlive]),
            totalAgents=num_agents,
            executionTime=duration,
            success=False,
            approach=current_approach
        )

        state.historical_stats.append(state.stats)
        state.is_running = False

        add_log(state, "Missão interrompida por limite de passos.", 'error')
        msg = "Missão concluída com êxito!" if success else "Missão interrompida por perda de agentes."
        add_log(state, msg, 'success' if success else 'error')

def main():
    # Verificar imports
    try:
        # Testar se as classes existem
        test_cell = Cell(0, 0, 'L', False)
        test_agent = Agent(0, MLModelType.KNN, 0, 0, True, False, [], '🔴')
        test_hub = CommunicationHub()
    except Exception as e:
        st.error(f"Erro nos módulos importados: {e}")
        return
    
    try:
        st.set_page_config(page_title="IA Exploradora Multiagente", page_icon="🤖", layout="wide")

        st.title("IA Exploradora Multiagente")
        st.markdown("*Protocolo Colaborativo v2.5 - Versão Python*")

        # Inicializar estado da sessão
        if 'app_state' not in st.session_state:
            st.session_state.app_state = AppState()

        state = st.session_state.app_state

        # Controles da barra lateral
        with st.sidebar:
            st.header("Configuração da Simulação")

            col1, col2 = st.columns(2)
            with col1:
                # número mínimo de agentes 
                num_agents = st.slider("Número de Agentes", 2, 10, 2)
            with col2:
                # Valor inicial do ratio de bombas permanece em 50% 
                bomb_options = [50, 75]

                bomb_labels = [f"{v}%" for v in bomb_options]
                bomb_selected = st.radio("Ratio de Bombas (%)", bomb_labels, index=1, key='bomb_radio')
                bomb_ratio = int(bomb_selected.rstrip('%'))

            # Usar botões rádio para selecionar a abordagem 
            approach_options = [
                ("Abordagem A", Approach.A),
                ("Abordagem B", Approach.B),
                ("Abordagem C", Approach.C),
            ]
            approach_labels = [opt[0] for opt in approach_options]
            selected_label = st.radio("Abordagem", approach_labels)
            # Mapear rótulo selecionado de volta para o enum Approach
            current_approach = next(a for (lbl, a) in approach_options if lbl == selected_label)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Iniciar", disabled=state.is_running):
                    initialize_environment(state, num_agents, bomb_ratio, current_approach)
                    state.start_time = time.time()
                    state.is_running = True

            with col2:
                if st.button("Parar", disabled=not state.is_running):
                    state.is_running = False

            with col3:
                if st.button("Resetar"):
                    state.is_running = False
                    state.mission_finished = False
                    initialize_environment(state, num_agents, bomb_ratio, current_approach)

            # Painel de estatísticas
            if state.stats:
                st.header("Estatísticas")
                st.metric("Células Exploradas", f"{state.stats.cellsExplored}%")
                st.metric("Agentes Vivos", f"{state.stats.agentsAlive}/{state.stats.totalAgents}")
                st.metric("Tempo", f"{state.stats.executionTime:.2f}s")
                st.metric("Sucesso", "✅" if state.stats.success else "❌")

        # Conteúdo principal
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("Mapa de Operações")

            # Visualização da grelha
            grid_html = '<div style="display: grid; grid-template-columns: repeat(10, 1fr); gap: 2px; width: 400px; height: 400px; border: 4px solid #374151; padding: 8px; background: #111827;">'

            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    cell = state.grid[y][x] if state.grid else Cell(x, y, 'U', False)
                    agent_here = next((a for a in state.agents if a.x == x and a.y == y and a.isAlive), None)

                    bg_color = '#374151'  # Por defeito não explorado
                    content = ''

                    if cell.isExplored:
                        if cell.type == 'B':
                            bg_color = '#dc2626'
                            content = 'B'
                        elif cell.type == 'T':
                            bg_color = '#f59e0b'
                            content = 'T'
                        elif cell.type == 'F' and current_approach == Approach.C:
                            bg_color = '#10b981'
                            content = 'F'
                        else:
                            bg_color = '#1f2937'
                    else:
                        if cell.type == 'F' and current_approach == Approach.C:
                            bg_color = '#064e3b'
                            content = 'F'

                    if agent_here:
                        content = agent_here.color

                    grid_html += f'<div style="background: {bg_color}; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 12px;">{content}</div>'

            grid_html += '</div>'
            st.html(grid_html)

        with col2:
            st.header("Logs")
            log_container = st.container(height=300)

            with log_container:
                for log_entry in state.logs[:20]:  # Mostrar últimos 20 logs
                    icon = {'info': '[INFO]', 'success': '[SUCCESS]', 'error': '[ERROR]', 'warning': '[WARNING]'}[log_entry['type']]
                    st.write(f"{icon} {log_entry['message']}")

        # Secção de análise
        st.header("Análise de Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Rankings Globais")
            rankings_df = []
            for approach in ['A', 'B', 'C']:
                for model in ['KNN', 'DecisionTree', 'NaiveBayes']:
                    rankings_df.append({
                        'Abordagem': f'Abordagem {approach}',
                        'Modelo': model,
                        'Vitórias': state.global_rankings[approach][model]
                    })

            st.dataframe(rankings_df, width='stretch')

        with col2:
            st.subheader("Histórico de Sessões")
            if state.historical_stats:
                history_df = [{
                    'Abordagem': stat.approach.value,
                    'Exploração (%)': stat.cellsExplored,
                    'Tempo (s)': round(stat.executionTime, 2),
                    'Sucesso': 'Sim' if stat.success else 'Não'
                } for stat in state.historical_stats[-10:]]  # Last 10

                st.dataframe(history_df, width='stretch')

        # Auto-run simulation
        if state.is_running and not state.mission_finished:
            step_simulation(state, current_approach, num_agents)
            time.sleep(0.1)  # Control simulation speed
            st.rerun()
    except Exception as e:
        st.error(f"Erro no app: {e}")
        st.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erro fatal no app: {e}")
        import traceback
        traceback.print_exc()