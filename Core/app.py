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
        self.hubs = {
            MLModelType.KNN: CommunicationHub(),
            MLModelType.DECISION_TREE: CommunicationHub(),
            MLModelType.NAIVE_BAYES: CommunicationHub()
        }
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
    for hub in state.hubs.values():
        hub.reset()
    state.logged_destroyed_ids.clear()
    state.mission_finished = False
    state.step_count = 0

    flat_grid = ['L'] * TOTAL_CELLS
    num_bombs = 50

    # Colocar bombas, deixando bordas livres para travessibilidade (linhas 0-1 e colunas 0-1)
    placed = 0
    while placed < num_bombs:
        idx = random.randint(0, TOTAL_CELLS - 1)
        x = idx % GRID_SIZE
        y = idx // GRID_SIZE
        if flat_grid[idx] == 'L' and idx != 0 and y > 1 and x > 1:
            flat_grid[idx] = 'B'
            placed += 1

    # Colocar tesouros para todas as abordagens
    if current_approach in [Approach.A, Approach.B, Approach.C]:
        placed = 0
        while placed < 25:
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
            row.append(Cell(x=j, y=i, type=cell_type, isExplored=False, collected=False, neutralized=False))
        state.grid.append(row)

    # Explorar posição inicial
    state.grid[0][0].isExplored = True
    for hub in state.hubs.values():
        hub.registerExploration(state.grid[0][0])

    # Criar agentes
    models = [MLModelType.KNN, MLModelType.DECISION_TREE, MLModelType.NAIVE_BAYES]
    state.agents = []
    for i in range(num_agents):
        state.agents.append(Agent(
            id=i,
            model=models[i % len(models)],
            x=0, y=0,
            isAlive=True,
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

        hub = state.hubs[agent.model]
        current_pos = Point(x=agent.x, y=agent.y)
        valid_moves = [
            Point(x=agent.x + 1, y=agent.y), Point(x=agent.x - 1, y=agent.y),
            Point(x=agent.x, y=agent.y + 1), Point(x=agent.x, y=agent.y - 1)
        ]
        valid_moves = [m for m in valid_moves if 0 <= m.x < GRID_SIZE and 0 <= m.y < GRID_SIZE and not hub.isKnownBomb(m.x, m.y)]

        unexplored = [m for m in valid_moves if not hub.isExplored(m.x, m.y)]
        pool = unexplored if unexplored else valid_moves
        if not pool:
            continue

        # Escolher movimento baseado no modelo
        if agent.model == MLModelType.KNN:
            move = knnPredictNextMove(current_pos, pool, state.grid, hub)
        elif agent.model == MLModelType.DECISION_TREE:
            move = decisionTreePredictNextMove(current_pos, pool, state.grid, hub)
        else:  # NAIVE_BAYES
            move = naiveBayesPredictNextMove(current_pos, pool, state.grid, hub)

        is_new = not hub.isExplored(move.x, move.y)
        cell = state.grid[move.y][move.x]
        agent.x, agent.y = move.x, move.y
        agent.path.append({'x': move.x, 'y': move.y})

        if is_new:
            if cell.type == 'B':
                if agent.shield_count > 0:
                    agent.shield_count -= 1
                    cell.neutralized = True
                    cell.type = 'L'  # Defuse the bomb
                    key = f"{cell.x},{cell.y}"
                    if key in hub.knownBombs:
                        hub.knownBombs.remove(key)
                    batch_logs.append({
                        'timestamp': time.time(),
                        'message': f"Agente {agent.id} neutralizou obstáculo em [{move.x},{move.y}] (Escudos restantes: {agent.shield_count})",
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
                agent.shield_count += 1
                cell.collected = True
                cell.type = 'L'  # Consumir o tesouro, tornando a célula vazia
                batch_logs.append({
                    'timestamp': time.time(),
                    'message': f"Agente {agent.id} coletou T (Escudos: {agent.shield_count})",
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

            hub.registerExploration(cell)
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
        test_cell = Cell(0, 0, 'L', False, False, False)
        test_agent = Agent(0, MLModelType.KNN, 0, 0, True)
        test_hub = CommunicationHub()
    except Exception as e:
        st.error(f"Erro nos módulos importados: {e}")
        return
    
    try:
        st.set_page_config(page_title="AGENTES COLABORATIVOS", page_icon=None, layout="wide")

        st.title("AGENTES COLABORATIVOS")
        st.markdown("*...*")

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
                # Número fixo de bombas para equilíbrio
                bomb_ratio = 30

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
                        if cell.collected:
                            bg_color = '#f59e0b'
                            content = 'T'
                        elif cell.neutralized:
                            bg_color = '#6b7280'  # Cor para neutralizado, ex: cinza
                            content = 'N'
                        elif cell.type == 'B':
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

        # Nova seção de análise baseada no que realmente aconteceu na matriz
        st.header("Análise de Ciclo")

        # 1) Melhor grupo — agente que progrediu mais (maior comprimento de caminho)
        st.subheader("1. Melhor Grupo de Modelos (por progresso)")
        agent_progress = [(a.id, a.model.value, len(a.path)) for a in state.agents]
        if agent_progress:
            best_agent = max(agent_progress, key=lambda t: t[2])
            st.write(f"Agente que progrediu mais: **Agente {best_agent[0]}** ({best_agent[1]}) — passos: {best_agent[2]}")
        else:
            st.write("Nenhum agente presente nesta sessão.")

        # 2) Vantagem da colaboração heterogênea — comparar cobertura combinada vs individuais
        st.subheader("2. Vantagem da Colaboração Heterogênea (neste ciclo)")
        # Recolher conjuntos de células exploradas por cada modelo (usando os hubs por modelo)
        model_explored: Dict[str, set] = {}
        for model_enum, hub in state.hubs.items():
            model_explored[model_enum.value] = set(hub.exploredCells)

        union_explored = set().union(*model_explored.values()) if model_explored else set()
        per_model_counts = {m: len(s) for m, s in model_explored.items()}
        if per_model_counts:
            best_single = max(per_model_counts.items(), key=lambda x: x[1])
            extra_by_collab = len(union_explored) - best_single[1]
            st.write(f"Cobertura combinada: **{len(union_explored)}** células únicas exploradas; melhor modelo individual (**{best_single[0]}**) explorou **{best_single[1]}** células.")
            if extra_by_collab > 0:
                st.write(f"A colaboração heterogênea permitiu explorar **{extra_by_collab}** células adicionais além do melhor modelo individual — mostra benefício prático nesta sessão.")
            else:
                st.write("A colaboração não trouxe ganho de cobertura nesta sessão (modelos sobrepuseram exploração).")
        else:
            st.write("Sem dados de exploração por modelo nesta sessão.")

        # 3) Menos vs Mais Agentes — adaptar resposta ao que aconteceu
        st.subheader("3. Exploração: Menos vs. Mais Agentes (análise da sessão)")
        agents_total = num_agents
        agents_alive = len([a for a in state.agents if a.isAlive])
        avg_path = (sum(len(a.path) for a in state.agents) / len(state.agents)) if state.agents else 0
        explored_cells = sum(1 for row in state.grid for c in row if c.isExplored)

        st.write(f"Agentes configurados: **{agents_total}** — Agentes vivos no momento: **{agents_alive}**.")
        st.write(f"Células exploradas nesta sessão: **{explored_cells}/{TOTAL_CELLS}**. Comprimento médio de caminho por agente: **{avg_path:.1f}** passos.")

        if agents_total <= 4:
            st.write("Com poucos agentes, a exploração tende a ser mais lenta; se muitos agentes morreram (baixo número de vivos), a missão frequentemente falha — observe os sobreviventes acima.")
        else:
            st.write("Com mais agentes, a cobertura costuma aumentar (mais caminhos simultâneos). Se muitos agentes ainda estiverem vivos e a exploração for alta, isso confirma vantagem de escala.")

        # Observação prática baseada na sessão
        if explored_cells == TOTAL_CELLS and agents_alive >= 1:
            st.write("Resultado: Ambiente totalmente explorado com agentes sobreviventes — missão bem-sucedida nesta sessão.")
        elif explored_cells == TOTAL_CELLS and agents_alive == 0:
            st.write("Resultado: Ambiente totalmente explorado, mas sem agentes vivos — missão tecnicamente completa, porém sem sobreviventes.")
        else:
            st.write("Resultado: Exploração incompleta nesta sessão — considere aumentar agentes, ajustar posição de tesouros, ou reduzir densidade de bombas para melhorar cobertura.")

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