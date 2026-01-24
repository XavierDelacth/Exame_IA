"""
testes.py

Arquivo consolidado contendo todos os testes e exemplos de uso do sistema.
Este arquivo inclui:
- Testes básicos da simulação
- Testes do ambiente
- Testes de dispersão inicial
- Exemplos de uso
- Executor de experimentos

INSTRUÇÕES DE USO:
1. Execute o arquivo: python testes.py
2. Os testes básicos serão executados automaticamente
3. Para executar exemplos completos ou experimentos, descomente as chamadas no final do arquivo

TESTES INCLUÍDOS:
- test_simple(): Teste básico de criação e execução de simulação
- test_environment(): Teste de geração de ambientes para diferentes abordagens
- test_dispersion(): Teste de movimento e dispersão dos agentes
- example_usage(): Exemplos passo a passo de uso do sistema
- run_experiments(): Executor completo de experimentos com múltiplas configurações
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import sys
import time

# Adicionar o diretório pai ao path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Importar módulos do projeto
from simulation import Simulation, Environment
from visualization import generate_all_visualizations
from experiment_runner import analyze_specific_configuration

# Diretórios
RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================================================================
# TESTE SIMPLES
# =============================================================================

def test_simple():
    """Teste básico da simulação"""
    print("=== TESTE SIMPLES ===")
    print("Testando simulação...")

    sim = Simulation('A', 2, 0.3, 'homogeneous')
    print('Simulação criada com sucesso')

    result = sim.run(max_iterations=2)
    print('Simulação executada com sucesso')

    print('Posições finais:', [a.position for a in sim.agents])
    print('Sucesso!')
    print()

# =============================================================================
# TESTE DO AMBIENTE
# =============================================================================

def test_environment():
    """Teste da geração de ambientes"""
    print("=== TESTE DO AMBIENTE ===")

    # Testar geração de ambiente para cada abordagem
    for approach in ['A', 'B', 'C']:
        print(f'\n--- Testando abordagem {approach} ---')
        env = Environment(approach=approach)

        print(f'Ambiente gerado com sucesso para abordagem {approach}')
        print(f'Tamanho: {env.size}x{env.size}')
        print(f'Célula inicial (0,0): {env.grid[0,0]}')

        # Contar tipos de células
        unique, counts = np.unique(env.grid, return_counts=True)
        cell_counts = dict(zip(unique, counts))
        print(f'Distribuição de células: {cell_counts}')

        # Verificar factibilidade
        feasible = env.is_feasible()
        print(f'Ambiente factível: {feasible}')

        if approach == 'A':
            treasures = np.sum(env.grid == 'T')
            print(f'Tesouros encontrados: {treasures}')
        elif approach == 'C':
            flags = np.sum(env.grid == 'F')
            print(f'Bandeiras encontradas: {flags}')
    print()

# =============================================================================
# TESTE DE DISPERSÃO
# =============================================================================

def test_dispersion():
    """Teste da dispersão inicial dos agentes"""
    print("=== TESTE DE DISPERSÃO INICIAL ===")

    # Criar simulação com seed para reprodutibilidade
    np.random.seed(42)
    sim = Simulation('A', 3, 0.3, 'homogeneous')  # Removido seed=42

    print(f"Ambiente criado: {sim.environment.size}x{sim.environment.size}")
    print(f"Posições iniciais dos agentes: {[agent.position for agent in sim.agents]}")

    # Executar alguns passos para ver movimento
    for i in range(3):
        for agent in sim.agents:
            if agent.alive:
                move = agent.decide_next_move(sim.environment)
                if move:
                    agent.move(sim.environment, move)

    print(f"Posições após alguns movimentos: {[agent.position for agent in sim.agents]}")
    print(f"Células exploradas: {np.sum(sim.environment.explored)}")

    # Verificar se nenhum agente ficou em (0,0)
    agents_in_start = sum(1 for agent in sim.agents if agent.position == (0, 0))
    print(f"Agentes ainda em (0,0): {agents_in_start}")

    # Verificar se posições são diferentes
    positions = [agent.position for agent in sim.agents]
    unique_positions = len(set(positions))
    print(f"Posições únicas: {unique_positions}/{len(positions)}")

    print("Teste concluído!")
    print()

# =============================================================================
# EXEMPLO DE USO
# =============================================================================

def example_usage():
    """Tutorial interativo com exemplos passo a passo"""

    def print_section(title):
        """Função auxiliar para imprimir seções"""
        print(f"\n{'='*80}")
        print(f"{title}")
        print(f"{'='*80}\n")

    # 1. Simulação simples
    print_section("1. SIMULAÇÃO SIMPLES")
    print("Executando uma única simulação com configuração básica:")
    print(" - Abordagem: A")
    print(" - Agentes: 3")
    print(" - Taxa de bombas: 50%")
    print(" - Tipo de grupo: homogeneous")
    print(" - Semente: 42 para reprodutibilidade\n")

    sim = Simulation(
        approach='A',
        num_agents=3,
        bomb_ratio=0.5,
        group_type='homogeneous',
        seed=42
    )

    # Executar simulação
    print("Executando simulação...")
    result = sim.run(max_iterations=10)

    print("Resultado da simulação:")
    print(f" - Iterações: {result['iterations']}")
    print(f" - Tempo: {result['time']:.2f}s")
    print(f" - Agentes vivos: {sum(1 for a in sim.agents if a.alive)}/{len(sim.agents)}")
    print(f" - Tesouros encontrados: {sum(a.treasures_found for a in sim.agents)}")
    print(f" - Bombas desativadas: {sum(a.bombs_deactivated for a in sim.agents)}")
    print(f" - Bombas detonadas: {sum(a.bombs_activated for a in sim.agents)}")

    # 2. Experimentos
    print_section("2. EXPERIMENTOS")
    print("Executando experimento com configuração específica...")

    # Exemplo de configuração
    config = {
        'approach': 'A',
        'num_agents': 3,
        'bomb_ratio': 0.5,
        'group_type': 'homogeneous',
        'repetitions': 5
    }

    print(f"Configuração: {config}")

    # Executar experimento
    results = []
    for i in range(config['repetitions']):
        sim = Simulation(
            approach=config['approach'],
            num_agents=config['num_agents'],
            bomb_ratio=config['bomb_ratio'],
            group_type=config['group_type'],
            seed=i
        )
        result = sim.run(max_iterations=20)
        results.append(result)

    # Calcular médias
    avg_iterations = np.mean([r['iterations'] for r in results])
    avg_time = np.mean([r['time'] for r in results])

    print("Resultados médios:")
    print(f" - Iterações: {avg_iterations:.1f}")
    print(f" - Tempo: {avg_time:.2f}s")

    # 3. Visualizações
    print_section("3. VISUALIZAÇÕES")
    print("Gerando visualizações...")

    # Salvar dados da simulação
    simulation_data = {
        'config': config,
        'results': results
    }

    with open(os.path.join(RESULTS_DIR, 'example_simulation.json'), 'w') as f:
        json.dump(simulation_data, f, indent=2)

    print(f"Dados salvos em {RESULTS_DIR}/example_simulation.json")
    print("Para gerar visualizações, execute: python visualization.py")

    print_section("EXEMPLO CONCLUÍDO")
    print("Todos os exemplos foram executados com sucesso!")

# =============================================================================
# EXECUTOR DE EXPERIMENTOS
# =============================================================================

def run_experiments():
    """Executa experimentos completos"""
    print("=== EXECUTOR DE EXPERIMENTOS ===")

    # Configurações reduzidas para teste
    approaches = ['A', 'B']
    agent_counts = [2, 3]
    bomb_ratios = [0.5, 0.6]
    group_types = ['homogeneous']
    repetitions = 3  # Reduzido para teste

    # Criar diretório para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    print("="*60)
    print("INICIANDO EXPERIMENTOS DE TESTE")
    print("="*60)

    total_experiments = len(approaches) * len(agent_counts) * len(bomb_ratios) * len(group_types)
    current_experiment = 0

    # Para cada abordagem
    for approach in approaches:
        print(f"\n{'='*60}")
        print(f"ABORDAGEM {approach}")
        print(f"{'='*60}")

        approach_results = []

        # Experimentos com agentes ML
        for num_agents in agent_counts:
            for bomb_ratio in bomb_ratios:
                for group_type in group_types:
                    current_experiment += 1
                    print(f"\nExperimento {current_experiment}/{total_experiments}:")
                    print(f"  Agentes: {num_agents}, Bombas: {bomb_ratio}, Grupo: {group_type}")

                    experiment_results = []

                    for rep in range(repetitions):
                        # Criar simulação
                        sim = Simulation(
                            approach=approach,
                            num_agents=num_agents,
                            bomb_ratio=bomb_ratio,
                            group_type=group_type,
                            seed=rep
                        )

                        # Executar
                        result = sim.run(max_iterations=50)

                        # Coletar métricas
                        metrics = {
                            'approach': approach,
                            'num_agents': num_agents,
                            'bomb_ratio': bomb_ratio,
                            'group_type': group_type,
                            'repetition': rep,
                            'iterations': result['iterations'],
                            'time': result['time'],
                            'alive_agents': sum(1 for a in sim.agents if a.alive),
                            'total_treasures': sum(a.treasures_found for a in sim.agents),
                            'total_bombs_deactivated': sum(a.bombs_deactivated for a in sim.agents),
                            'total_bombs_activated': sum(a.bombs_activated for a in sim.agents)
                        }

                        experiment_results.append(metrics)

                    # Calcular médias para o experimento
                    avg_metrics = {
                        'approach': approach,
                        'num_agents': num_agents,
                        'bomb_ratio': bomb_ratio,
                        'group_type': group_type,
                        'avg_iterations': np.mean([r['iterations'] for r in experiment_results]),
                        'avg_time': np.mean([r['time'] for r in experiment_results]),
                        'avg_alive_agents': np.mean([r['alive_agents'] for r in experiment_results]),
                        'avg_treasures': np.mean([r['total_treasures'] for r in experiment_results]),
                        'avg_bombs_deactivated': np.mean([r['total_bombs_deactivated'] for r in experiment_results]),
                        'avg_bombs_activated': np.mean([r['total_bombs_activated'] for r in experiment_results])
                    }

                    approach_results.append(avg_metrics)
                    all_results.extend(experiment_results)

                    print(f"  Resultados médios: {avg_metrics['avg_iterations']:.1f} iterações, "
                          f"{avg_metrics['avg_alive_agents']:.1f} agentes vivos")

        # Salvar resultados da abordagem
        df_approach = pd.DataFrame(approach_results)
        approach_file = os.path.join(results_dir, f'results_{approach}.csv')
        df_approach.to_csv(approach_file, index=False)
        print(f"Resultados da abordagem {approach} salvos em {approach_file}")

    # Salvar todos os resultados
    df_all = pd.DataFrame(all_results)
    all_file = os.path.join(results_dir, 'all_results.csv')
    df_all.to_csv(all_file, index=False)
    print(f"Todos os resultados salvos em {all_file}")

    print(f"\n{'='*60}")
    print("EXPERIMENTOS CONCLUÍDOS")
    print(f"Resultados salvos em: {results_dir}")
    print(f"{'='*60}")

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("ARQUIVO DE TESTES CONSOLIDADO")
    print("Descomente as funções que deseja executar:\n")

    # Testes básicos
    test_simple()
    test_environment()
    test_dispersion()

    # Exemplos e experimentos (descomente se desejar executar)
    # example_usage()
    # run_experiments()

    print("\nPara executar exemplos ou experimentos completos,")
    print("descomente as chamadas no final do arquivo.")