"""
example_usage.py

Tutorial interativo com exemplos passo a passo de uso do sistema.
Este script demonstra como utilizar as principais funcionalidades do projeto.

Para executar:
python example_usage.py

Cada seção pode ser executada independentemente, mas é recomendável rodar sequencialmente.
"""

import pandas as pd
import os
import time
import sys

# Adicionar o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Importar módulos do projeto
from simulation import Simulation, run_experiment, run_baseline
from visualization import generate_all_visualizations
from experiment_runner import analyze_specific_configuration

# Diretórios
RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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

metrics = sim.run()

print("Métricas obtidas:")
for key, value in metrics.items():
    print(f"  {key}: {value}")
print("\nSucesso alcançado?" if metrics['success'] else "\nFalha na missão.")

# Salvar métricas
simple_sim_file = os.path.join(RESULTS_DIR, 'simple_simulation.json')
with open(simple_sim_file, 'w') as f:
    import json
    json.dump(metrics, f, indent=4)
print(f"✓ Resultados salvos em: {simple_sim_file}")

# 2. Experimento com 30 repetições
print_section("2. EXPERIMENTO COM 30 REPETIÇÕES")
print("Executando 30 repetições independentes:")
print(" - Abordagem: B")
print(" - Agentes: 5")
print(" - Taxa de bombas: 60%")
print(" - Tipo de grupo: heterogeneous\n")

results_df = run_experiment(
    approach='B',
    num_agents=5,
    bomb_ratio=0.6,
    group_type='heterogeneous',
    repetitions=30
)

print("Estatísticas resumidas:")
print(results_df.describe())

# Salvar CSV
experiment_csv = os.path.join(RESULTS_DIR, 'experiment_30_reps.csv')
results_df.to_csv(experiment_csv, index=False)
print(f"✓ CSV salvo em: {experiment_csv}")

# 3. Algoritmo baseline
print_section("3. ALGORITMO BASELINE")
print("Executando algoritmo clássico baseline:")
print(" - Abordagem: C")
print(" - Taxa de bombas: 70%")
print(" - Semente: 0\n")

baseline_metrics = run_baseline(
    approach='C',
    bomb_ratio=0.7,
    seed=0
)

print("Métricas do baseline:")
for key, value in baseline_metrics.items():
    print(f"  {key}: {value}")

# 4. Geração de visualizações
print_section("4. GERAÇÃO DE VISUALIZAÇÕES")
print("Gerando visualizações a partir de um arquivo CSV existente.")
print("Assumindo que o CSV do experimento anterior existe.\n")

# Gerar visualizações para o CSV gerado na seção 2
generate_all_visualizations(
    csv_file=experiment_csv,
    approach='B'  # Deve corresponder à abordagem do CSV
)

print("\nVerifique o diretório 'plots/abordagem_B' para os gráficos gerados.")

# 5. Comparação homogêneo vs heterogêneo
print_section("5. COMPARAÇÃO HOMOGÊNEO vs HETEROGÊNEO")
print("Executando experimentos para comparação:")
print(" - Abordagem: A")
print(" - Agentes: 4")
print(" - Taxa de bombas: 50%")
print(" - 10 repetições por tipo de grupo (para demonstração rápida)\n")

# Homogêneo
homo_df = run_experiment(
    approach='A',
    num_agents=4,
    bomb_ratio=0.5,
    group_type='homogeneous',
    repetitions=10
)

# Heterogêneo
hetero_df = run_experiment(
    approach='A',
    num_agents=4,
    bomb_ratio=0.5,
    group_type='heterogeneous',
    repetitions=10
)

print("Comparação de taxas de sucesso:")
print(f"  Homogêneo: {homo_df['success'].mean() * 100:.2f}%")
print(f"  Heterogêneo: {hetero_df['success'].mean() * 100:.2f}%")

print("\nComparação de tempo médio:")
print(f"  Homogêneo: {homo_df['execution_time'].mean():.4f}s")
print(f"  Heterogêneo: {hetero_df['execution_time'].mean():.4f}s")

# Salvar CSVs para análise posterior
homo_csv = os.path.join(RESULTS_DIR, 'homogeneous_comparison.csv')
hetero_csv = os.path.join(RESULTS_DIR, 'heterogeneous_comparison.csv')
homo_df.to_csv(homo_csv, index=False)
hetero_df.to_csv(hetero_csv, index=False)
print(f"✓ CSVs de comparação salvos em: {RESULTS_DIR}")

# 6. Análise de impacto de agentes
print_section("6. ANÁLISE DE IMPACTO DO NÚMERO DE AGENTES")
print("Analisando impacto do número de agentes no desempenho.")
print(" - Abordagem: C")
print(" - Taxa de bombas: 80%")
print(" - Tipo de grupo: homogeneous")
print(" - Números de agentes testados: 2, 5, 10")
print(" - 5 repetições por configuração (para demonstração rápida)\n")

agent_counts = [2, 5, 10]
impact_results = {}

for num_agents in agent_counts:
    df = run_experiment(
        approach='C',
        num_agents=num_agents,
        bomb_ratio=0.8,
        group_type='homogeneous',
        repetitions=5
    )
    impact_results[num_agents] = {
        'success_rate': df['success'].mean() * 100,
        'avg_time': df['execution_time'].mean(),
        'avg_exploration': df['exploration_percentage'].mean()
    }

print("Resultados:")
for num, res in impact_results.items():
    print(f"  {num} agentes:")
    print(f"    Taxa de sucesso: {res['success_rate']:.2f}%")
    print(f"    Tempo médio: {res['avg_time']:.4f}s")
    print(f"    Exploração média: {res['avg_exploration']:.2f}%\n")

print_section("FIM DO TUTORIAL")
print("Todos os exemplos foram executados com sucesso!")
print("Verifique os diretórios 'results/' e 'plots/' para os arquivos gerados.")
print("Para análises mais profundas, use o script experiment_runner.py para experimentos completos.")