"""
Script para executar todos os experimentos e gerar CSVs
Executa 30 repetições para cada configuração
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import sys

# Adicionar o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Importar do arquivo principal
from simulation import Simulation, run_experiment

def run_all_experiments():
    """Executa todos os experimentos configurados"""
    
    # Configurações
    approaches = ['A', 'B', 'C']
    agent_counts = [2, 3, 5, 7, 10]
    bomb_ratios = [0.5, 0.6, 0.7, 0.8]
    group_types = ['homogeneous', 'heterogeneous']
    repetitions = 30
    
    # Criar diretório para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    print("="*60)
    print("INICIANDO EXPERIMENTOS COMPLETOS")
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
                    print(f"\n[{current_experiment}/{total_experiments}] "
                          f"Abordagem {approach} | Agentes: {num_agents} | "
                          f"Bombas: {bomb_ratio*100:.0f}% | Grupo: {group_type}")
                    
                    # Executar 30 repetições
                    results_df = run_experiment(
                        approach=approach,
                        num_agents=num_agents,
                        bomb_ratio=bomb_ratio,
                        group_type=group_type,
                        repetitions=repetitions
                    )
                    
                    # Adicionar à lista geral
                    approach_results.append(results_df)
                    
                    # Calcular estatísticas
                    success_rate = results_df['success'].mean() * 100
                    avg_time = results_df['execution_time'].mean()
                    
                    print(f"  ✓ Taxa de sucesso: {success_rate:.1f}%")
                    print(f"  ✓ Tempo médio: {avg_time:.3f}s")
        
        # Experimentos com baseline
        print(f"\n  Executando baseline para Abordagem {approach}...")
        baseline_results = []
        
        for bomb_ratio in bomb_ratios:
            for rep in range(repetitions):
                sim = Simulation(approach=approach, num_agents=1, bomb_ratio=bomb_ratio, group_type='baseline', seed=rep)
                metrics = sim.run()
                metrics['repetition'] = rep
                metrics['num_agents'] = 1
                baseline_results.append(metrics)
        
        baseline_df = pd.DataFrame(baseline_results)
        approach_results.append(baseline_df)
        
        # Combinar todos os resultados da abordagem
        combined_df = pd.concat(approach_results, ignore_index=True)
        
        # Salvar CSV da abordagem
        csv_filename = f"{results_dir}/abordagem_{approach}.csv"
        combined_df.to_csv(csv_filename, index=False)
        print(f"\n  ✓ Resultados salvos: {csv_filename}")
        
        # Adicionar à lista geral
        all_results.append(combined_df)
    
    # Combinar TODOS os resultados
    final_df = pd.concat(all_results, ignore_index=True)
    final_csv = f"{results_dir}/resultados_completos.csv"
    final_df.to_csv(final_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENTOS CONCLUÍDOS!")
    print(f"{'='*60}")
    print(f"Total de execuções: {len(final_df)}")
    print(f"Arquivo final: {final_csv}")
    
    # Gerar relatório estatístico
    generate_statistical_report(final_df, results_dir)
    
    return results_dir


def generate_statistical_report(df, results_dir):
    """Gera relatório estatístico dos experimentos"""
    
    report = []
    report.append("="*80)
    report.append("RELATÓRIO ESTATÍSTICO DOS EXPERIMENTOS")
    report.append("="*80)
    report.append("")
    
    # Para cada abordagem
    for approach in ['A', 'B', 'C']:
        approach_df = df[df['approach'] == approach]
        
        report.append(f"\n{'='*80}")
        report.append(f"ABORDAGEM {approach}")
        report.append(f"{'='*80}\n")
        
        # Comparação Homogêneo vs Heterogêneo vs Baseline
        for group_type in ['homogeneous', 'heterogeneous', 'baseline']:
            group_df = approach_df[approach_df['group_type'] == group_type]
            
            if len(group_df) == 0:
                continue
            
            report.append(f"\n{'-'*80}")
            report.append(f"{group_type.upper()}")
            report.append(f"{'-'*80}")
            
            # Estatísticas gerais
            if 'success' in group_df.columns:
                success_rate = group_df['success'].mean() * 100
                report.append(f"Taxa de Sucesso: {success_rate:.2f}%")
            
            avg_time = group_df['execution_time'].mean()
            std_time = group_df['execution_time'].std()
            min_time = group_df['execution_time'].min()
            max_time = group_df['execution_time'].max()
            
            report.append(f"Tempo de Execução:")
            report.append(f"  Média: {avg_time:.4f}s")
            report.append(f"  Desvio Padrão: {std_time:.4f}s")
            report.append(f"  Mínimo: {min_time:.4f}s")
            report.append(f"  Máximo: {max_time:.4f}s")
            
            if 'total_steps' in group_df.columns:
                avg_steps = group_df['total_steps'].mean()
                report.append(f"Passos Médios: {avg_steps:.2f}")
            
            if 'exploration_percentage' in group_df.columns:
                avg_exploration = group_df['exploration_percentage'].mean()
                report.append(f"Exploração Média: {avg_exploration:.2f}%")
            
            if approach == 'A' and 'treasures_found' in group_df.columns:
                avg_treasures = group_df['treasures_found'].mean()
                report.append(f"Tesouros Encontrados (média): {avg_treasures:.2f}")
            
            if 'bombs_activated' in group_df.columns:
                avg_bombs = group_df['bombs_activated'].mean()
                report.append(f"Bombas Ativadas (média): {avg_bombs:.2f}")
        
        # Análise por número de agentes (apenas ML)
        report.append(f"\n{'-'*80}")
        report.append("ANÁLISE POR NÚMERO DE AGENTES")
        report.append(f"{'-'*80}\n")
        
        ml_df = approach_df[approach_df['group_type'] != 'baseline']
        
        if 'num_agents' in ml_df.columns:
            for num_agents in sorted(ml_df['num_agents'].unique()):
                agent_df = ml_df[ml_df['num_agents'] == num_agents]
                
                if 'success' in agent_df.columns:
                    success_rate = agent_df['success'].mean() * 100
                else:
                    success_rate = 0
                
                avg_time = agent_df['execution_time'].mean()
                
                report.append(f"{num_agents} agentes: Taxa sucesso={success_rate:.1f}%, Tempo={avg_time:.4f}s")
    
    # Comparação Final
    report.append(f"\n{'='*80}")
    report.append("COMPARAÇÃO GERAL: HOMOGÊNEO vs HETEROGÊNEO vs BASELINE")
    report.append(f"{'='*80}\n")
    
    summary_data = []
    
    for approach in ['A', 'B', 'C']:
        approach_df = df[df['approach'] == approach]
        
        row = {'Abordagem': approach}
        
        for group_type in ['homogeneous', 'heterogeneous', 'baseline']:
            group_df = approach_df[approach_df['group_type'] == group_type]
            
            if len(group_df) > 0:
                if 'success' in group_df.columns:
                    success = group_df['success'].mean() * 100
                else:
                    success = 0
                
                time_avg = group_df['execution_time'].mean()
                
                row[f'{group_type}_success'] = f"{success:.1f}%"
                row[f'{group_type}_time'] = f"{time_avg:.4f}s"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    report.append(summary_df.to_string(index=False))
    
    # Salvar relatório
    report_text = "\n".join(report)
    report_file = f"{results_dir}/relatorio_estatistico.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ Relatório estatístico salvo: {report_file}")
    
    # Também imprimir no console
    print("\n" + report_text)


def analyze_specific_configuration(csv_file, approach, num_agents, bomb_ratio, group_type):
    """Analisa uma configuração específica"""
    
    df = pd.read_csv(csv_file)
    
    # Filtrar dados
    filtered = df[
        (df['approach'] == approach) &
        (df['num_agents'] == num_agents) &
        (df['bomb_ratio'] == bomb_ratio) &
        (df['group_type'] == group_type)
    ]
    
    if len(filtered) == 0:
        print("Nenhum dado encontrado para essa configuração.")
        return
    
    print(f"\n{'='*60}")
    print(f"Análise: Abordagem {approach} | {num_agents} agentes | "
          f"Bombas {bomb_ratio*100:.0f}% | {group_type}")
    print(f"{'='*60}\n")
    
    print(f"Repetições: {len(filtered)}")
    
    if 'success' in filtered.columns:
        success_rate = filtered['success'].mean() * 100
        print(f"Taxa de Sucesso: {success_rate:.2f}%")
    
    print(f"\nTempo de Execução:")
    print(f"  Média: {filtered['execution_time'].mean():.4f}s")
    print(f"  Mínimo: {filtered['execution_time'].min():.4f}s")
    print(f"  Máximo: {filtered['execution_time'].max():.4f}s")
    print(f"  Desvio Padrão: {filtered['execution_time'].std():.4f}s")
    
    if 'total_steps' in filtered.columns:
        print(f"\nPassos Totais:")
        print(f"  Média: {filtered['total_steps'].mean():.2f}")
        print(f"  Mínimo: {filtered['total_steps'].min()}")
        print(f"  Máximo: {filtered['total_steps'].max()}")
    
    if 'exploration_percentage' in filtered.columns:
        print(f"\nExploração:")
        print(f"  Média: {filtered['exploration_percentage'].mean():.2f}%")
    
    if approach == 'A' and 'treasures_found' in filtered.columns:
        print(f"\nTesouros:")
        print(f"  Média: {filtered['treasures_found'].mean():.2f}")
    
    return filtered


# Função principal
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║   SISTEMA DE EXPERIMENTAÇÃO - AGENTES INTELIGENTES          ║
    ║   Projeto de IA - ISPTEC 2024/2025                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\nEste script executará:")
    print("  • 3 Abordagens (A, B, C)")
    print("  • 5 configurações de agentes (2, 3, 5, 7, 10)")
    print("  • 4 níveis de bombas (50%, 60%, 70%, 80%)")
    print("  • 2 tipos de grupo (homogêneo, heterogêneo)")
    print("  • + Baseline para cada abordagem")
    print("  • 30 repetições por configuração")
    print(f"\n  Total estimado: ~{3*5*4*2*30 + 3*4*30} execuções\n")
    
    response = input("Deseja iniciar os experimentos? (s/n): ")
    
    if response.lower() == 's':
        results_dir = run_all_experiments()
        print(f"\n✓ Todos os resultados salvos em: {results_dir}/")
    else:
        print("\nExperimentos cancelados.")