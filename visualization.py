"""
Script para gerar visualizações profissionais dos resultados experimentais
Gera gráficos, tabelas comparativas e análises estatísticas
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def load_results(csv_file):
    """Carrega resultados de um arquivo CSV"""
    return pd.read_csv(csv_file)


def plot_execution_time_histogram(df, approach, output_dir='plots'):
    """Gera histograma de tempo de execução por grupo"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    group_types = df['group_type'].unique()
    
    for idx, group_type in enumerate(group_types):
        data = df[df['group_type'] == group_type]['execution_time']
        
        axes[idx].hist(data, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {data.mean():.4f}s')
        axes[idx].axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {data.median():.4f}s')
        
        axes[idx].set_title(f'{group_type.capitalize()}\n(n={len(data)})', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Tempo de Execução (s)')
        axes[idx].set_ylabel('Frequência')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'Distribuição de Tempo de Execução - Abordagem {approach}', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f'{output_dir}/histogram_time_abordagem_{approach}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Histograma salvo: {filename}")
    plt.close()


def plot_comparison_boxplot(df, approach, output_dir='plots'):
    """Gera boxplot comparativo entre grupos"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Tempo de execução
    sns.boxplot(data=df, x='group_type', y='execution_time', ax=axes[0,0], palette='Set2')
    axes[0,0].set_title('Tempo de Execução', fontweight='bold')
    axes[0,0].set_ylabel('Tempo (s)')
    axes[0,0].set_xlabel('')
    
    # Total de passos
    if 'total_steps' in df.columns:
        sns.boxplot(data=df, x='group_type', y='total_steps', ax=axes[0,1], palette='Set2')
        axes[0,1].set_title('Total de Passos', fontweight='bold')
        axes[0,1].set_ylabel('Passos')
        axes[0,1].set_xlabel('')
    
    # Percentagem de exploração
    if 'exploration_percentage' in df.columns:
        sns.boxplot(data=df, x='group_type', y='exploration_percentage', ax=axes[1,0], palette='Set2')
        axes[1,0].set_title('Exploração do Ambiente', fontweight='bold')
        axes[1,0].set_ylabel('Exploração (%)')
        axes[1,0].set_xlabel('')
    
    # Bombas ativadas
    if 'bombs_activated' in df.columns:
        sns.boxplot(data=df, x='group_type', y='bombs_activated', ax=axes[1,1], palette='Set2')
        axes[1,1].set_title('Bombas Ativadas', fontweight='bold')
        axes[1,1].set_ylabel('Número de Bombas')
        axes[1,1].set_xlabel('')
    
    plt.suptitle(f'Comparação de Métricas - Abordagem {approach}', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    filename = f'{output_dir}/boxplot_comparison_abordagem_{approach}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Boxplot salvo: {filename}")
    plt.close()


def plot_success_rate_bar(df, approach, output_dir='plots'):
    """Gera gráfico de barras de taxa de sucesso"""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'success' not in df.columns:
        print("⚠ Coluna 'success' não encontrada")
        return
    
    # Calcular taxa de sucesso por grupo
    success_rates = df.groupby('group_type')['success'].mean() * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(success_rates.index, success_rates.values, color=colors[:len(success_rates)], 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Taxa de Sucesso (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tipo de Grupo', fontsize=12, fontweight='bold')
    ax.set_title(f'Taxa de Sucesso por Tipo de Grupo - Abordagem {approach}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'{output_dir}/success_rate_abordagem_{approach}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de taxa de sucesso salvo: {filename}")
    plt.close()


def plot_agents_vs_performance(df, approach, output_dir='plots'):
    """Analisa impacto do número de agentes no desempenho"""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'num_agents' not in df.columns:
        print("⚠ Coluna 'num_agents' não encontrada")
        return
    
    # Filtrar apenas grupos ML (sem baseline)
    ml_df = df[df['group_type'] != 'baseline'].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Tempo médio por número de agentes
    time_by_agents = ml_df.groupby(['num_agents', 'group_type'])['execution_time'].mean().reset_index()
    
    for group_type in time_by_agents['group_type'].unique():
        data = time_by_agents[time_by_agents['group_type'] == group_type]
        axes[0].plot(data['num_agents'], data['execution_time'], marker='o', 
                     linewidth=2, markersize=8, label=group_type.capitalize())
    
    axes[0].set_xlabel('Número de Agentes', fontweight='bold')
    axes[0].set_ylabel('Tempo Médio de Execução (s)', fontweight='bold')
    axes[0].set_title('Tempo vs Número de Agentes', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Taxa de sucesso por número de agentes
    if 'success' in ml_df.columns:
        success_by_agents = ml_df.groupby(['num_agents', 'group_type'])['success'].mean().reset_index()
        success_by_agents['success'] *= 100
        
        for group_type in success_by_agents['group_type'].unique():
            data = success_by_agents[success_by_agents['group_type'] == group_type]
            axes[1].plot(data['num_agents'], data['success'], marker='s', 
                         linewidth=2, markersize=8, label=group_type.capitalize())
        
        axes[1].set_xlabel('Número de Agentes', fontweight='bold')
        axes[1].set_ylabel('Taxa de Sucesso (%)', fontweight='bold')
        axes[1].set_title('Taxa de Sucesso vs Número de Agentes', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
    
    plt.suptitle(f'Impacto do Número de Agentes - Abordagem {approach}', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    filename = f'{output_dir}/agents_vs_performance_abordagem_{approach}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Análise de agentes salva: {filename}")
    plt.close()


def generate_statistical_comparison_table(df, approach, output_dir='plots'):
    """Gera tabela estatística comparativa"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for group_type in df['group_type'].unique():
        group_df = df[df['group_type'] == group_type]
        
        row = {
            'Grupo': group_type.capitalize(),
            'N': len(group_df),
            'Tempo Médio (s)': f"{group_df['execution_time'].mean():.4f}",
            'Tempo DP': f"{group_df['execution_time'].std():.4f}",
            'Tempo Min': f"{group_df['execution_time'].min():.4f}",
            'Tempo Max': f"{group_df['execution_time'].max():.4f}"
        }
        
        if 'success' in group_df.columns:
            row['Taxa Sucesso (%)'] = f"{group_df['success'].mean() * 100:.2f}"
        
        if 'total_steps' in group_df.columns:
            row['Passos Médios'] = f"{group_df['total_steps'].mean():.2f}"
        
        if 'exploration_percentage' in group_df.columns:
            row['Exploração (%)'] = f"{group_df['exploration_percentage'].mean():.2f}"
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    
    # Salvar como CSV
    csv_filename = f'{output_dir}/tabela_estatisticas_abordagem_{approach}.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"✓ Tabela estatística salva: {csv_filename}")
    
    # Criar visualização da tabela
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=results_df.values, colLabels=results_df.columns,
                     cellLoc='center', loc='center', colColours=['#4CAF50']*len(results_df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.title(f'Estatísticas Comparativas - Abordagem {approach}', 
              fontsize=14, fontweight='bold', pad=20)
    
    png_filename = f'{output_dir}/tabela_estatisticas_abordagem_{approach}.png'
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Imagem da tabela salva: {png_filename}")
    plt.close()
    
    return results_df


def perform_statistical_tests(df, approach, output_dir='plots'):
    """Realiza testes estatísticos entre grupos"""
    os.makedirs(output_dir, exist_ok=True)
    
    report = []
    report.append(f"{'='*80}")
    report.append(f"TESTES ESTATÍSTICOS - ABORDAGEM {approach}")
    report.append(f"{'='*80}\n")
    
    groups = df['group_type'].unique()
    
    if len(groups) < 2:
        report.append("⚠ Necessário pelo menos 2 grupos para comparação estatística\n")
        return
    
    # Teste de normalidade (Shapiro-Wilk)
    report.append("1. TESTE DE NORMALIDADE (Shapiro-Wilk)")
    report.append("-" * 80)
    
    for group in groups:
        data = df[df['group_type'] == group]['execution_time']
        stat, p_value = stats.shapiro(data)
        normal = "SIM" if p_value > 0.05 else "NÃO"
        report.append(f"{group.capitalize()}: W={stat:.4f}, p-valor={p_value:.4f} -> Normal? {normal}")
    
    report.append("")
    
    # Teste de Mann-Whitney U (não-paramétrico) entre pares
    report.append("2. TESTE DE MANN-WHITNEY U (comparações pareadas)")
    report.append("-" * 80)
    
    from itertools import combinations
    
    for g1, g2 in combinations(groups, 2):
        data1 = df[df['group_type'] == g1]['execution_time']
        data2 = df[df['group_type'] == g2]['execution_time']
        
        stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        significant = "SIM (p < 0.05)" if p_value < 0.05 else "NÃO (p >= 0.05)"
        
        report.append(f"{g1} vs {g2}:")
        report.append(f"  U-statistic = {stat:.2f}")
        report.append(f"  p-valor = {p_value:.4f}")
        report.append(f"  Diferença significativa? {significant}\n")
    
    # Teste de Kruskal-Wallis (mais de 2 grupos)
    if len(groups) >= 3:
        report.append("3. TESTE DE KRUSKAL-WALLIS (comparação múltipla)")
        report.append("-" * 80)
        
        group_data = [df[df['group_type'] == g]['execution_time'].values for g in groups]
        stat, p_value = stats.kruskal(*group_data)
        significant = "SIM (p < 0.05)" if p_value < 0.05 else "NÃO (p >= 0.05)"
        
        report.append(f"H-statistic = {stat:.4f}")
        report.append(f"p-valor = {p_value:.4f}")
        report.append(f"Diferença significativa entre grupos? {significant}\n")
    
    # Salvar relatório
    report_text = "\n".join(report)
    report_filename = f'{output_dir}/testes_estatisticos_abordagem_{approach}.txt'
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Relatório de testes estatísticos salvo: {report_filename}")
    print("\n" + report_text)


def generate_all_visualizations(csv_file, approach):
    """Gera todas as visualizações de uma vez"""
    print(f"\n{'='*80}")
    print(f"GERANDO VISUALIZAÇÕES PARA ABORDAGEM {approach}")
    print(f"{'='*80}\n")
    
    df = load_results(csv_file)
    
    output_dir = f'plots/abordagem_{approach}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Dados carregados: {len(df)} execuções")
    print(f"Grupos encontrados: {', '.join(df['group_type'].unique())}\n")
    
    # Gerar visualizações
    plot_execution_time_histogram(df, approach, output_dir)
    plot_comparison_boxplot(df, approach, output_dir)
    plot_success_rate_bar(df, approach, output_dir)
    plot_agents_vs_performance(df, approach, output_dir)
    
    # Gerar tabela estatística
    stats_df = generate_statistical_comparison_table(df, approach, output_dir)
    print("\nTabela de Estatísticas:")
    print(stats_df.to_string(index=False))
    
    # Realizar testes estatísticos
    perform_statistical_tests(df, approach, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✓ VISUALIZAÇÕES CONCLUÍDAS!")
    print(f"✓ Arquivos salvos em: {output_dir}/")
    print(f"{'='*80}\n")


# Exemplo de uso
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     GERADOR DE VISUALIZAÇÕES E ANÁLISES ESTATÍSTICAS        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Exemplo: gerar visualizações para todos os arquivos de resultado
    results_dir = 'results'
    
    if os.path.exists(results_dir):
        csv_files = [f for f in os.listdir(results_dir) if f.startswith('abordagem_') and f.endswith('.csv')]
        
        for csv_file in csv_files:
            # Extrair abordagem do nome do arquivo
            approach = csv_file.split('_')[1].split('.')[0]
            
            csv_path = os.path.join(results_dir, csv_file)
            generate_all_visualizations(csv_path, approach)
    else:
        print(f"⚠ Diretório '{results_dir}' não encontrado.")
        print("Execute primeiro os experimentos para gerar os dados.")
