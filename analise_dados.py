import json
import pandas as pd
import numpy as np

with open('simulation_results/all_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("ANÁLISE DETALHADA POR ABORDAGEM E GRUPO")
print("=" * 80)

# Abordagem A
print("\n--- ABORDAGEM A: COLETA DE TESOUROS ---\n")
a_df = pd.DataFrame([
    {
        'group': entry['group_type'],
        'treasure_col': entry['metrics'].get('treasure_percentage', 0),
        'success': entry['metrics'].get('success_rate', 0),
        'efficiency': entry['metrics'].get('exploration_efficiency', 0),
        'reward_risk': entry['metrics'].get('reward_risk_ratio', 0),
        'steps_tesouro': entry['metrics'].get('avg_steps_to_treasure', 0),
    }
    for entry in data['A']
])

for group in ['baseline', 'homogeneous', 'heterogeneous']:
    subset = a_df[a_df['group'] == group]
    print(f"{group.upper()}:")
    print(f"  Tesouro coletado: {subset['treasure_col'].mean():.2f}% (±{subset['treasure_col'].std():.2f})")
    print(f"  Taxa sucesso: {subset['success'].mean():.2f}% (±{subset['success'].std():.2f})")
    print(f"  Eficiência exploração: {subset['efficiency'].mean():.4f} (±{subset['efficiency'].std():.4f})")
    print(f"  Razão recompensa-risco: {subset['reward_risk'].mean():.2f} (±{subset['reward_risk'].std():.2f})")
    print()

# Abordagem B
print("\n--- ABORDAGEM B: EXPLORAÇÃO COMPLETA ---\n")
b_df = pd.DataFrame([
    {
        'group': entry['group_type'],
        'explored': entry['metrics'].get('explored_percentage', 0),
        'vivos': entry['metrics'].get('agents_alive', 0),
        'cells_por_passo': entry['metrics'].get('cells_per_step', 0),
    }
    for entry in data['B']
])

for group in ['baseline', 'homogeneous', 'heterogeneous']:
    subset = b_df[b_df['group'] == group]
    print(f"{group.upper()}:")
    print(f"  % Explorado: {subset['explored'].mean():.2f}% (±{subset['explored'].std():.2f})")
    print(f"  Agentes vivos: {subset['vivos'].mean():.2f} (±{subset['vivos'].std():.2f})")
    print(f"  Células por passo: {subset['cells_por_passo'].mean():.4f} (±{subset['cells_por_passo'].std():.4f})")
    print()

# Abordagem C
print("\n--- ABORDAGEM C: LOCALIZAÇÃO DE BANDEIRA ---\n")
c_df = pd.DataFrame([
    {
        'group': entry['group_type'],
        'success': entry['metrics'].get('success_rate', 0),
        'steps_bandeira': entry['metrics'].get('avg_steps_to_flag', 0),
        'path_eff': entry['metrics'].get('path_efficiency', 0),
    }
    for entry in data['C']
])

for group in ['baseline', 'homogeneous', 'heterogeneous']:
    subset = c_df[c_df['group'] == group]
    print(f"{group.upper()}:")
    print(f"  Taxa sucesso: {subset['success'].mean():.2f}% (±{subset['success'].std():.2f})")
    print(f"  Steps até bandeira: {subset['steps_bandeira'].mean():.2f} (±{subset['steps_bandeira'].std():.2f})")
    print(f"  Eficiência caminho: {subset['path_eff'].mean():.2f} (±{subset['path_eff'].std():.2f})")
    print()

print("\n" + "=" * 80)
print("RANKING GERAL (melhor grupo por métrica principal)")
print("=" * 80)

print("\nAbordagem A - Melhor em coleta de tesouro:")
melhor_a = a_df.groupby('group')['treasure_col'].mean().idxmax()
print(f"  {melhor_a}: {a_df[a_df['group']==melhor_a]['treasure_col'].mean():.2f}%")

print("\nAbordagem B - Melhor em exploração:")
melhor_b = b_df.groupby('group')['explored'].mean().idxmax()
print(f"  {melhor_b}: {b_df[b_df['group']==melhor_b]['explored'].mean():.2f}%")

print("\nAbordagem C - Melhor em encontrar bandeira:")
melhor_c = c_df.groupby('group')['success'].mean().idxmax()
print(f"  {melhor_c}: {c_df[c_df['group']==melhor_c]['success'].mean():.2f}%")

# Análise de variação com número de agentes
print("\n" + "=" * 80)
print("ANÁLISE DE VARIAÇÃO DO NÚMERO DE AGENTES")
print("=" * 80)

for approach in ['A', 'B', 'C']:
    entries = data[approach]
    agents_count = {}
    
    for entry in entries:
        num_agents = entry['parameters'].get('num_agents')
        group = entry['group_type']
        
        if num_agents not in agents_count:
            agents_count[num_agents] = {'homogeneous': [], 'heterogeneous': [], 'baseline': []}
    
    # Verificar se há variação
    num_agents_set = set([e['parameters'].get('num_agents') for e in entries])
    print(f"\nAbordagem {approach}: Números de agentes encontrados: {sorted(num_agents_set)}")
    if len(num_agents_set) == 1:
        print(f"  (Apenas um número de agentes testado: {list(num_agents_set)[0]})")
