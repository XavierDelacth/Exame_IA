from simulation import Environment
import numpy as np

# Testar geração de ambiente para cada abordagem
for approach in ['A', 'B', 'C']:
    print(f'\n=== Testando abordagem {approach} ===')
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