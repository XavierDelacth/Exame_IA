from abordagem.abordagem_a import ApproachASimulation

print('Testando abordagem B com algoritmo greedy...')
sim = ApproachASimulation(num_agents=2, bomb_ratio=0.3, treasure_count=0, homogeneous=True, max_steps=10, approach='B')

# Verificar estado inicial
print(f"Posição inicial agentes: {[agent.position for agent in sim.agents]}")
print(f"Célula inicial (0,0): {sim.env.get_cell(0, 0)}")
print(f"Vizinhos de (0,0): {sim.env.get_neighbors(0, 0)}")

# Verificar quais vizinhos são considerados seguros
safe_neighbors = []
for neighbor in sim.env.get_neighbors(0, 0):
    is_safe = sim.shared_memory.is_safe_cell(neighbor)
    explored = sim.shared_memory.cell_knowledge[neighbor]['explored']
    print(f"Vizinho {neighbor}: seguro={is_safe}, explorado={explored}")

result = sim.run_simulation(verbose=True)

print(f'\nResultado: {result["success"]}')
print(f'Tempo: {result["execution_time"]:.2f}s')
print(f'Passos: {result["steps_taken"]}')
print(f'Explorado: {result["explored_percentage"]:.1f}%')
print(f'Agentes vivos: {result["agents_alive"]}')

# Mostrar logs
print('\n=== LOGS DA SIMULAÇÃO ===')
for log in sim.logs[-20:]:  # Últimos 20 logs
    print(log)