from simulation import Simulation
import time

# Testar simulação com parada manual
print('Testando simulação com parada manual...')

sim = Simulation('A', 3, 0.3, 'homogeneous')  # Menos bombas
sim.collect_states = True

# Definir tempos para evitar erro
sim.metrics['start_time'] = time.time()

# Simular alguns passos
iteration = 0
max_iterations = 50

while iteration < max_iterations:
    iteration += 1

    # Cada agente decide e move
    any_agent_moved = False
    for agent in sim.agents:
        if agent.alive:
            next_move = agent.decide_next_move(sim.environment)
            if next_move:
                agent.move(sim.environment, next_move)
                any_agent_moved = True

    # Salva estado
    if sim.collect_states:
        sim.states.append(sim._get_current_state())

    # Forçar parada na iteração 2 (antes que morram)
    if iteration == 2:
        print(f'Forçando parada na iteração {iteration}')
        sim.stop_simulation()
        print(f'Flag force_stop após chamada: {sim.force_stop}')

    print(f'Iteração {iteration}: agentes vivos = {sum(1 for a in sim.agents if a.alive)}, force_stop = {sim.force_stop}')

    # Verificar condições de parada
    if sim._check_termination():
        print(f'Simulação parada na iteração {iteration}')
        break

    if not any_agent_moved:
        print(f'Sem movimentos possíveis na iteração {iteration}')
        break

# Definir tempo final
sim.metrics['end_time'] = time.time()

# Calcular métricas
sim._calculate_metrics()

print('Resultados:')
print(f'  Iterações executadas: {iteration}')
print(f'  Agentes vivos: {sim.metrics["agents_alive"]}/{sim.metrics["num_agents"]}')
print(f'  Todos agentes mortos: {all(not agent.alive for agent in sim.agents)}')
print(f'  Forçada parada: {sim.force_stop}')
print(f'  Estados coletados: {len(sim.states)}')