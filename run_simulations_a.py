"""
Script para executar 30 simulações por grupo (A):
- homogeneous
- heterogeneous
- baseline
Salva resultados em simulation_results/all_results.json via DataStorage
"""
from analise.comparative_analysis import DataStorage
from analise.comparative_analysis import MetricsCalculator
from time import sleep

def main(runs_per_group=30, num_agents=4, bomb_ratio=0.3, treasure_count=12, max_steps=300):
    storage = DataStorage()
    calc = MetricsCalculator()

    groups = ['homogeneous', 'heterogeneous', 'baseline']

    for group in groups:
        for i in range(runs_per_group):
            if group == 'baseline':
                from abordagem.abordagem_a import BaselineSimulation as SimClass
                sim = SimClass(num_agents=num_agents, bomb_ratio=bomb_ratio, treasure_count=treasure_count, max_steps=max_steps)
                metrics = sim.run_simulation(verbose=False)
            else:
                from abordagem.abordagem_a import ApproachASimulation as SimClass
                sim = SimClass(num_agents=num_agents, bomb_ratio=bomb_ratio, treasure_count=treasure_count, homogeneous=(group=='homogeneous'), max_steps=max_steps)
                metrics = sim.run_simulation(verbose=False)

            sim_metrics = getattr(sim, 'metrics', metrics if isinstance(metrics, dict) else {})
            # Compute processed metrics using MetricsCalculator
            processed_metrics = calc.calculate_approach_a_metrics(sim)

            parameters = {
                'num_agents': num_agents,
                'bomb_ratio': bomb_ratio,
                'treasure_count': treasure_count,
                'max_steps': max_steps,
                'homogeneous': (group == 'homogeneous')
            }
            storage.save_result('A', group, processed_metrics, parameters)
            print(f"[A] Saved run {i+1}/{runs_per_group} for group {group}")
            sleep(0.01)

if __name__ == '__main__':
    main()
