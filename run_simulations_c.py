"""
Script para executar 30 simulações por grupo (C):
- homogeneous
- heterogeneous
- baseline
Salva resultados em simulation_results/all_results.json via DataStorage
"""
from analise.comparative_analysis import DataStorage
from analise.comparative_analysis import MetricsCalculator
from time import sleep

def main(runs_per_group=30, num_agents=4, bomb_ratio=0.3, treasure_count=10, max_steps=300):
    storage = DataStorage()
    calc = MetricsCalculator()

    groups = ['homogeneous', 'heterogeneous', 'baseline']

    for group in groups:
        for i in range(runs_per_group):
            if group == 'baseline':
                # Baseline C: A*
                from abordagem.abordagem_a import BaselineC_AStar as SimClass
                sim = SimClass(bomb_ratio=bomb_ratio, treasure_count=treasure_count, max_steps=max_steps)
                metrics = sim.run()
                class _Wrap: pass
                sim_wrap = _Wrap()
                sim_wrap.env = sim.env
                sim_wrap.metrics = metrics
                sim = sim_wrap
            else:
                from abordagem.abordagem_c import ApproachCSimulation as SimClass
                sim = SimClass(num_agents=num_agents, bomb_ratio=bomb_ratio, treasure_count=treasure_count, homogeneous=(group=='homogeneous'), max_steps=max_steps)
                metrics = sim.run_simulation(verbose=False)

            sim_metrics = getattr(sim, 'metrics', metrics if isinstance(metrics, dict) else {})
            # Compute processed metrics using MetricsCalculator
            processed_metrics = calc.calculate_approach_c_metrics(sim)

            parameters = {
                'num_agents': num_agents,
                'bomb_ratio': bomb_ratio,
                'treasure_count': treasure_count,
                'max_steps': max_steps,
                'homogeneous': (group == 'homogeneous')
            }
            storage.save_result('C', group, processed_metrics, parameters)
            print(f"[C] Saved run {i+1}/{runs_per_group} for group {group}")
            sleep(0.01)

if __name__ == '__main__':
    main()
