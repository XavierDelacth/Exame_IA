"""
Script para executar 30 simulações por grupo (B):
- homogeneous
- heterogeneous
- baseline
Salva resultados em simulation_results/all_results.json via DataStorage
"""
from analise.comparative_analysis import DataStorage
from analise.comparative_analysis import MetricsCalculator
from time import sleep

def main(runs_per_group=30, num_agents=4, bomb_ratio=0.3, max_steps=300):
    storage = DataStorage()
    calc = MetricsCalculator()

    groups = ['homogeneous', 'heterogeneous', 'baseline']

    for group in groups:
        for i in range(runs_per_group):
            if group == 'baseline':
                # Baseline B: BFS explorer
                from abordagem.abordagem_a import BaselineB_BFS as SimClass
                sim_raw = SimClass(bomb_ratio=bomb_ratio, max_steps=max_steps)
                metrics = sim_raw.run()
                # wrap into simulation-like object expected by MetricsCalculator
                class _Wrap:
                    pass
                sim = _Wrap()
                sim.env = sim_raw.env
                sim.metrics = metrics
                # provide minimal shared_memory expected by calculators
                sim.shared_memory = type('M', (), {})()
                sim.shared_memory.bombs_found = set()
                sim.shared_memory.explored = set()
                sim.shared_memory.treasures_collected = set()
                # populate explored set size from metrics if possible
                try:
                    explored_pct = metrics.get('explored_percentage', 0)
                    total_cells = sim.env.size * sim.env.size
                    bomb_count = sum(1 for row in sim.env.grid for cell in row if cell == 'B')
                    free_cells = total_cells - bomb_count
                    sim.shared_memory.explored = set(range(int((explored_pct/100.0) * free_cells)))
                except Exception:
                    pass
            else:
                from abordagem.abordagem_b import ApproachBSimulation as SimClass
                sim = SimClass(num_agents=num_agents, bomb_ratio=bomb_ratio, homogeneous=(group=='homogeneous'), max_steps=max_steps)
                metrics = sim.run_simulation() if hasattr(sim, 'run_simulation') else sim.run()

            sim_metrics = getattr(sim, 'metrics', metrics if isinstance(metrics, dict) else {})
            # Compute processed metrics using MetricsCalculator
            processed_metrics = calc.calculate_approach_b_metrics(sim)

            parameters = {
                'num_agents': num_agents,
                'bomb_ratio': bomb_ratio,
                'treasure_count': 0,
                'max_steps': max_steps,
                'homogeneous': (group == 'homogeneous')
            }
            storage.save_result('B', group, processed_metrics, parameters)
            print(f"[B] Saved run {i+1}/{runs_per_group} for group {group}")
            sleep(0.01)

if __name__ == '__main__':
    main()
