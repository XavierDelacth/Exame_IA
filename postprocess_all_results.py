"""
Post-process existing simulation_results/all_results.json entries to ensure
metrics contain the calculated keys expected by comparative analysis.

This script will load the JSON, compute processed metrics using
MetricsCalculator for each entry (when possible) and overwrite the file.
"""
from pathlib import Path
import json
from analise.comparative_analysis import DataStorage, MetricsCalculator

def postprocess(path='simulation_results/all_results.json'):
    p = Path(path)
    if not p.exists():
        print('No file to postprocess')
        return

    with open(p, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print('Failed to load JSON:', e)
            return

    calc = MetricsCalculator()

    changed = False
    for approach in ['A','B','C']:
        entries = data.get(approach, [])
        for i, entry in enumerate(entries):
            # Skip if already contains main metrics
            metrics = entry.get('metrics', {})
            if approach == 'A' and 'treasure_percentage' in metrics:
                continue
            if approach == 'B' and 'explored_percentage' in metrics and 'safe_exploration_rate' in metrics:
                continue
            if approach == 'C' and 'success_rate' in metrics:
                continue

            # Build a lightweight simulation-like object if possible
            class SimObj:
                pass

            sim = SimObj()
            # Attach env if possible
            params = entry.get('parameters', {})
            try:
                # Try to reconstruct env from saved parameters minimally
                from abordagem import abordagem_a, abordagem_b, abordagem_c
                if approach == 'A':
                    sim.env = abordagem_a.Environment(bomb_ratio=params.get('bomb_ratio',0.3), treasure_count=params.get('treasure_count',10), approach='A')
                elif approach == 'B':
                    sim.env = abordagem_b.EnvironmentB(bomb_ratio=params.get('bomb_ratio',0.3))
                elif approach == 'C':
                    sim.env = abordagem_c.EnvironmentC(bomb_ratio=params.get('bomb_ratio',0.3), treasure_count=params.get('treasure_count',10))
            except Exception:
                sim.env = None

            # Attach metrics and minimal fields
            sim.metrics = metrics
            # For shared_memory dependent fields, try to populate reasonable defaults
            sim.shared_memory = type('M', (), {
                'treasures_collected': set(),
                'bombs_found': set()
            })()
            # If raw metrics have treasures_found, populate set size
            if 'treasures_found' in metrics:
                # create dummy set of that size
                sim.shared_memory.treasures_collected = set(range(int(metrics.get('treasures_found',0))))
            if 'bombs_identified' in metrics:
                sim.shared_memory.bombs_found = set(range(int(metrics.get('bombs_identified',0))))

            # Call appropriate calculator
            try:
                if approach == 'A':
                    processed = calc.calculate_approach_a_metrics(sim)
                elif approach == 'B':
                    processed = calc.calculate_approach_b_metrics(sim)
                else:
                    processed = calc.calculate_approach_c_metrics(sim)
                entry['metrics'] = processed
                data[approach][i] = entry
                changed = True
            except Exception as e:
                print('Failed to process entry', i, approach, e)
                continue

    if changed:
        # save back
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print('Post-processing complete, file updated.')
    else:
        print('No changes made.')

if __name__ == '__main__':
    postprocess()
