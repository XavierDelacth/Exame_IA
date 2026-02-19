"""
Utility to initialize `simulation_results/all_results.json` with default structure
if the file is missing or empty.
"""
from pathlib import Path
import json

def init(path='simulation_results/all_results.json'):
    p = Path(path)
    p.parent.mkdir(exist_ok=True)
    data = {'A': [], 'B': [], 'C': []}
    try:
        if not p.exists() or p.stat().st_size == 0:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f'Initialized {p}')
        else:
            print(f'{p} already initialized')
    except Exception as e:
        print(f'Error initializing {p}: {e}')

if __name__ == '__main__':
    init()
