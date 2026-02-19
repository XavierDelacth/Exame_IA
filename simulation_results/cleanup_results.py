"""
CLI utilitário para remover entradas de simulation_results/all_results.json

Exemplos:
  # Remover todas entradas anteriores a 2026-02-11T10:00:00
  python simulation_results/cleanup_results.py --before 2026-02-11T10:00:00

  # Remover todas entradas da abordagem A / grupo homogeneous
  python simulation_results/cleanup_results.py --approach A --group homogeneous

  # Manter apenas as 10 mais recentes de B (por abordagem+grupo)
  python simulation_results/cleanup_results.py --approach B --keep-last 10

O script pede confirmação antes de escrever o ficheiro.
"""
import argparse
from analise.comparative_analysis import DataStorage


def parse_args():
    p = argparse.ArgumentParser(description='Limpar simulation_results/all_results.json')
    p.add_argument('--approach', choices=['A','B','C'], help='Abordagem a filtrar')
    p.add_argument('--group', choices=['homogeneous','heterogeneous','baseline'], help='Tipo de grupo a filtrar')
    p.add_argument('--before', help='Remover entradas anteriores a timestamp ISO (ex: 2026-02-11T10:00:00)')
    p.add_argument('--keep-last', type=int, help='Manter apenas as N entradas mais recentes (por abordagem+grupo)')
    p.add_argument('--yes', action='store_true', help='Confirmar sem prompt')
    return p.parse_args()


def main():
    args = parse_args()
    storage = DataStorage()

    print('Resumo do estado atual:')
    for app in ['A','B','C']:
        res = storage.get_results_by_approach(app)
        print(f'  Abordagem {app}: {len(res)} entradas')

    print('\nFiltros selecionados:')
    print(f'  approach: {args.approach or "ALL"}')
    print(f'  group: {args.group or "ALL"}')
    print(f'  before: {args.before or "NONE"}')
    print(f'  keep_last: {args.keep_last if args.keep_last is not None else "NONE"}')

    if not args.yes:
        confirm = input('\nConfirmar remoção com estes filtros? (y/N): ').strip().lower()
        if confirm != 'y':
            print('Operação cancelada.')
            return

    removed = storage.remove_results(approach=args.approach, group_type=args.group, before_timestamp=args.before, keep_last=args.keep_last)
    print(f'Removidas {removed} entradas.')

if __name__ == '__main__':
    main()
