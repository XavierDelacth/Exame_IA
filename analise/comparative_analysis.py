# comparative_analysis.py 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch
import json
from datetime import datetime
from pathlib import Path
import seaborn as sns
from tkinter import scrolledtext
import tkinter as tk

class DataStorage:
    def __init__(self, storage_dir="simulation_results"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.results_file = self.storage_dir / "all_results.json"
        self.all_results = self.load_results()
    
    def load_results(self):
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {'A': [], 'B': [], 'C': []}
        return {'A': [], 'B': [], 'C': []}
    
    def save_result(self, approach, group_type, metrics, parameters):
        result = {
            'timestamp': datetime.now().isoformat(),
            'approach': approach,
            'group_type': group_type,
            'parameters': parameters,
            'metrics': metrics
        }
        self.all_results[approach].append(result)
        self._save_to_file()
        return result
    
    def _save_to_file(self):
        try:
            # Write atomically: write to temp file then replace
            tmp_file = self.results_file.with_suffix('.tmp')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(self.all_results, f, indent=2, ensure_ascii=False)
            try:
                tmp_file.replace(self.results_file)
            except Exception:
                # Fallback to non-atomic write
                with open(self.results_file, 'w', encoding='utf-8') as f:
                    json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar: {e}")
    
    def get_results_by_approach(self, approach):
        return self.all_results.get(approach, [])
    
    def export_to_csv(self, approach=None):
        """Exporta resultados para CSV na pasta data."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            if approach:
                results = self.all_results.get(approach, [])
            else:
                results = []
                for app in ['A', 'B', 'C']:
                    results.extend(self.all_results.get(app, []))
            
            if results:
                flat_results = []
                for r in results:
                    flat = {'timestamp': r['timestamp'], 'approach': r['approach'], 'group_type': r['group_type']}
                    for k, v in r.get('parameters', {}).items():
                        flat[f'param_{k}'] = v
                    for k, v in r.get('metrics', {}).items():
                        if isinstance(v, (int, float, bool, str)):
                            flat[f'metric_{k}'] = v
                    flat_results.append(flat)
                
                df = pd.DataFrame(flat_results)
                filename = data_dir / f"{'approach_'+approach if approach else 'all'}_{timestamp}.csv"
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                return filename
        except Exception as e:
            print(f"Erro ao exportar: {e}")
        return None

    def remove_results(self, approach=None, group_type=None, before_timestamp=None, keep_last=None):
        """Remove resultados do armazenamento com filtros.

        - approach: 'A','B','C' ou None para todos
        - group_type: 'homogeneous','heterogeneous','baseline' ou None
        - before_timestamp: ISO string; remove entradas com timestamp < this
        - keep_last: int, keep last N matching entries (per approach+group)
        Returns: number of removed entries
        """
        removed = 0
        approaches = [approach] if approach else list(self.all_results.keys())

        for app in approaches:
            entries = self.all_results.get(app, [])
            # Filter by group_type
            if group_type:
                filtered = [e for e in entries if e.get('group_type') == group_type]
            else:
                filtered = list(entries)

            # Filter by timestamp
            if before_timestamp:
                try:
                    from datetime import datetime
                    cutoff = datetime.fromisoformat(before_timestamp)
                    to_remove = [e for e in filtered if datetime.fromisoformat(e.get('timestamp')) < cutoff]
                except Exception:
                    to_remove = []
            else:
                to_remove = list(filtered)

            # If keep_last specified, compute which to actually remove
            if keep_last is not None and keep_last >= 0:
                # Sort matching entries by timestamp desc and keep last N
                try:
                    sorted_matches = sorted(filtered, key=lambda x: x.get('timestamp', ''), reverse=True)
                    to_keep = set(id(x) for x in sorted_matches[:keep_last])
                    # to_remove are those in filtered but not in to_keep
                    to_remove = [e for e in filtered if id(e) not in to_keep]
                except Exception:
                    pass

            if not to_remove:
                continue

            # Remove entries from entries list
            new_entries = [e for e in entries if e not in to_remove]
            removed += (len(entries) - len(new_entries))
            self.all_results[app] = new_entries

        if removed > 0:
            self._save_to_file()

        return removed

class MetricsCalculator:
    @staticmethod
    def safe_div(a, b, default=0.0):
        try:
            return float(a / b) if b != 0 else default
        except:
            return default
    
    @staticmethod
    def calculate_approach_a_metrics(simulation):
        metrics = {}
        try:
            total_treasures = getattr(simulation.env, 'treasure_count', 0)
            treasures_found = len(getattr(simulation.shared_memory, 'treasures_collected', set()))
            execution_time = max(simulation.metrics.get('execution_time', 0.001), 0.001)
            steps_taken = max(simulation.metrics.get('steps_taken', 1), 1)
            agents_alive = simulation.metrics.get('agents_alive', 0)
            total_agents = max(getattr(simulation, 'num_agents', 1), 1)
            
            metrics['treasure_percentage'] = MetricsCalculator.safe_div(treasures_found * 100, total_treasures)
            metrics['treasures_per_second'] = MetricsCalculator.safe_div(treasures_found, execution_time)
            metrics['treasures_per_step'] = MetricsCalculator.safe_div(treasures_found, steps_taken)
            
            bombs_triggered = total_agents - agents_alive
            metrics['bombs_triggered'] = bombs_triggered
            metrics['risk_ratio'] = MetricsCalculator.safe_div(bombs_triggered, total_agents)
            metrics['exploration_efficiency'] = MetricsCalculator.safe_div(treasures_found, steps_taken)
            metrics['success'] = simulation.metrics.get('success', False)
            metrics['success_rate'] = 1.0 if metrics['success'] else 0.0
            metrics['reward_risk_ratio'] = MetricsCalculator.safe_div(treasures_found, bombs_triggered + 1)
            metrics['explored_percentage'] = simulation.metrics.get('explored_percentage', 0.0)
            metrics['avg_steps_to_treasure'] = MetricsCalculator.safe_div(steps_taken, max(treasures_found, 1))
            metrics['execution_time'] = execution_time
        except Exception as e:
            print(f"Erro métricas A: {e}")
            metrics = {'treasure_percentage': 0.0, 'success_rate': 0.0, 'execution_time': 0.0}
        return metrics
    
    @staticmethod
    def calculate_approach_b_metrics(simulation):
        metrics = {}
        try:
            execution_time = max(simulation.metrics.get('execution_time', 0.001), 0.001)
            steps_taken = max(simulation.metrics.get('steps_taken', 1), 1)
            agents_alive = simulation.metrics.get('agents_alive', 0)
            total_agents = max(getattr(simulation, 'num_agents', 1), 1)
            explored_pct = simulation.metrics.get('explored_percentage', 0.0)
            
            total_cells = getattr(simulation.env, 'size', 10) ** 2
            bomb_count = sum(1 for row in simulation.env.grid for cell in row if cell == 'B')
            free_cells = max(total_cells - bomb_count, 1)
            explored_free_cells = simulation.metrics.get('explored_free_cells', 0)
            
            survival_rate = MetricsCalculator.safe_div(agents_alive * 100, total_agents)
            
            metrics['explored_percentage'] = explored_pct
            metrics['safe_exploration_rate'] = MetricsCalculator.safe_div(explored_free_cells * 100, free_cells)
            metrics['agents_alive'] = agents_alive
            metrics['survival_rate'] = survival_rate
            metrics['bombs_triggered'] = total_agents - agents_alive
            metrics['cells_per_second'] = MetricsCalculator.safe_div(explored_free_cells, execution_time)
            metrics['cells_per_step'] = MetricsCalculator.safe_div(explored_free_cells, steps_taken)
            metrics['bombs_identified'] = len(getattr(simulation.shared_memory, 'bombs_found', set()))
            metrics['safe_decisions'] = explored_free_cells
            metrics['success'] = simulation.metrics.get('success', False)
            metrics['success_rate'] = 1.0 if metrics['success'] else 0.0
            metrics['safety_coverage_score'] = MetricsCalculator.safe_div(explored_pct * survival_rate, 100)
            metrics['redundancy_rate'] = 0.0
            metrics['execution_time'] = execution_time
        except Exception as e:
            print(f"Erro métricas B: {e}")
            import traceback
            traceback.print_exc()
            metrics = {'explored_percentage': 0.0, 'success_rate': 0.0, 'execution_time': 0.0}
        return metrics
    
    @staticmethod
    def calculate_approach_c_metrics(simulation):
        metrics = {}
        try:
            execution_time = max(simulation.metrics.get('execution_time', 0.001), 0.001)
            steps_taken = max(simulation.metrics.get('steps_taken', 1), 1)
            agents_alive = simulation.metrics.get('agents_alive', 0)
            total_agents = max(getattr(simulation, 'num_agents', 1), 1)
            flag_found = simulation.metrics.get('flag_found', False)
            
            steps_list = []
            if hasattr(simulation, 'agents'):
                for agent in simulation.agents:
                    if hasattr(agent, 'steps_taken') and agent.steps_taken > 0:
                        steps_list.append(agent.steps_taken)
            
            if not steps_list:
                steps_list = [steps_taken]
            
            metrics['avg_steps_to_flag'] = float(np.mean(steps_list))
            metrics['min_steps_to_flag'] = float(min(steps_list))
            metrics['max_steps_to_flag'] = float(max(steps_list))
            
            min_path_cost = simulation.metrics.get('min_path_cost', 0)
            avg_path_cost = simulation.metrics.get('avg_path_cost', 0)
            metrics['min_path_cost'] = 0.0 if min_path_cost == float('inf') else float(min_path_cost)
            metrics['avg_path_cost'] = 0.0 if avg_path_cost == float('inf') else float(avg_path_cost)
            
            if hasattr(simulation.env, 'flag_position') and simulation.env.flag_position:
                fx, fy = simulation.env.flag_position
                optimal_distance = abs(fx) + abs(fy)
                metrics['optimal_distance'] = float(optimal_distance)
                metrics['actual_distance'] = metrics['min_steps_to_flag']
                metrics['path_deviation'] = metrics['actual_distance'] - optimal_distance
                metrics['path_efficiency'] = MetricsCalculator.safe_div(optimal_distance * 100, metrics['actual_distance'])
            else:
                metrics['optimal_distance'] = 0.0
                metrics['actual_distance'] = 0.0
                metrics['path_deviation'] = 0.0
                metrics['path_efficiency'] = 0.0
            
            bombs_triggered = total_agents - agents_alive
            metrics['bombs_triggered'] = bombs_triggered
            metrics['risk_ratio'] = MetricsCalculator.safe_div(bombs_triggered, total_agents)
            metrics['treasures_collected'] = simulation.metrics.get('treasures_found', 0)
            metrics['time_to_flag'] = execution_time
            metrics['steps_per_second'] = MetricsCalculator.safe_div(steps_taken, execution_time)
            metrics['flag_found'] = flag_found
            metrics['success'] = simulation.metrics.get('success', False)
            metrics['success_rate'] = 1.0 if flag_found else 0.0
            metrics['execution_time'] = execution_time
            
            if flag_found:
                metrics['overall_score'] = (
                    metrics['min_steps_to_flag'] * 0.4 +
                    metrics['min_path_cost'] * 0.3 +
                    bombs_triggered * 10 * 0.3
                )
            else:
                metrics['overall_score'] = 999999.0
        except Exception as e:
            print(f"Erro métricas C: {e}")
            import traceback
            traceback.print_exc()
            metrics = {'min_steps_to_flag': 0.0, 'success_rate': 0.0, 'execution_time': 0.0}
        return metrics

class ComparativeAnalyzer:
    def __init__(self, storage):
        self.storage = storage
        self.calculator = MetricsCalculator()
    
    def analyze_approach(self, approach):
        results = self.storage.get_results_by_approach(approach)
        if not results:
            return None
        
        homogeneous = [r for r in results if r['group_type'] == 'homogeneous']
        heterogeneous = [r for r in results if r['group_type'] == 'heterogeneous']
        baseline = [r for r in results if r['group_type'] == 'baseline']
        
        return {
            'approach': approach,
            'total_simulations': len(results),
            'groups': {
                'homogeneous': self._analyze_group(homogeneous),
                'heterogeneous': self._analyze_group(heterogeneous),
                'baseline': self._analyze_group(baseline)
            },
            'comparison': self._compare_groups(homogeneous, heterogeneous, baseline, approach)
        }
    
    def _analyze_group(self, results):
        if not results:
            return {'count': 0, 'metrics': {}}
        
        all_metrics = {}
        for result in results:
            for key, value in result.get('metrics', {}).items():
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        stats = {}
        for key, values in all_metrics.items():
            if values:
                stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return {'count': len(results), 'metrics': stats}
    
    def _compare_groups(self, homo, hetero, baseline, approach):
        comparison = {}
        main_metric_map = {
            'A': 'treasure_percentage',
            'B': 'explored_percentage',
            'C': 'success_rate'
        }
        main_metric = main_metric_map.get(approach, 'success_rate')
        
        for group_name, group_results in {'homogeneous': homo, 'heterogeneous': hetero, 'baseline': baseline}.items():
            if group_results:
                values = [r['metrics'].get(main_metric, 0) for r in group_results]
                if values:
                    comparison[group_name] = {'mean': float(np.mean(values)), 'count': len(values)}
        
        if comparison:
            best = max(comparison.items(), key=lambda x: x[1]['mean'])
            comparison['best_group'] = best[0]
            comparison['best_value'] = best[1]['mean']
        
        return comparison

class ComparisonVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        sns.set_style("whitegrid")
    
    def create_comprehensive_comparison(self, parent_frame):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)
        
        colors = {'homogeneous': '#e67e22', 'heterogeneous': '#9b59b6', 'baseline': '#e74c3c'}
        
        approaches_data = {app: self.analyzer.analyze_approach(app) for app in ['A', 'B', 'C']}
        
        titles = {'A': 'ABORDAGEM A\nMaximizar Tesouros', 'B': 'ABORDAGEM B\nExploracao Total', 'C': 'ABORDAGEM C\nOtimizar Caminho'}
        
        for col, approach in enumerate(['A', 'B', 'C']):
            ax = fig.add_subplot(gs[0, col])
            analysis = approaches_data.get(approach)
            best_text = ''
            if analysis and analysis.get('comparison'):
                comp = analysis['comparison']
                best = comp.get('best_group')
                val = comp.get('best_value')
                if best:
                    if approach == 'A':
                        best_text = f"\nMelhor: {best.title()} ({val:.1f}%)"
                    elif approach == 'B':
                        best_text = f"\nMelhor: {best.title()} ({val:.1f}%)"
                    else:
                        best_text = f"\nMelhor: {best.title()} ({val:.1%})"

            ax.text(0.5, 0.5, titles[approach] + best_text, ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.axis('off')
        
        metrics_lines = {
            1: {'A': ('treasure_percentage', 'Tesouros (%)', 0, 100), 
                'B': ('explored_percentage', 'Explorado (%)', 0, 100), 
                'C': ('success_rate', 'Sucesso', 0, 1)},
            2: {'A': ('treasures_per_second', 'Tesouros/s', 0, None), 
                'B': ('cells_per_second', 'Celulas/s', 0, None), 
                'C': ('min_steps_to_flag', 'Passos', 0, None)},
            3: {'A': ('risk_ratio', 'Risco', 0, 1), 
                'B': ('safe_exploration_rate', 'Exploracao Segura (%)', 0, 100), 
                'C': ('risk_ratio', 'Risco', 0, 1)},
            4: {'A': ('reward_risk_ratio', 'Recomp/Risco', 0, None), 
                'B': ('execution_time', 'Tempo (s)', 0, None), 
                'C': ('path_efficiency', 'Efic. (%)', 0, 100)}
        }
        
        for line, line_metrics in metrics_lines.items():
            for col, approach in enumerate(['A', 'B', 'C']):
                ax = fig.add_subplot(gs[line, col])
                metric_key, title, ymin, ymax = line_metrics[approach]
                self._plot_group_comparison(ax, approaches_data[approach], metric_key, title, colors, (ymin, ymax))
        
        fig.suptitle('COMPARACAO: Grupos ML por Abordagem', fontsize=14, fontweight='bold', y=0.98)
        
        legend_elements = [Patch(facecolor=colors['homogeneous'], label='Homogeneo'),
                          Patch(facecolor=colors['heterogeneous'], label='Heterogeneo'),
                          Patch(facecolor=colors['baseline'], label='Baseline')]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3)
        
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        return canvas
    
    def _plot_group_comparison(self, ax, analysis, metric_key, title, colors, ylim):
        if not analysis or analysis['total_simulations'] == 0:
            ax.text(0.5, 0.5, 'Sem dados\nExecute simulacoes', ha='center', va='center', fontsize=9)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
            return
        groups, means, stds, group_colors = [], [], [], []
        for group_name in ['homogeneous', 'heterogeneous', 'baseline']:
            group_data = analysis['groups'][group_name]
            if group_data['count'] > 0:
                metrics = group_data['metrics'].get(metric_key, {})
                if metrics:
                    groups.append(group_name.title()[:5])
                    means.append(metrics.get('mean', 0))
                    stds.append(metrics.get('std', 0))
                    group_colors.append(colors[group_name])
        if not groups:
            ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center')
            ax.set_title(title, fontsize=9)
            ax.axis('off')
            return
        x = np.arange(len(groups))
        bars = ax.bar(x, means, color=group_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        if max(stds) > 0:
            ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', capsize=4, alpha=0.6)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{mean:.1f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        if ylim[0] is not None or ylim[1] is not None:
            ax.set_ylim(ylim)
    
    def create_summary_table(self, parent_frame):
        from tkinter import ttk
        total_data = sum(len(self.analyzer.storage.get_results_by_approach(app)) for app in ['A', 'B', 'C'])
        if total_data == 0:
            tk.Label(parent_frame, text="Nenhum dado - Execute simulacoes!", font=('Arial', 12), fg='orange').pack(pady=50)
            return
        notebook = ttk.Notebook(parent_frame)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        self._create_approach_a_table(notebook)
        self._create_approach_b_table(notebook)
        self._create_approach_c_table(notebook)
    
    def _create_approach_a_table(self, notebook):
        from tkinter import ttk
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Abordagem A - Tesouros")
        results = self.analyzer.storage.get_results_by_approach('A')
        if not results:
            tk.Label(frame, text="Sem dados para Abordagem A", font=('Arial', 11), fg='orange').pack(pady=50)
            return
        table_data = []
        for r in results:
            metrics = r.get('metrics', {})
            params = r.get('parameters', {})
            row = {'Grupo': r.get('group_type', '').title(), 'Agentes': params.get('num_agents', '-'),
                'Bombas (%)': f"{params.get('bomb_ratio', 0) * 100:.0f}", 'Tesouros Meta': params.get('treasure_count', '-'),
                'Tesouros (%)': f"{metrics.get('treasure_percentage', 0):.1f}", 'Tesouros/s': f"{metrics.get('treasures_per_second', 0):.3f}",
                'Risco': f"{metrics.get('risk_ratio', 0):.2f}", 'Recompensa/Risco': f"{metrics.get('reward_risk_ratio', 0):.2f}",
                'Eficiencia': f"{metrics.get('exploration_efficiency', 0):.3f}", 'Tempo (s)': f"{metrics.get('execution_time', 0):.2f}"}
            table_data.append(row)
        df = pd.DataFrame(table_data)
        self._render_table(frame, df, "Abordagem A: Maximizar Tesouros (>50%)")
    
    def _create_approach_b_table(self, notebook):
        from tkinter import ttk
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Abordagem B - Exploracao")
        results = self.analyzer.storage.get_results_by_approach('B')
        if not results:
            tk.Label(frame, text="Sem dados para Abordagem B", font=('Arial', 11), fg='orange').pack(pady=50)
            return
        table_data = []
        for r in results:
            metrics = r.get('metrics', {})
            params = r.get('parameters', {})
            row = {'Grupo': r.get('group_type', '').title(), 'Agentes': params.get('num_agents', '-'),
                'Bombas (%)': f"{params.get('bomb_ratio', 0) * 100:.0f}", 'Explorado (%)': f"{metrics.get('explored_percentage', 0):.1f}",
                'Exploracao Segura (%)': f"{metrics.get('safe_exploration_rate', 0):.1f}", 'Celulas/s': f"{metrics.get('cells_per_second', 0):.2f}",
                'Bombas Identificadas': f"{metrics.get('bombs_identified', 0):.0f}", 'Tempo (s)': f"{metrics.get('execution_time', 0):.2f}"}
            table_data.append(row)
        df = pd.DataFrame(table_data)
        self._render_table(frame, df, "Abordagem B: Exploracao Total")
    
    def _create_approach_c_table(self, notebook):
        from tkinter import ttk
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Abordagem C - Bandeira")
        results = self.analyzer.storage.get_results_by_approach('C')
        if not results:
            tk.Label(frame, text="Sem dados para Abordagem C", font=('Arial', 11), fg='orange').pack(pady=50)
            return
        table_data = []
        for r in results:
            metrics = r.get('metrics', {})
            params = r.get('parameters', {})
            flag_found = metrics.get('flag_found', False)
            row = {'Grupo': r.get('group_type', '').title(), 'Agentes': params.get('num_agents', '-'),
                'Bombas (%)': f"{params.get('bomb_ratio', 0) * 100:.0f}", 'Bandeira': 'Sim' if flag_found else 'Nao',
                'Sucesso': '100.0%' if flag_found else '0.0%', 'Passos (min)': f"{metrics.get('min_steps_to_flag', 0):.1f}",
                'Passos (media)': f"{metrics.get('avg_steps_to_flag', 0):.1f}", 'Eficiencia (%)': f"{metrics.get('path_efficiency', 0):.1f}",
                'Dist. Otima': f"{metrics.get('optimal_distance', 0):.0f}", 'Tempo (s)': f"{metrics.get('execution_time', 0):.2f}"}
            table_data.append(row)
        df = pd.DataFrame(table_data)
        self._render_table(frame, df, "Abordagem C: Encontrar Bandeira")
    
    def _render_table(self, parent, df, title):
        from tkinter import ttk
        title_label = tk.Label(parent, text=title, font=('Arial', 12, 'bold'), bg='#e8f4f8', pady=10)
        title_label.pack(fill='x', padx=10, pady=(10, 0))
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill='both', expand=True, padx=10, pady=10)
        vsb = ttk.Scrollbar(table_frame, orient="vertical")
        hsb = ttk.Scrollbar(table_frame, orient="horizontal")
        tree = ttk.Treeview(table_frame, columns=list(df.columns), show='headings', yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        for col in df.columns:
            tree.heading(col, text=col)
            width = 120 if col in ['Grupo', 'Bandeira', 'Sucesso'] else (80 if col in ['Agentes', 'Bombas (%)', 'Tesouros Meta'] else 110)
            tree.column(col, width=width, anchor='center')
        for idx, row in df.iterrows():
            tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
            tree.insert("", "end", values=list(row), tags=(tag,))
        tree.tag_configure('evenrow', background='#f9f9f9')
        tree.tag_configure('oddrow', background='#ffffff')
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)
        self._add_summary_stats(parent, df)
    
    def _add_summary_stats(self, parent, df):
        stats_frame = tk.Frame(parent, bg='#f0f0f0', relief='ridge', bd=2)
        stats_frame.pack(fill='x', padx=10, pady=10)
        total_rows = len(df)
        stats_text = f"Total de simulacoes: {total_rows}"
        if 'Grupo' in df.columns:
            groups = df['Grupo'].value_counts()
            stats_text += "  |  " + "  ".join(f"{g}: {c}" for g, c in groups.items())
        tk.Label(stats_frame, text=stats_text, bg='#f0f0f0', font=('Arial', 10), pady=5).pack()

def create_comparison_window(root, storage):
    from tkinter import ttk, messagebox
    window = tk.Toplevel(root)
    window.title("Analise Comparativa por Abordagem")
    window.geometry("1400x900")
    analyzer = ComparativeAnalyzer(storage)
    visualizer = ComparisonVisualizer(analyzer)
    notebook = ttk.Notebook(window)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)
    graphs_frame = ttk.Frame(notebook)
    notebook.add(graphs_frame, text="Graficos")
    try:
        visualizer.create_comprehensive_comparison(graphs_frame)
    except Exception as e:
        print(f"Erro graficos: {e}")
        tk.Label(graphs_frame, text=f"Erro: {str(e)}", font=('Arial', 11), fg='red').pack(pady=50)
    table_frame = ttk.Frame(notebook)
    notebook.add(table_frame, text="Tabela")
    try:
        visualizer.create_summary_table(table_frame)
    except Exception as e:
        tk.Label(table_frame, text=f"Erro: {str(e)}", font=('Arial', 11), fg='red').pack(pady=50)
    analysis_frame = ttk.Frame(notebook)
    notebook.add(analysis_frame, text="Analise")
    
    # Criar Text widget com tags para formatação rica
    analysis_text = tk.Text(analysis_frame, wrap=tk.WORD, font=("Segoe UI", 10), height=30, bg='#f8f9fa')
    analysis_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Configurar tags de formatação
    analysis_text.tag_config('titulo', font=('Segoe UI', 16, 'bold'), foreground='#2c3e50', spacing1=10, spacing3=10)
    analysis_text.tag_config('subtitulo', font=('Segoe UI', 13, 'bold'), foreground='#34495e', spacing1=8, spacing3=5)
    analysis_text.tag_config('secao', font=('Segoe UI', 11, 'bold'), foreground='#7f8c8d', spacing1=5, spacing3=3)
    analysis_text.tag_config('destaque', font=('Segoe UI', 10, 'bold'), foreground='#27ae60')
    analysis_text.tag_config('alerta', font=('Segoe UI', 10, 'bold'), foreground='#e74c3c')
    analysis_text.tag_config('info', font=('Segoe UI', 10), foreground='#34495e')
    analysis_text.tag_config('metrica', font=('Consolas', 10), foreground='#2980b9', background='#ecf0f1')
    analysis_text.tag_config('separador', font=('Segoe UI', 10), foreground='#bdc3c7')
    analysis_text.tag_config('badge_homo', font=('Segoe UI', 9, 'bold'), foreground='white', background='#e67e22')
    analysis_text.tag_config('badge_hete', font=('Segoe UI', 9, 'bold'), foreground='white', background='#9b59b6')
    analysis_text.tag_config('badge_base', font=('Segoe UI', 9, 'bold'), foreground='white', background='#e74c3c')
    
    # Gerar conteúdo formatado
    def insert_text(text, tag=None):
        analysis_text.insert(tk.END, text, tag)
    
    # Título principal
    insert_text("📊 ANÁLISE COMPARATIVA POR ABORDAGEM\n", 'titulo')
    insert_text("━" * 80 + "\n\n", 'separador')
    
    # Análise por abordagem
    approach_names = {
        'A': ('🎯 ABORDAGEM A', 'Maximizar Tesouros'),
        'B': ('🔍 ABORDAGEM B', 'Exploração Total do Ambiente'),
        'C': ('🚩 ABORDAGEM C', 'Otimizar Caminho até Bandeira')
    }
    
    for approach in ['A', 'B', 'C']:
        analysis = analyzer.analyze_approach(approach)
        
        # Título da abordagem
        title, subtitle = approach_names[approach]
        insert_text(f"\n{title} - {subtitle}\n", 'subtitulo')
        insert_text("─" * 80 + "\n", 'separador')
        
        if not analysis or analysis['total_simulations'] == 0:
            insert_text("  ⚠️  Sem dados disponíveis para esta abordagem\n\n", 'alerta')
            continue
        
        # Estatísticas gerais
        insert_text(f"\n  📈 Total de Simulações: ", 'secao')
        insert_text(f"{analysis['total_simulations']}\n", 'destaque')
        
        # Distribuição por grupo
        groups = analysis['groups']
        insert_text("\n  📊 Distribuição por Grupo:\n", 'secao')
        
        for group_name, group_data in groups.items():
            count = group_data['count']
            if count == 0:
                continue
            
            # Badge colorido
            badge_tag = f'badge_{group_name[:4]}'  # homo, hete, base
            pct = (count / analysis['total_simulations']) * 100
            insert_text(f"     • {group_name.upper()} ", badge_tag)
            insert_text(f" → {count} simulações ({pct:.1f}%)\n", 'info')
        
        # Comparação e melhor grupo
        if 'best_group' in analysis['comparison']:
            insert_text("\n  🏆 Melhor Desempenho:\n", 'secao')
            best = analysis['comparison']['best_group']
            best_val = analysis['comparison']['best_value']
            
            # Determinar métrica e formatação
            if approach == 'A':
                metric_text = f"{best_val:.1f}% de tesouros coletados"
            elif approach == 'B':
                metric_text = f"{best_val:.1f}% do ambiente explorado"
            elif approach == 'C':
                metric_text = f"{best_val:.1%} taxa de sucesso"
            
            insert_text(f"     ✅ ", 'destaque')
            insert_text(f"{best.upper()}", f'badge_{best[:4]}')
            insert_text(f"  →  {metric_text}\n", 'info')
        
        # Métricas principais por grupo
        insert_text("\n  📋 Métricas Principais:\n", 'secao')
        
        # Definir métricas principais por abordagem
        if approach == 'A':
            main_metrics = {
                'treasure_percentage': ('Tesouros Coletados', '%'),
                'treasures_per_second': ('Tesouros por Segundo', ''),
                'risk_ratio': ('Índice de Risco', ''),
                'reward_risk_ratio': ('Recompensa/Risco', '')
            }
        elif approach == 'B':
            main_metrics = {
                'explored_percentage': ('Exploração', '%'),
                'survival_rate': ('Taxa de Sobrevivência', '%'),
                'safe_exploration_rate': ('Exploração Segura', '%'),
                'cells_per_second': ('Células/Segundo', '')
            }
        elif approach == 'C':
            main_metrics = {
                'success_rate': ('Taxa de Sucesso', '%'),
                'min_steps_to_flag': ('Passos Mínimos', ''),
                'path_efficiency': ('Eficiência do Caminho', '%'),
                'execution_time': ('Tempo de Execução', 's')
            }
        
        for group_name, group_data in groups.items():
            if group_data['count'] == 0:
                continue
            
            metrics = group_data.get('metrics', {})
            if not metrics:
                continue
            
            insert_text(f"\n     {group_name.upper()}:\n", 'info')
            
            for metric_key, (metric_name, unit) in main_metrics.items():
                if metric_key in metrics:
                    stat = metrics[metric_key]
                    mean_val = stat['mean']
                    std_val = stat['std']
                    
                    if unit == '%' and metric_key != 'success_rate':
                        value_text = f"{mean_val:.1f}{unit} ± {std_val:.1f}"
                    elif metric_key == 'success_rate':
                        value_text = f"{mean_val*100:.1f}% ± {std_val*100:.1f}"
                    elif metric_key == 'min_steps_to_flag':
                        value_text = f"{mean_val:.0f} ± {std_val:.0f}"
                    else:
                        value_text = f"{mean_val:.3f} ± {std_val:.3f}{unit}"
                    
                    insert_text(f"       • {metric_name}: ", 'info')
                    insert_text(f"{value_text}\n", 'metrica')
        
        insert_text("\n", 'info')
    
    # Resumo final
    insert_text("\n" + "━" * 80 + "\n", 'separador')
    insert_text("💡 RESUMO GERAL\n", 'subtitulo')
    insert_text("─" * 80 + "\n", 'separador')
    
    total_sims = sum(analyzer.analyze_approach(app)['total_simulations'] 
                     for app in ['A', 'B', 'C'] 
                     if analyzer.analyze_approach(app))
    
    insert_text(f"\n  📊 Total de Simulações Realizadas: ", 'secao')
    insert_text(f"{total_sims}\n", 'destaque')
    
    # Melhores de cada abordagem
    insert_text("\n  🏆 Melhores Configurações:\n", 'secao')
    for approach in ['A', 'B', 'C']:
        analysis = analyzer.analyze_approach(approach)
        if analysis and 'best_group' in analysis['comparison']:
            approach_name = approach_names[approach][0]
            best = analysis['comparison']['best_group']
            insert_text(f"     • {approach_name}: ", 'info')
            insert_text(f"{best.upper()}\n", f'badge_{best[:4]}')
    
    insert_text("\n\n", 'info')
    
    # Desabilitar edição
    analysis_text.config(state=tk.DISABLED)
    button_frame = ttk.Frame(window)
    button_frame.pack(fill='x', padx=10, pady=10)
    def export():
        f = storage.export_to_csv()
        if f:
            messagebox.showinfo("Exportacao", f"Dados exportados:\n{f}")
        else:
            messagebox.showwarning("Aviso", "Nenhum dado")
    ttk.Button(button_frame, text="Exportar CSV", command=export).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Fechar", command=window.destroy).pack(side='right', padx=5)
    return window

if __name__ == "__main__":
    print("Sistema de Analise Comparativa - VERSAO FINAL")