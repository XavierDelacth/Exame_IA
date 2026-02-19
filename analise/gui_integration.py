# gui_integration.py
# Integração do Sistema de Análise Comparativa com GUI Existente
# Adiciona funcionalidade de armazenamento e comparação avançada

import sys
from pathlib import Path

# Importar módulo de análise comparativa (dentro do pacote analise)
from .comparative_analysis import (
    DataStorage, MetricsCalculator, ComparativeAnalyzer,
    ComparisonVisualizer, create_comparison_window
)

# ============================================
# EXTENSÃO DA CLASSE IAProjectGUI
# ============================================

class ComparativeExtension:
    """
    Extensão para adicionar funcionalidade comparativa à GUI existente
    
    USO:
    1. Importar no gui.py
    2. Inicializar no __init__ da GUI
    3. Conectar aos eventos de simulação
    """
    
    def __init__(self, gui_instance):
        self.gui = gui_instance
        
        # Inicializar storage
        self.storage = DataStorage()
        self.calculator = MetricsCalculator()
        
        # Conectar eventos
        self._connect_events()
        
        print(f"✅ Sistema de análise comparativa inicializado")
        print(f"📁 Dados salvos em: {self.storage.storage_dir}")
    
    def _connect_events(self):
        """Conecta eventos da GUI para salvar dados automaticamente"""
        # Substituir método finalize_simulation da GUI
        original_finalize = self.gui.finalize_simulation
        
        def new_finalize(start_time):
            # Executar finalização original
            original_finalize(start_time)
            
            # Salvar dados automaticamente
            self.save_current_simulation()
        
        self.gui.finalize_simulation = new_finalize
    
    def save_current_simulation(self):
        """Salva simulação atual no storage"""
        if not self.gui.current_simulation:
            return
        
        sim = self.gui.current_simulation
        approach = self.gui.approach_var.get()
        group_type = self.gui.group_type.get()
        
        # Calcular métricas específicas
        if approach == 'A':
            metrics = self.calculator.calculate_approach_a_metrics(sim)
        elif approach == 'B':
            metrics = self.calculator.calculate_approach_b_metrics(sim)
        elif approach == 'C':
            metrics = self.calculator.calculate_approach_c_metrics(sim)
        else:
            return
        
        # Parâmetros da simulação
        parameters = {
            'num_agents': int(self.gui.agent_spinbox.get()),
            'bomb_ratio': float(self.gui.bomb_scale.get()) / 100.0,
            'treasure_count': int(self.gui.treasure_spinbox.get()),
            'max_steps': int(self.gui.max_steps_spinbox.get()),
            'homogeneous': (group_type == 'homogeneous')
        }
        
        # Salvar
        result = self.storage.save_result(approach, group_type, metrics, parameters)
        
        # Log
        self.gui.log(f"💾 Simulação salva: {approach}-{group_type}", "INFO")
        
        return result
    
    def open_comparison_window(self):
        """Abre janela de comparação avançada"""
        try:
            create_comparison_window(self.gui.root, self.storage)
            self.gui.log("📊 Janela de comparação aberta", "INFO")
        except Exception as e:
            self.gui.log(f"Erro ao abrir comparação: {str(e)}", "ERROR")
    
    def add_comparison_button(self):
        """Adiciona botão de comparação avançada à GUI"""
        import tkinter as tk
        from tkinter import ttk
        
        # Localizar frame de controle (ajustar conforme necessário)
        # Assumindo que existe um button_frame na GUI
        if hasattr(self.gui, 'compare_button'):
            # Adicionar ao lado do botão de comparação existente
            ttk.Button(
                self.gui.compare_button.master,
                text="🔬 SIMULAÇÃO AVANÇADA",
                command=self.open_simulation_runner,
                style="Start.TButton"
            ).pack(fill='x', pady=5)

    def open_simulation_runner(self):
        """Abre e inicia execução das simulações em lote (background)."""
        try:
            # Rodar em thread para não bloquear a GUI
            import threading

            t = threading.Thread(target=self.run_batch_simulations)
            t.daemon = True
            t.start()
            self.gui.log("🔬 Execução em lote iniciada (30 sims/grupo por abordagem)", "INFO")
        except Exception as e:
            self.gui.log(f"Erro ao iniciar execução em lote: {e}", "ERROR")

    def run_batch_simulations(self, runs_per_group=30, num_agents=4, bomb_ratio=0.3, treasure_count=12, max_steps=300):
        """Executa em lote as simulações para cada abordagem e grupo e salva no storage.

        Organização de salvamento: storage.save_result(approach, group_type, metrics, parameters)
        """
        try:
            approaches = ['A', 'B', 'C']
            groups = ['homogeneous', 'heterogeneous', 'baseline']

            for approach in approaches:
                for group in groups:
                    for i in range(runs_per_group):
                        # Instanciar simulação correta por abordagem/grupo
                        if approach == 'A':
                            if group == 'baseline':
                                from abordagem.abordagem_a import BaselineSimulation as SimClass
                                sim = SimClass(num_agents=num_agents, bomb_ratio=bomb_ratio, treasure_count=treasure_count, max_steps=max_steps)
                                metrics = sim.run_simulation(verbose=False)
                            else:
                                from abordagem.abordagem_a import ApproachASimulation as SimClass
                                sim = SimClass(num_agents=num_agents, bomb_ratio=bomb_ratio, treasure_count=treasure_count, homogeneous=(group=='homogeneous'), max_steps=max_steps)
                                metrics = sim.run_simulation(verbose=False)
                        elif approach == 'B':
                            if group == 'baseline':
                                from abordagem.abordagem_a import BaselineB_BFS
                                sim = BaselineB_BFS(bomb_ratio=bomb_ratio, max_steps=max_steps)
                                metrics = sim.run()
                                # Wrap to mimic expected sim object
                                class _SimWrap: pass
                                sim_wrapper = _SimWrap()
                                sim_wrapper.env = sim.env
                                sim_wrapper.metrics = metrics
                                sim = sim_wrapper
                            else:
                                from abordagem.abordagem_b import ApproachBSimulation as SimClass
                                sim = SimClass(num_agents=num_agents, bomb_ratio=bomb_ratio, homogeneous=(group=='homogeneous'), max_steps=max_steps)
                                metrics = sim.run_simulation() if hasattr(sim, 'run_simulation') else sim.run()
                        elif approach == 'C':
                            if group == 'baseline':
                                from abordagem.abordagem_a import BaselineC_AStar as SimClass
                                sim = SimClass(bomb_ratio=bomb_ratio, treasure_count=treasure_count, max_steps=max_steps)
                                metrics = sim.run()
                                class _SimWrapC: pass
                                sim_wrapper = _SimWrapC()
                                sim_wrapper.env = sim.env
                                sim_wrapper.metrics = metrics
                                sim = sim_wrapper
                            else:
                                from abordagem.abordagem_c import ApproachCSimulation as SimClass
                                sim = SimClass(num_agents=num_agents, bomb_ratio=bomb_ratio, treasure_count=treasure_count, homogeneous=(group=='homogeneous'), max_steps=max_steps)
                                metrics = sim.run_simulation(verbose=False)
                        else:
                            continue

                        # Em alguns casos metrics são retornos diretos (dict)
                        sim_metrics = getattr(sim, 'metrics', metrics if isinstance(metrics, dict) else {})

                        parameters = {
                            'num_agents': num_agents,
                            'bomb_ratio': bomb_ratio,
                            'treasure_count': treasure_count,
                            'max_steps': max_steps,
                            'homogeneous': (group == 'homogeneous')
                        }

                        # Salvar usando DataStorage
                        try:
                            self.storage.save_result(approach, group, sim_metrics, parameters)
                            self.gui.log(f"Salvo: Abordagem {approach} | Grupo {group} | Run {i+1}", "INFO")
                        except Exception as e:
                            self.gui.log(f"Erro ao salvar resultado: {e}", "ERROR")

            self.gui.log("🔬 Execução em lote concluída e salva em simulation_results/all_results.json", "SUCCESS")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.gui.log(f"Erro durante execução em lote: {e}", "ERROR")
    
    def get_statistics_summary(self):
        """Retorna resumo estatístico para exibição"""
        summary = {}
        
        for approach in ['A', 'B', 'C']:
            results = self.storage.get_results_by_approach(approach)
            
            summary[approach] = {
                'total': len(results),
                'homogeneous': len([r for r in results if r['group_type'] == 'homogeneous']),
                'heterogeneous': len([r for r in results if r['group_type'] == 'heterogeneous']),
                'baseline': len([r for r in results if r['group_type'] == 'baseline'])
            }
        
        return summary

# ============================================
# FUNÇÃO DE INTEGRAÇÃO RÁPIDA
# ============================================

def integrate_comparison_system(gui_instance):
    """
    Integra sistema de comparação na GUI existente
    
    Adiciona:
    - Salvamento automático de simulações
    - Botão de comparação avançada
    - Armazenamento persistente
    
    Uso no gui.py:
        from analise.gui_integration import integrate_comparison_system
        
        class IAProjectGUI:
            def __init__(self, root):
                # ... código existente ...
                
                # Integrar sistema comparativo
                self.comparison_ext = integrate_comparison_system(self)
    """
    extension = ComparativeExtension(gui_instance)
    extension.add_comparison_button()
    
    return extension

# ============================================
# MODIFICAÇÕES SUGERIDAS PARA GUI.PY
# ============================================

INTEGRATION_GUIDE = """
=================================================================
GUIA DE INTEGRAÇÃO: Sistema de Análise Comparativa
=================================================================

1. IMPORTAÇÃO
   No início do gui.py, adicionar:
   
   from analise.gui_integration import integrate_comparison_system

2. INICIALIZAÇÃO
   No método __init__ da classe IAProjectGUI, após setup_ui():
   
   # Integrar sistema de comparação avançada
   self.comparison_ext = integrate_comparison_system(self)

3. SALVAMENTO AUTOMÁTICO
   O sistema já intercepta finalize_simulation() automaticamente.
   Cada simulação concluída será salva com métricas específicas.

4. BOTÃO DE COMPARAÇÃO
   Um novo botão "COMPARAÇÃO AVANÇADA" será adicionado automaticamente.

5. DADOS PERSISTENTES
   Todos os dados ficam salvos em: simulation_results/all_results.json
   Podem ser exportados para CSV a qualquer momento.

=================================================================
ESTRUTURA DE DADOS SALVA
=================================================================

{
  "A": [
    {
      "timestamp": "2025-01-25T10:30:00",
      "approach": "A",
      "group_type": "heterogeneous",
      "parameters": {
        "num_agents": 4,
        "bomb_ratio": 0.3,
        "treasure_count": 10,
        "max_steps": 100
      },
      "metrics": {
        "treasure_percentage": 75.5,
        "treasures_per_second": 0.15,
        "risk_ratio": 0.25,
        "reward_risk_ratio": 3.2,
        "exploration_efficiency": 0.08,
        ...
      }
    },
    ...
  ],
  "B": [...],
  "C": [...]
}

=================================================================
MÉTRICAS CALCULADAS POR ABORDAGEM
=================================================================

ABORDAGEM A (Maximizar Tesouros):
  ✓ treasure_percentage - % de tesouros encontrados
  ✓ treasures_per_second - Tesouros por tempo
  ✓ treasures_per_step - Tesouros por passo
  ✓ risk_ratio - Risco assumido (bombas/agentes)
  ✓ reward_risk_ratio - Razão recompensa/risco
  ✓ exploration_efficiency - Eficiência de exploração
  ✓ avg_steps_to_treasure - Média de passos até tesouro

ABORDAGEM B (Exploração Total):
  ✓ explored_percentage - % do ambiente explorado
  ✓ safe_exploration_rate - Taxa de exploração segura
  ✓ survival_rate - % de agentes sobreviventes
  ✓ bombs_identified - Bombas identificadas
  ✓ safe_decisions - Número de decisões seguras
  ✓ cells_per_second - Células exploradas por tempo
  ✓ safety_coverage_score - Score geral de segurança
  ✓ redundancy_rate - Taxa de redundância

ABORDAGEM C (Otimizar Caminho):
  ✓ min_steps_to_flag - Passos até bandeira (mínimo)
  ✓ avg_steps_to_flag - Passos até bandeira (média)
  ✓ min_path_cost - Custo mínimo do caminho
  ✓ avg_path_cost - Custo médio do caminho
  ✓ optimal_distance - Distância ótima (Manhattan)
  ✓ path_deviation - Desvio do caminho ótimo
  ✓ path_efficiency - Eficiência do caminho (%)
  ✓ overall_score - Score geral (menor = melhor)

=================================================================
VISUALIZAÇÕES DISPONÍVEIS
=================================================================

1. GRÁFICOS COMPARATIVOS
   - Layout 3 colunas (A, B, C)
   - 5 linhas de métricas
   - Comparação visual entre grupos ML

2. TABELA RESUMO
   - Organizada por abordagem
   - Métricas principais
   - Estatísticas descritivas

3. ANÁLISE DETALHADA
   - Texto formatado
   - Comparações entre grupos
   - Recomendações

=================================================================
EXPORTAÇÃO DE DADOS
=================================================================

CSV por abordagem:
  approach_A_20250125_103000.csv
  approach_B_20250125_103000.csv
  approach_C_20250125_103000.csv

CSV completo:
  all_approaches_20250125_103000.csv

JSON persistente:
  simulation_results/all_results.json

=================================================================
"""

def print_integration_guide():
    """Imprime guia de integração"""
    print(INTEGRATION_GUIDE)

# ============================================
# TESTE RÁPIDO
# ============================================

if __name__ == "__main__":
    import numpy as np
    print_integration_guide()
    
    # Demonstração básica
    storage = DataStorage()
    
    # Exemplo: salvar alguns resultados
    for approach in ['A', 'B', 'C']:
        for group_type in ['homogeneous', 'heterogeneous', 'baseline']:
            metrics = {
                'success_rate': np.random.uniform(0.5, 0.9),
                'execution_time': np.random.uniform(1, 10),
                'agents_alive': np.random.randint(1, 5)
            }
            
            if approach == 'A':
                metrics.update({
                    'treasure_percentage': np.random.uniform(50, 90),
                    'treasures_per_second': np.random.uniform(0.1, 0.3),
                    'risk_ratio': np.random.uniform(0.1, 0.5)
                })
            elif approach == 'B':
                metrics.update({
                    'explored_percentage': np.random.uniform(70, 100),
                    'survival_rate': np.random.uniform(60, 100),
                    'safety_coverage_score': np.random.uniform(40, 80)
                })
            elif approach == 'C':
                metrics.update({
                    'min_steps_to_flag': np.random.randint(20, 60),
                    'path_efficiency': np.random.uniform(60, 95),
                    'flag_found': True
                })
            
            parameters = {
                'num_agents': 4,
                'bomb_ratio': 0.3,
                'treasure_count': 10,
                'max_steps': 100
            }
            
            storage.save_result(approach, group_type, metrics, parameters)
    
    print(f"\n✅ Dados de teste salvos em: {storage.storage_dir}")
    print(f"📊 Total de resultados: {sum(len(v) for v in storage.all_results.values())}")
    
    # Criar análise
    analyzer = ComparativeAnalyzer(storage)
    
    print("\n📈 ANÁLISE GERADA:")
    for approach in ['A', 'B', 'C']:
        analysis = analyzer.analyze_approach(approach)
        if analysis:
            print(f"\nAbordagem {approach}: {analysis['total_simulations']} simulações")
            if 'best_group' in analysis['comparison']:
                print(f"  Melhor grupo: {analysis['comparison']['best_group']}")
