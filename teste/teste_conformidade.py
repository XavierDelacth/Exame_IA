"""
teste_conformidade.py

Script de teste de conformidade para todas as abordagens e tipos de grupos.
Verifica se cada combinação está funcionando corretamente.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from simulation import Simulation
import numpy as np

def test_conformidade():
    """Testa todas as abordagens com todos os tipos de grupos"""
    
    approaches = ['A', 'B', 'C']
    group_types = ['homogeneous', 'heterogeneous']
    num_agents = 3
    bomb_ratio = 0.5
    max_iterations = 10
    
    print("=" * 80)
    print("TESTE DE CONFORMIDADE - TODAS AS ABORDAGENS E TIPOS DE GRUPOS")
    print("=" * 80)
    print()
    
    results = []
    
    for approach in approaches:
        print(f"{'='*80}")
        print(f"ABORDAGEM {approach}")
        print(f"{'='*80}")
        print()
        
        for group_type in group_types:
            print(f"Testando com {group_type} group...")
            
            try:
                # Criar simulação
                sim = Simulation(
                    approach=approach,
                    num_agents=num_agents,
                    bomb_ratio=bomb_ratio,
                    group_type=group_type
                )
                
                print(f"  ✓ Simulação criada com sucesso")
                print(f"    - Ambiente: {sim.environment.size}x{sim.environment.size}")
                print(f"    - Agentes: {len(sim.agents)}")
                print(f"    - Tipo de grupo: {group_type}")
                
                # Verificar ambiente
                if approach == 'A':
                    treasures = np.sum(sim.environment.grid == 'T')
                    bombs = np.sum(sim.environment.grid == 'B')
                    print(f"    - Tesouros: {treasures}")
                    print(f"    - Bombas: {bombs}")
                    
                    if treasures > 0:
                        print(f"  ✓ Abordagem A: Tesouros presentes")
                    else:
                        print(f"  ✗ ERRO: Abordagem A sem tesouros!")
                        
                elif approach == 'B':
                    bombs = np.sum(sim.environment.grid == 'B')
                    print(f"    - Bombas: {bombs}")
                    
                    # Abordagem B não deve ter tesouros
                    treasures = np.sum(sim.environment.grid == 'T')
                    if treasures == 0:
                        print(f"  ✓ Abordagem B: Sem tesouros (conforme esperado)")
                    else:
                        print(f"  ✗ ERRO: Abordagem B tem tesouros!")
                        
                elif approach == 'C':
                    treasures = np.sum(sim.environment.grid == 'T')
                    flags = np.sum(sim.environment.grid == 'F')
                    bombs = np.sum(sim.environment.grid == 'B')
                    print(f"    - Tesouros: {treasures}")
                    print(f"    - Bandeiras: {flags}")
                    print(f"    - Bombas: {bombs}")
                    
                    if flags > 0:
                        flag_pos = np.argwhere(sim.environment.grid == 'F')[0]
                        print(f"  ✓ Abordagem C: Bandeira em posição {tuple(flag_pos)}")
                    else:
                        print(f"  ✗ ERRO: Abordagem C sem bandeira!")
                
                # Executar simulação
                result = sim.run(max_iterations=max_iterations)
                
                print(f"  ✓ Simulação executada com sucesso")
                print(f"    - Iterações: {result['iterations_executed']}")
                print(f"    - Tempo: {result['execution_time']:.2f}s")
                print(f"    - Agentes vivos: {result['agents_alive']}")
                print(f"    - Tesouros encontrados: {result['treasures_found']}")
                print(f"    - Bombas desativadas: {result['bombs_deactivated']}")
                print(f"    - Bombas detonadas: {result['bombs_activated']}")
                print(f"    - Sucesso: {'Sim' if result['success'] else 'Não'}")
                
                results.append({
                    'approach': approach,
                    'group_type': group_type,
                    'status': 'OK',
                    'iterations': result['iterations_executed'],
                    'agents_alive': result['agents_alive'],
                    'treasures': result['treasures_found'],
                    'bombs_deactivated': result['bombs_deactivated'],
                    'bombs_activated': result['bombs_activated'],
                    'success': result['success']
                })
                
            except Exception as e:
                print(f"  ✗ ERRO na simulação: {str(e)}")
                results.append({
                    'approach': approach,
                    'group_type': group_type,
                    'status': 'ERRO',
                    'error': str(e)
                })
            
            print()
    
    # Resumo final
    print("=" * 80)
    print("RESUMO DOS TESTES")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for r in results:
        approach = r['approach']
        group_type = r['group_type']
        status = r['status']
        
        if status == 'OK':
            print(f"✓ Abordagem {approach} - {group_type}: OK")
            passed += 1
        else:
            print(f"✗ Abordagem {approach} - {group_type}: ERRO - {r.get('error', 'Desconhecido')}")
            failed += 1
    
    print()
    print(f"Total de testes: {len(results)}")
    print(f"Aprovados: {passed}")
    print(f"Falhados: {failed}")
    print()
    
    if failed == 0:
        print("=" * 80)
        print("TODOS OS TESTES PASSARAM COM SUCESSO!")
        print("=" * 80)
    else:
        print("=" * 80)
        print(f"ATENÇÃO: {failed} teste(s) falharam!")
        print("=" * 80)
    
    return results

if __name__ == "__main__":
    test_conformidade()
