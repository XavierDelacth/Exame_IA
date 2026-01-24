#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from simulation import Simulation

print("Testando simulação...")
sim = Simulation('A', 2, 0.3, 'homogeneous')
print('Simulação criada com sucesso')
result = sim.run(max_iterations=2)
print('Simulação executada com sucesso')
print('Posições finais:', [a.position for a in sim.agents])
print('Sucesso!')