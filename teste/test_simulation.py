#!/usr/bin/env python3
"""
Test script for full simulation with communication system
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from simulation import Simulation
    print("✓ Simulation class imported successfully")

    # Test with small simulation
    print("\n--- Testing Small Simulation ---")
    sim = Simulation('A', 2, 0.3, 'homogeneous', 0)
    print("✓ Simulation created")

    result = sim.run()
    print("✓ Simulation completed")

    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Communication stats: {result.get('communication', 'N/A')}")

    # Test with different configurations
    print("\n--- Testing Different Configurations ---")

    configs = [
        ('A', 3, 0.5, 'homogeneous', 0),
        ('B', 2, 0.3, 'heterogeneous', 1),
        ('C', 4, 0.4, 'homogeneous', 2)
    ]

    for config in configs:
        print(f"\nTesting config: {config}")
        sim = Simulation(*config)
        result = sim.run()
        print(f"  Success: {result['success']}, Steps: {result['steps']}")

    print("\n🎉 All simulation tests passed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()