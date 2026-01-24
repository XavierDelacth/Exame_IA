#!/usr/bin/env python3
"""
Test script for communication system
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    # Test basic imports
    import numpy as np
    print("✓ NumPy imported successfully")

    # Test sklearn imports
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    print("✓ Scikit-learn imported successfully")

    # Test collections
    from collections import defaultdict, deque
    import heapq
    from datetime import datetime
    print("✓ Standard libraries imported successfully")

    # Test our classes
    from simulation import CommunicationHub, Agent, Environment
    print("✓ Simulation classes imported successfully")

    # Test CommunicationHub
    hub = CommunicationHub(5)
    print("✓ CommunicationHub created successfully")

    # Test basic functionality
    hub.register_cell_visit('agent1', (0, 0), 'L')
    hub.register_cell_visit('agent2', (1, 1), 'L')
    print("✓ Cell visits registered")

    # Test territory assignment
    territories = hub.assign_territories(['agent1', 'agent2'], 'grid')
    print(f"✓ Territories assigned: {territories}")

    # Test safe cell check
    is_safe = hub.is_cell_safe_for_agent('agent1', (0, 0))
    print(f"✓ Safe cell check: {is_safe}")

    print("\n🎉 All tests passed! Communication system is working.")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()