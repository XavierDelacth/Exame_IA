#!/usr/bin/env python3
"""
Simple test for agent creation
"""

from simulation import Agent, Environment
import numpy as np

# Create environment
env = Environment(5, 0.3)
print('Environment created')

# Create agent
agent = Agent('test', (0,0))
print('Agent created')

# Test basic functionality
obs = agent.observe(env, 0, 0)
print(f'Observation shape: {obs.shape}')

# Test model prediction with simple data
try:
    # First train the model with some data
    agent.update_knowledge(env, 0, 0, 'L')
    agent.update_knowledge(env, 1, 1, 'B')
    agent.update_knowledge(env, 2, 2, 'T')
    print('Model trained with sample data')

    # Now test prediction
    test_obs = agent.observe(env, 3, 3)
    proba = agent.models['knn'].predict_proba(test_obs.reshape(1, -1))
    print(f'Prediction successful: {proba.shape}')
except Exception as e:
    print(f'Model prediction error: {e}')