# Collaborative Multi-Agent Exploration System (Python Version)

This is a Python implementation of the collaborative multi-agent exploration system with a web interface using Streamlit.

## Overview

The system simulates multiple agents exploring a grid-based environment containing:
- Land cells (L)
- Bombs (B)
- Treasures (T)
- A flag (F)
- Unknown cells (U)

Agents use different machine learning heuristics (KNN, Decision Tree, Naive Bayes) to decide their next moves and share knowledge through a communication hub.

## Approaches

- **Approach A**: Success if more than 50% of treasures are found.
- **Approach B**: Success if the entire grid is explored and at least one agent survives.
- **Approach C**: Success if the flag is found.

## Models

- **Decision Tree**: Move towards the flag if known, otherwise random.
- **KNN**: Move towards the centroid of unexplored areas.
- **Naive Bayes**: Avoid areas near known bombs.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Direct Streamlit
```bash
streamlit run Core/app.py
```

### Option 2: Using the run script
```bash
python run.py
```

The application will open in your default web browser at http://localhost:8501 (or the next available port) with the same interface as the original TypeScript version.

## Features

- Interactive grid visualization
- Real-time simulation
- Configurable parameters (number of agents, bomb ratio, approach)
- Live statistics and logs
- Performance rankings
- Session history

## Architecture

### Project Structure
```
python_version/
├── Core/                    # Core application files
│   ├── app.py              # Main Streamlit application
│   ├── data_types.py       # Type definitions and data structures
│   ├── run.py              # Execution script
│   └── test.py             # Test utilities
├── Agentes/                # AI Agent models
│   ├── decision_tree_model.py
│   ├── knn_model.py
│   └── naive_bayes_model.py
├── Comunicacao/            # Communication system
│   └── communication_hub.py
├── Abordagem/              # Success evaluation approaches
│   ├── approach_a.py
│   ├── approach_b.py
│   └── approach_c.py
├── requirements.txt        # Python dependencies
└── README.md              # This documentation
```

### Components
- **Core**: Main application logic and data structures
- **Agentes**: AI models for agent decision-making
- **Comunicacao**: Knowledge sharing between agents
- **Abordagem**: Success criteria evaluation functions