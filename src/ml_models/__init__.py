"""
Módulo de Modelos de Machine Learning
Responsáveis: 
- Adriana: Decision Tree, KNN, Preprocessing (parte)
- Henrique: Naive Bayes, Model Evaluator, Preprocessing (parte)

Este módulo contém:
- Preprocessing: Preparação de dados
- DecisionTreeModel: Modelo de Árvore de Decisão
- KNNModel: Modelo K-Nearest Neighbors
- NaiveBayesModel: Modelo Naive Bayes
- ModelEvaluator: Comparação e avaliação de modelos
"""

from .preprocessing import (
    DataPreprocessor,
    FeatureExtractor,
    generate_training_data,
    load_training_data,
    save_training_data
)
from .decision_tree import DecisionTreeModel
from .knn import KNNModel
from .naive_bayes import NaiveBayesModel
from .model_evaluator import ModelEvaluator, EvaluationMetrics

__all__ = [
    # Preprocessing
    'DataPreprocessor',
    'FeatureExtractor',
    'generate_training_data',
    'load_training_data',
    'save_training_data',
    
    # Models
    'DecisionTreeModel',
    'KNNModel',
    'NaiveBayesModel',
    
    # Evaluation
    'ModelEvaluator',
    'EvaluationMetrics'
]