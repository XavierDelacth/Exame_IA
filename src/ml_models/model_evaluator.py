"""
Módulo de Avaliação de Modelos
Responsável: Xavier Delacth
Data: 11/01/2026

Este módulo compara e avalia diferentes modelos de ML:
- Decision Tree vs KNN vs Naive Bayes
- Métricas de desempenho
- Visualizações comparativas
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import time
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EvaluationMetrics:
    """
    Métricas de avaliação de um modelo
    
    Attributes:
        model_name: Nome do modelo
        accuracy: Acurácia
        precision: Precisão (macro avg)
        recall: Recall (macro avg)
        f1_score: F1-Score (macro avg)
        confusion_matrix: Matriz de confusão
        classification_report: Relatório detalhado
        training_time: Tempo de treinamento (segundos)
        prediction_time: Tempo de predição (segundos)
        additional_metrics: Métricas adicionais
    """
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    classification_report: str
    training_time: float = 0.0
    prediction_time: float = 0.0
    additional_metrics: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return {
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'classification_report': self.classification_report,
            'additional_metrics': self.additional_metrics
        }
    
    def __repr__(self):
        return (f"EvaluationMetrics({self.model_name}: "
                f"acc={self.accuracy:.3f}, "
                f"f1={self.f1_score:.3f})")


class ModelEvaluator:
    """
    Avaliador de modelos de Machine Learning
    
    Compara múltiplos modelos e gera relatórios detalhados.
    """
    
    def __init__(self):
        self.models: Dict[str, any] = {}
        self.evaluations: Dict[str, EvaluationMetrics] = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def add_model(self, name: str, model: any):
        """
        Adiciona um modelo para avaliação
        
        Args:
            name: Nome do modelo
            model: Instância do modelo (sklearn ou custom)
        """
        self.models[name] = model
    
    def set_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ):
        """
        Define os dados de treino e teste
        
        Args:
            X_train: Features de treino
            X_test: Features de teste
            y_train: Labels de treino
            y_test: Labels de teste
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def evaluate_model(
        self,
        model_name: str,
        model: any = None
    ) -> EvaluationMetrics:
        """
        Avalia um modelo específico
        
        Args:
            model_name: Nome do modelo
            model: Modelo (usa o adicionado se None)
            
        Returns:
            EvaluationMetrics com resultados
        """
        if model is None:
            if model_name not in self.models:
                raise ValueError(f"Modelo '{model_name}' não encontrado")
            model = self.models[model_name]
        
        if self.X_train is None or self.X_test is None:
            raise ValueError("Dados não definidos. Use set_data() primeiro.")
        
        # Treinar modelo
        start_train = time.time()
        model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_train
        
        # Fazer predições
        start_pred = time.time()
        y_pred = model.predict(self.X_test)
        prediction_time = time.time() - start_pred
        
        # Calcular métricas
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, zero_division=0)
        
        # Métricas adicionais
        additional = {
            'num_train_samples': len(self.X_train),
            'num_test_samples': len(self.X_test),
            'num_features': self.X_train.shape[1] if len(self.X_train.shape) > 1 else 1,
            'training_time_per_sample': training_time / len(self.X_train),
            'prediction_time_per_sample': prediction_time / len(self.X_test)
        }
        
        # Criar métricas
        metrics = EvaluationMetrics(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            classification_report=report,
            training_time=training_time,
            prediction_time=prediction_time,
            additional_metrics=additional
        )
        
        self.evaluations[model_name] = metrics
        return metrics
    
    def evaluate_all(self) -> Dict[str, EvaluationMetrics]:
        """
        Avalia todos os modelos adicionados
        
        Returns:
            Dicionário com métricas de todos os modelos
        """
        for model_name in self.models.keys():
            self.evaluate_model(model_name)
        
        return self.evaluations
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compara todos os modelos avaliados
        
        Returns:
            DataFrame com comparação de métricas
        """
        if not self.evaluations:
            raise ValueError("Nenhum modelo foi avaliado ainda")
        
        comparison_data = []
        
        for model_name, metrics in self.evaluations.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.accuracy,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'F1-Score': metrics.f1_score,
                'Training Time (s)': metrics.training_time,
                'Prediction Time (s)': metrics.prediction_time,
                'Time per Sample (ms)': metrics.additional_metrics.get(
                    'prediction_time_per_sample', 0
                ) * 1000
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1-Score', ascending=False)
        
        return df
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, EvaluationMetrics]:
        """
        Retorna o melhor modelo baseado em uma métrica
        
        Args:
            metric: Métrica para comparação ('accuracy', 'f1_score', etc.)
            
        Returns:
            Tupla (nome_do_modelo, métricas)
        """
        if not self.evaluations:
            raise ValueError("Nenhum modelo foi avaliado ainda")
        
        best_model_name = None
        best_value = -1
        
        for model_name, metrics in self.evaluations.items():
            value = getattr(metrics, metric)
            if value > best_value:
                best_value = value
                best_model_name = model_name
        
        return best_model_name, self.evaluations[best_model_name]
    
    def get_ranking(self, metric: str = 'f1_score') -> List[Tuple[str, float]]:
        """
        Retorna ranking de modelos por métrica
        
        Args:
            metric: Métrica para ranking
            
        Returns:
            Lista de tuplas (nome, valor) ordenada
        """
        if not self.evaluations:
            raise ValueError("Nenhum modelo foi avaliado ainda")
        
        ranking = []
        for model_name, metrics in self.evaluations.items():
            value = getattr(metrics, metric)
            ranking.append((model_name, value))
        
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
    
    def print_summary(self):
        """Imprime resumo da avaliação"""
        if not self.evaluations:
            print("Nenhum modelo avaliado ainda.")
            return
        
        print("=" * 80)
        print("RESUMO DA AVALIAÇÃO DE MODELOS")
        print("=" * 80)
        print()
        
        # Comparação geral
        comparison_df = self.compare_models()
        print(comparison_df.to_string(index=False))
        print()
        
        # Melhor modelo
        best_name, best_metrics = self.get_best_model('f1_score')
        print(f"🏆 MELHOR MODELO: {best_name}")
        print(f"   F1-Score: {best_metrics.f1_score:.4f}")
        print(f"   Accuracy: {best_metrics.accuracy:.4f}")
        print()
        
        # Detalhes de cada modelo
        for model_name, metrics in self.evaluations.items():
            print("-" * 80)
            print(f"📊 {model_name}")
            print("-" * 80)
            print(f"Accuracy:  {metrics.accuracy:.4f}")
            print(f"Precision: {metrics.precision:.4f}")
            print(f"Recall:    {metrics.recall:.4f}")
            print(f"F1-Score:  {metrics.f1_score:.4f}")
            print(f"Training Time: {metrics.training_time:.3f}s")
            print(f"Prediction Time: {metrics.prediction_time:.3f}s")
            print()
            print("Confusion Matrix:")
            print(metrics.confusion_matrix)
            print()
        
        print("=" * 80)
    
    def save_results(self, filepath: str):
        """
        Salva resultados em arquivo CSV
        
        Args:
            filepath: Caminho do arquivo
        """
        comparison_df = self.compare_models()
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        comparison_df.to_csv(path, index=False)
    
    def get_confusion_matrices(self) -> Dict[str, np.ndarray]:
        """
        Retorna matrizes de confusão de todos os modelos
        
        Returns:
            Dicionário {nome_modelo: matriz_confusão}
        """
        return {
            name: metrics.confusion_matrix
            for name, metrics in self.evaluations.items()
        }
    
    def get_detailed_report(self, model_name: str) -> str:
        """
        Retorna relatório detalhado de um modelo
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            String com relatório completo
        """
        if model_name not in self.evaluations:
            raise ValueError(f"Modelo '{model_name}' não foi avaliado")
        
        metrics = self.evaluations[model_name]
        
        report = f"""
{'='*80}
RELATÓRIO DETALHADO: {model_name}
{'='*80}

MÉTRICAS GERAIS:
  Accuracy:  {metrics.accuracy:.4f}
  Precision: {metrics.precision:.4f}
  Recall:    {metrics.recall:.4f}
  F1-Score:  {metrics.f1_score:.4f}

PERFORMANCE:
  Training Time:   {metrics.training_time:.3f}s
  Prediction Time: {metrics.prediction_time:.3f}s
  Time per Sample: {metrics.additional_metrics.get('prediction_time_per_sample', 0)*1000:.2f}ms

DADOS:
  Training Samples: {metrics.additional_metrics.get('num_train_samples', 0)}
  Test Samples:     {metrics.additional_metrics.get('num_test_samples', 0)}
  Features:         {metrics.additional_metrics.get('num_features', 0)}

MATRIZ DE CONFUSÃO:
{metrics.confusion_matrix}

RELATÓRIO DE CLASSIFICAÇÃO:
{metrics.classification_report}

{'='*80}
"""
        return report