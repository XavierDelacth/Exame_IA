"""
Módulo de Pré-processamento de Dados
Responsáveis: Adriana & Xavier (Conjunto)
Data: 11/01/2026

Este módulo contém funções para:
- Geração de dados de treinamento
- Extração de features
- Normalização e transformação
- Balanceamento de dados
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import sys

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.grid import Grid
from environment.cell import Cell, CellType
from environment.config import EnvironmentConfig


class FeatureExtractor:
    """
    Extrator de features para modelos de ML
    
    Extrai características relevantes de células e vizinhanças
    para alimentar os modelos de aprendizado.
    """
    
    @staticmethod
    def extract_cell_features(
        cell: Cell,
        grid: Grid,
        agent_position: Tuple[int, int] = None
    ) -> Dict[str, float]:
        """
        Extrai features de uma célula específica
        
        Args:
            cell: Célula a extrair features
            grid: Grid do ambiente
            agent_position: Posição do agente (opcional)
            
        Returns:
            Dicionário com features extraídas
        """
        features = {}
        
        # Posição da célula
        features['cell_x'] = cell.x
        features['cell_y'] = cell.y
        
        # Distância até o agente (se fornecida)
        if agent_position:
            agent_x, agent_y = agent_position
            features['distance_to_agent'] = abs(cell.x - agent_x) + abs(cell.y - agent_y)
            features['euclidean_distance'] = np.sqrt(
                (cell.x - agent_x)**2 + (cell.y - agent_y)**2
            )
        else:
            features['distance_to_agent'] = 0
            features['euclidean_distance'] = 0
        
        # Analisar vizinhos
        neighbors = grid.get_neighbors(cell.x, cell.y, include_diagonals=True)
        
        features['num_neighbors'] = len(neighbors)
        features['num_bombs_nearby'] = sum(1 for n in neighbors if n.is_bomb())
        features['num_treasures_nearby'] = sum(1 for n in neighbors if n.is_treasure())
        features['num_free_nearby'] = sum(1 for n in neighbors if n.is_free())
        features['num_explored_nearby'] = sum(1 for n in neighbors if n.explored)
        features['num_unexplored_nearby'] = sum(1 for n in neighbors if not n.explored)
        
        # Densidades
        total_neighbors = len(neighbors) if len(neighbors) > 0 else 1
        features['bomb_density'] = features['num_bombs_nearby'] / total_neighbors
        features['treasure_density'] = features['num_treasures_nearby'] / total_neighbors
        features['exploration_ratio'] = features['num_explored_nearby'] / total_neighbors
        
        # Estado da célula
        features['is_explored'] = 1 if cell.explored else 0
        features['is_edge'] = 1 if (cell.x == 0 or cell.x == grid.size - 1 or 
                                      cell.y == 0 or cell.y == grid.size - 1) else 0
        
        # Posição relativa no grid
        features['relative_x'] = cell.x / grid.size
        features['relative_y'] = cell.y / grid.size
        
        return features
    
    @staticmethod
    def extract_movement_features(
        from_cell: Cell,
        to_cell: Cell,
        grid: Grid
    ) -> Dict[str, float]:
        """
        Extrai features de um movimento entre células
        
        Args:
            from_cell: Célula de origem
            to_cell: Célula de destino
            grid: Grid do ambiente
            
        Returns:
            Dicionário com features do movimento
        """
        features = {}
        
        # Features da célula de destino
        dest_features = FeatureExtractor.extract_cell_features(
            to_cell, grid, agent_position=(from_cell.x, from_cell.y)
        )
        features.update(dest_features)
        
        # Direção do movimento
        dx = to_cell.x - from_cell.x
        dy = to_cell.y - from_cell.y
        
        features['move_direction_x'] = dx
        features['move_direction_y'] = dy
        
        # Tipo de movimento
        if dx == -1 and dy == 0:
            features['direction'] = 0  # Norte
        elif dx == 1 and dy == 0:
            features['direction'] = 1  # Sul
        elif dx == 0 and dy == 1:
            features['direction'] = 2  # Leste
        elif dx == 0 and dy == -1:
            features['direction'] = 3  # Oeste
        else:
            features['direction'] = 4  # Outro
        
        return features


class DataPreprocessor:
    """
    Pré-processador de dados para modelos de ML
    
    Normaliza, transforma e prepara dados para treinamento.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Ajusta o preprocessador aos dados
        
        Args:
            X: Features
            y: Labels (opcional)
        """
        self.scaler.fit(X)
        
        if y is not None:
            self.label_encoder.fit(y)
        
        self.is_fitted = True
    
    def transform(
        self, 
        X: np.ndarray, 
        y: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforma os dados
        
        Args:
            X: Features
            y: Labels (opcional)
            
        Returns:
            Tupla (X_transformed, y_transformed)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessador não foi fitted. Chame fit() primeiro.")
        
        X_scaled = self.scaler.transform(X)
        
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        
        return X_scaled, None
    
    def fit_transform(
        self, 
        X: np.ndarray, 
        y: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Ajusta e transforma em uma operação
        
        Args:
            X: Features
            y: Labels (opcional)
            
        Returns:
            Tupla (X_transformed, y_transformed)
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Inverte a transformação dos labels
        
        Args:
            y_encoded: Labels codificados
            
        Returns:
            Labels originais
        """
        return self.label_encoder.inverse_transform(y_encoded)
    
    def save(self, filepath: str):
        """
        Salva o preprocessador
        
        Args:
            filepath: Caminho do arquivo
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """
        Carrega um preprocessador salvo
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            DataPreprocessor carregado
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls()
        preprocessor.scaler = data['scaler']
        preprocessor.label_encoder = data['label_encoder']
        preprocessor.feature_names = data['feature_names']
        preprocessor.is_fitted = data['is_fitted']
        
        return preprocessor


def generate_training_data(
    num_samples: int = 10000,
    config: EnvironmentConfig = None,
    seed: int = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Gera dados de treinamento através de simulações
    
    Args:
        num_samples: Número de amostras a gerar
        config: Configuração do ambiente
        seed: Seed para reproducibilidade
        
    Returns:
        Tupla (features_df, labels_series)
    """
    if config is None:
        config = EnvironmentConfig(grid_size=10, seed=seed)
    
    if seed is not None:
        np.random.seed(seed)
    
    all_features = []
    all_labels = []
    
    # Gerar múltiplos ambientes
    num_environments = max(1, num_samples // 100)
    samples_per_env = num_samples // num_environments
    
    for env_idx in range(num_environments):
        # Criar novo ambiente
        grid = Grid(config)
        grid.setup_random(seed=seed + env_idx if seed else None)
        
        # Extrair samples deste ambiente
        for _ in range(samples_per_env):
            # Escolher célula aleatória
            x = np.random.randint(0, grid.size)
            y = np.random.randint(0, grid.size)
            cell = grid.get_cell(x, y)
            
            # Escolher posição de "agente" aleatória
            agent_x = np.random.randint(0, grid.size)
            agent_y = np.random.randint(0, grid.size)
            
            # Extrair features
            features = FeatureExtractor.extract_cell_features(
                cell, grid, agent_position=(agent_x, agent_y)
            )
            
            # Label é o tipo da célula
            if cell.is_bomb():
                label = 'BOMB'
            elif cell.is_treasure():
                label = 'TREASURE'
            else:
                label = 'FREE'
            
            all_features.append(features)
            all_labels.append(label)
    
    # Converter para DataFrame
    features_df = pd.DataFrame(all_features)
    labels_series = pd.Series(all_labels, name='cell_type')
    
    return features_df, labels_series


def balance_dataset(
    X: pd.DataFrame, 
    y: pd.Series,
    strategy: str = 'undersample'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balanceia o dataset
    
    Args:
        X: Features
        y: Labels
        strategy: 'undersample', 'oversample' ou 'none'
        
    Returns:
        Tupla (X_balanced, y_balanced)
    """
    if strategy == 'none':
        return X, y
    
    # Contar classes
    class_counts = y.value_counts()
    
    if strategy == 'undersample':
        # Reduzir para o tamanho da menor classe
        min_count = class_counts.min()
        
        indices = []
        for class_label in class_counts.index:
            class_indices = y[y == class_label].index.tolist()
            sampled_indices = np.random.choice(
                class_indices, 
                size=min_count, 
                replace=False
            )
            indices.extend(sampled_indices)
        
        X_balanced = X.loc[indices]
        y_balanced = y.loc[indices]
        
    elif strategy == 'oversample':
        # Aumentar para o tamanho da maior classe
        max_count = class_counts.max()
        
        indices = []
        for class_label in class_counts.index:
            class_indices = y[y == class_label].index.tolist()
            sampled_indices = np.random.choice(
                class_indices, 
                size=max_count, 
                replace=True
            )
            indices.extend(sampled_indices)
        
        X_balanced = X.loc[indices]
        y_balanced = y.loc[indices]
    
    else:
        raise ValueError(f"Estratégia inválida: {strategy}")
    
    return X_balanced, y_balanced


def save_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    filepath: str
):
    """
    Salva dados de treinamento
    
    Args:
        X: Features
        y: Labels
        filepath: Caminho do arquivo
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combinar X e y
    data = X.copy()
    data['label'] = y
    
    # Salvar
    data.to_csv(path, index=False)


def load_training_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carrega dados de treinamento
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Tupla (X, y)
    """
    data = pd.read_csv(filepath)
    
    # Separar features e labels
    y = data['label']
    X = data.drop('label', axis=1)
    
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide dados em treino e teste
    
    Args:
        X: Features
        y: Labels
        test_size: Proporção de teste
        random_state: Seed
        
    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )