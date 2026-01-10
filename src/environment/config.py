"""
Módulo de Configuração do Ambiente
Responsável: Adriana
Data: 7/01/2026
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class EnvironmentConfig:
    """
    Configurações do ambiente de simulação
    
    Attributes:
        grid_size: Tamanho da matriz (padrão 10x10)
        bombs_percentage: Percentual de bombas (0.0 a 1.0)
        treasures_count: Número de tesouros no ambiente
        max_steps: Número máximo de steps por simulação
        enable_flag: Se deve colocar bandeira (Abordagem C)
        seed: Seed para reproducibilidade
    """
    
    grid_size: int = 10
    bombs_percentage: float = 0.5
    treasures_count: int = 5
    max_steps: int = 1000
    enable_flag: bool = False
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validação após inicialização"""
        self.validate()
    
    def validate(self):
        """
        Valida as configurações
        
        Raises:
            ValueError: Se alguma configuração for inválida
        """
        if self.grid_size < 5 or self.grid_size > 50:
            raise ValueError(f"grid_size deve estar entre 5 e 50, recebido: {self.grid_size}")
        
        if not 0.0 <= self.bombs_percentage <= 1.0:
            raise ValueError(f"bombs_percentage deve estar entre 0.0 e 1.0, recebido: {self.bombs_percentage}")
        
        max_cells = self.grid_size * self.grid_size
        max_treasures = max_cells - int(max_cells * self.bombs_percentage) - 1  # -1 para bandeira se necessário
        
        if self.treasures_count < 0:
            raise ValueError(f"treasures_count não pode ser negativo: {self.treasures_count}")
        
        if self.treasures_count > max_treasures:
            raise ValueError(
                f"treasures_count ({self.treasures_count}) muito alto para o espaço disponível "
                f"(máximo: {max_treasures})"
            )
        
        if self.max_steps < 1:
            raise ValueError(f"max_steps deve ser positivo, recebido: {self.max_steps}")
    
    def get_total_bombs(self) -> int:
        """
        Calcula o número total de bombas
        
        Returns:
            Número de bombas baseado no percentual
        """
        total_cells = self.grid_size * self.grid_size
        return int(total_cells * self.bombs_percentage)
    
    def get_free_cells(self) -> int:
        """
        Calcula o número de células livres
        
        Returns:
            Número de células livres
        """
        total_cells = self.grid_size * self.grid_size
        bombs = self.get_total_bombs()
        treasures = self.treasures_count
        flag = 1 if self.enable_flag else 0
        return total_cells - bombs - treasures - flag
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte configuração para dicionário
        
        Returns:
            Dicionário com as configurações
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentConfig':
        """
        Cria configuração a partir de dicionário
        
        Args:
            data: Dicionário com as configurações
            
        Returns:
            Nova instância de EnvironmentConfig
        """
        return cls(**data)
    
    @classmethod
    def from_json_file(cls, filepath: str) -> 'EnvironmentConfig':
        """
        Carrega configuração de arquivo JSON
        
        Args:
            filepath: Caminho do arquivo JSON
            
        Returns:
            Nova instância de EnvironmentConfig
            
        Raises:
            FileNotFoundError: Se o arquivo não existir
            json.JSONDecodeError: Se o JSON for inválido
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extrair apenas a seção de environment se existir
        if 'environment' in data:
            data = data['environment']
        
        return cls.from_dict(data)
    
    def save_to_json_file(self, filepath: str):
        """
        Salva configuração em arquivo JSON
        
        Args:
            filepath: Caminho do arquivo JSON
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def copy(self) -> 'EnvironmentConfig':
        """
        Cria uma cópia da configuração
        
        Returns:
            Nova instância com os mesmos valores
        """
        return EnvironmentConfig(**self.to_dict())
    
    def __repr__(self) -> str:
        """Representação em string"""
        return (
            f"EnvironmentConfig("
            f"grid={self.grid_size}x{self.grid_size}, "
            f"bombs={self.bombs_percentage:.0%}, "
            f"treasures={self.treasures_count}, "
            f"flag={self.enable_flag})"
        )


# Configurações pré-definidas para diferentes cenários
class PresetConfigs:
    """Configurações pré-definidas para testes"""
    
    @staticmethod
    def easy() -> EnvironmentConfig:
        """Configuração fácil: poucas bombas, muitos tesouros"""
        return EnvironmentConfig(
            grid_size=10,
            bombs_percentage=0.2,
            treasures_count=8,
            max_steps=500
        )
    
    @staticmethod
    def medium() -> EnvironmentConfig:
        """Configuração média: balanceado"""
        return EnvironmentConfig(
            grid_size=10,
            bombs_percentage=0.5,
            treasures_count=5,
            max_steps=1000
        )
    
    @staticmethod
    def hard() -> EnvironmentConfig:
        """Configuração difícil: muitas bombas, poucos tesouros"""
        return EnvironmentConfig(
            grid_size=10,
            bombs_percentage=0.7,
            treasures_count=3,
            max_steps=1500
        )
    
    @staticmethod
    def extreme() -> EnvironmentConfig:
        """Configuração extrema: máximo de bombas"""
        return EnvironmentConfig(
            grid_size=10,
            bombs_percentage=0.8,
            treasures_count=2,
            max_steps=2000
        )
    
    @staticmethod
    def approach_a() -> EnvironmentConfig:
        """Configuração otimizada para Abordagem A (>50% tesouros)"""
        return EnvironmentConfig(
            grid_size=10,
            bombs_percentage=0.5,
            treasures_count=6,
            max_steps=1000,
            enable_flag=False
        )
    
    @staticmethod
    def approach_b() -> EnvironmentConfig:
        """Configuração otimizada para Abordagem B (exploração completa)"""
        return EnvironmentConfig(
            grid_size=10,
            bombs_percentage=0.4,
            treasures_count=5,
            max_steps=1500,
            enable_flag=False
        )
    
    @staticmethod
    def approach_c() -> EnvironmentConfig:
        """Configuração otimizada para Abordagem C (encontrar bandeira)"""
        return EnvironmentConfig(
            grid_size=10,
            bombs_percentage=0.5,
            treasures_count=4,
            max_steps=1000,
            enable_flag=True
        )