# Pasta de Testes

Esta pasta contém todos os arquivos de teste e exemplos do sistema de simulação multi-agente.

## Arquivos Incluídos

### Arquivos Individuais de Teste
- **`test_simple.py`**: Teste básico de criação e execução de simulação
- **`test_environment.py`**: Teste de geração de ambientes para diferentes abordagens (A, B, C)
- **`test_dispersion.py`**: Teste de movimento e dispersão inicial dos agentes

### Arquivos de Exemplo e Experimentação
- **`example_usage.py`**: Tutorial interativo com exemplos passo a passo de uso do sistema
- **`experiment_runner.py`**: Executor completo de experimentos com múltiplas configurações

### Arquivo Consolidado
- **`testes.py`**: Arquivo que consolida todos os testes acima em um único local

## Como Usar

### Executar Todos os Testes Básicos
```bash
cd teste
python testes.py
```

### Executar Teste Específico
```bash
cd teste
python test_simple.py
python test_environment.py
python test_dispersion.py
```

### Executar Exemplos Completos
```bash
cd teste
python example_usage.py
```

### Executar Experimentos Completos
```bash
cd teste
python experiment_runner.py
```

## Funcionalidades dos Testes

- **Teste Simples**: Verifica se a simulação pode ser criada e executada
- **Teste Ambiente**: Valida a geração correta de ambientes para todas as abordagens
- **Teste Dispersão**: Verifica o movimento inicial e dispersão dos agentes
- **Exemplo de Uso**: Demonstra passo a passo como usar o sistema
- **Executor de Experimentos**: Roda experimentos completos com múltiplas configurações

## Dependências

Certifique-se de que o ambiente virtual está ativado:
```bash
# No diretório raiz do projeto
venv_ia\Scripts\activate  # Windows
# ou
source venv_ia/bin/activate  # Linux/Mac
```

Todos os testes importam módulos do diretório pai, então devem ser executados a partir desta pasta ou com o caminho correto configurado.

## Estrutura dos Arquivos

```
teste/
├── README.md              # Este arquivo
├── testes.py              # Arquivo consolidado de todos os testes
├── test_simple.py         # Teste básico
├── test_environment.py    # Teste de ambientes
├── test_dispersion.py     # Teste de dispersão
├── example_usage.py       # Exemplos de uso
└── experiment_runner.py   # Executor de experimentos
```