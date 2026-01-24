"# Exame_IA" 
# 🤖 Sistema de Exploração com Agentes Inteligentes

**Projeto de Inteligência Artificial - ISPTEC 2024/2025**

Sistema completo de simulação para avaliar o desempenho de agentes inteligentes na exploração colaborativa de ambientes desconhecidos, utilizando técnicas de aprendizagem de máquina e comparação com algoritmos clássicos.

---

## 📋 Sumário

- [Visão Geral](#visão-geral)
- [Características](#características)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Documentação](#documentação)
- [Resultados Esperados](#resultados-esperados)
- [Contribuidores](#contribuidores)
- [Licença](#licença)

---

## 🎯 Visão Geral

Este projeto implementa um **sistema multiagente** para exploração colaborativa de um ambiente discreto 10×10, onde agentes inteligentes devem:

- 🗺️ Explorar células desconhecidas
- 💣 Evitar bombas que os destroem
- 💎 Encontrar tesouros que conferem proteção
- 🚩 Localizar objetivos específicos
- 🤝 Compartilhar conhecimento em tempo real

### Três Abordagens de Sucesso

| Abordagem | Objetivo | Baseline |
|-----------|----------|----------|
| **A** | Descobrir >50% dos tesouros | Greedy Best-First |
| **B** | Explorar 100% do ambiente | BFS |
| **C** | Encontrar a bandeira | A* |

### Tipos de Grupos

- **Homogêneo:** Todos os agentes com mesma estratégia
- **Heterogêneo:** Agentes com estratégias diversificadas
- **Baseline:** Algoritmos clássicos de busca

---

## ✨ Características

### Modelos de Aprendizagem

- ✅ **K-Nearest Neighbors (KNN)**
- ✅ **Naive Bayes**
- ✅ **Random Forest**

### Sistema de Agentes

- ✅ Ensemble de modelos com pesos configuráveis
- ✅ Compartilhamento de conhecimento instantâneo
- ✅ Coordenação para evitar redundância
- ✅ Treinamento incremental

### Experimentação Científica

- ✅ 30 repetições independentes por configuração
- ✅ Sementes controladas para reprodutibilidade
- ✅ Geração automática de CSVs
- ✅ Análise estatística completa

### Visualizações

- ✅ Histogramas de tempo de execução
- ✅ Boxplots comparativos
- ✅ Gráficos de taxa de sucesso
- ✅ Análise de impacto de parâmetros
- ✅ Tabelas estatísticas

### Interface Web

- ✅ Configuração interativa
- ✅ Visualização em tempo real
- ✅ Logs detalhados
- ✅ Métricas ao vivo

---

## 🔧 Instalação

### Requisitos

- Python 3.8 ou superior
- pip

### Passo 1: Clonar ou baixar o projeto

```bash
mkdir projeto_ia_agentes
cd projeto_ia_agentes
```

### Passo 2: Criar estrutura de diretórios

```bash
mkdir static results plots
```

### Passo 3: Instalar dependências

```bash
pip install -r requirements.txt
```

**Dependências:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.1
flask==2.3.2
flask-cors==4.0.0
```

### Passo 4: Organizar arquivos

```
projeto_ia_agentes/
├── simulation.py              # Backend principal
├── experiment_runner.py       # Script de experimentação
├── visualization.py           # Geração de gráficos
├── server.py                  # Servidor Flask
├── example_usage.py           # Exemplos de uso
├── requirements.txt           # Dependências
├── README.md                  # Este arquivo
├── static/
│   └── index.html            # Interface web
├── results/                  # CSVs gerados
└── plots/                    # Gráficos gerados
```

---

## 🚀 Uso Rápido

### Opção 1: Interface Web (Recomendado)

```bash
# Iniciar servidor
python server.py

# Abrir navegador em:
# http://localhost:5000
```

### Opção 2: Linha de Comando

#### Executar simulação única

```python
from simulation import Simulation

sim = Simulation(
    approach='A',
    num_agents=5,
    bomb_ratio=0.6,
    group_type='heterogeneous',
    seed=42
)

metrics = sim.run()
print(metrics)
```

#### Executar 30 repetições

```python
from simulation import run_experiment

results_df = run_experiment(
    approach='A',
    num_agents=5,
    bomb_ratio=0.6,
    group_type='homogeneous',
    repetitions=30
)

print(results_df.describe())
```

#### Executar experimentos completos

```bash
python experiment_runner.py
```

Isto executará **~3.600 simulações** (leva 2-4 horas).

#### Gerar visualizações

```bash
python visualization.py
```

#### Exemplo passo a passo

```bash
python example_usage.py
```

---

## 📁 Estrutura do Projeto

### `simulation.py`

Classes principais do sistema:

- **Environment:** Ambiente 10×10 com células L, B, T, F
- **MLModel:** Wrapper para modelos ML
- **Agent:** Agente com ensemble de modelos
- **ClassicAlgorithm:** Greedy, BFS, A*
- **Simulation:** Sistema de simulação completo

### `experiment_runner.py`

Script de experimentação automatizada:

- Executa 30 repetições por configuração
- Testa todas as combinações de parâmetros
- Gera CSVs com resultados
- Produz relatório estatístico

### `visualization.py`

Geração de gráficos profissionais:

- Histogramas
- Boxplots
- Barras de taxa de sucesso
- Análise de agentes
- Tabelas estatísticas
- Testes estatísticos (Shapiro-Wilk, Mann-Whitney, Kruskal-Wallis)

### `server.py`

API REST com Flask:

```
POST /api/simulate      - Executar simulação
POST /api/experiment    - Executar 30 repetições
POST /api/compare       - Comparar configurações
GET  /api/results       - Listar resultados
GET  /api/result/<file> - Obter resultado específico
```

### `example_usage.py`

Tutorial interativo com 6 partes:

1. Simulação simples
2. Experimento com 30 repetições
3. Algoritmo baseline
4. Geração de visualizações
5. Comparação homogêneo vs heterogêneo
6. Análise de impacto de agentes

---

## 📊 Documentação

### Guias Disponíveis

1. **Guia de Instalação** - Instruções detalhadas de setup
2. **Template de Relatório** - Estrutura completa para documentação
3. **Exemplos de Uso** - Código comentado passo a passo
4. **Referências** - Links para documentação oficial

### Arquivos de Saída

#### CSVs (`results/`)

Campos principais:
- `approach` - A, B ou C
- `group_type` - homogeneous, heterogeneous, baseline
- `num_agents` - 2 a 10
- `bomb_ratio` - 0.5 a 0.8
- `execution_time` - Tempo em segundos
- `total_steps` - Passos executados
- `exploration_percentage` - % explorado
- `success` - True/False
- ... e mais métricas

#### Gráficos (`plots/`)

- `histogram_time_abordagem_X.png`
- `boxplot_comparison_abordagem_X.png`
- `success_rate_abordagem_X.png`
- `agents_vs_performance_abordagem_X.png`
- `tabela_estatisticas_abordagem_X.csv/png`
- `testes_estatisticos_abordagem_X.txt`

---

## 🎓 Resultados Esperados

### Comparações

O sistema permite comparar:

1. **Homogêneo vs Heterogêneo**
   - Qual tem melhor taxa de sucesso?
   - Qual é mais eficiente?

2. **ML vs Algoritmos Clássicos**
   - Agentes ML superam baselines?
   - Em quais métricas?

3. **Impacto de Parâmetros**
   - Mais agentes = melhor desempenho?
   - Densidade de bombas afeta qual grupo mais?

### Análise Estatística

- Testes de normalidade
- Comparações pareadas
- Comparações múltiplas
- Intervalos de confiança

### Visualizações

Gráficos profissionais prontos para:
- Relatório técnico
- Apresentação
- Defesa do projeto

---

## 👥 Contribuidores

**Grupo:** [NÚMERO DO GRUPO]

- [Nome do Integrante 1]
- [Nome do Integrante 2]
- [Nome do Integrante 3]

**Professor:** Bongo Cahisso

**Instituição:** ISPTEC

**Ano Letivo:** 2024/2025

---

## 📝 Licença

Este projeto é desenvolvido para fins educacionais como parte da disciplina de Inteligência Artificial do ISPTEC.

---

## 🆘 Suporte

### Dúvidas?

- **Email:** bongo.cahisso@isptec.co.ao
- **Sala dos Professores:** Procurar professor

### Problemas Comuns

#### Erro de importação
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Porta 5000 ocupada
Altere em `server.py`:
```python
app.run(port=5001)  # Mudar porta
```

#### Memória insuficiente
Reduza repetições ou configurações testadas.

---

## 📚 Referências

- **Russell & Norvig (2020).** Artificial Intelligence: A Modern Approach, 4th Ed.
- **Scikit-learn Documentation:** https://scikit-learn.org/
- **Pandas Documentation:** https://pandas.pydata.org/
- **Flask Documentation:** https://flask.palletsprojects.com/

---

## 🎯 Checklist para Defesa

- [ ] Todos os experimentos executados
- [ ] CSVs gerados (`results/`)
- [ ] Visualizações prontas (`plots/`)
- [ ] Interface web funcionando
- [ ] Relatório técnico completo
- [ ] Apresentação PowerPoint
- [ ] Código comentado e organizado
- [ ] Laptop testado e configurado

---

## 🚀 Próximos Passos

1. ✅ **Setup:** Instalar dependências
2. ✅ **Testar:** Executar `example_usage.py`
3. ✅ **Experimentar:** Rodar `experiment_runner.py`
4. ✅ **Visualizar:** Gerar gráficos com `visualization.py`
5. ✅ **Documentar:** Preencher template de relatório
6. ✅ **Preparar:** Criar apresentação
7. ✅ **Defender:** Apresentar projeto

---

**Desenvolvido com 💙 para o Projeto de IA - ISPTEC 2024/2025**

**Boa sorte! 🎓🚀**