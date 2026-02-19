# 📘 Documentação do Projeto IA - Exploração Colaborativa de Ambientes

## 📋 Índice
1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Estrutura de Ficheiros](#estrutura-de-ficheiros)
3. [Como O Projeto Funciona](#como-o-projeto-funciona)
4. [As Três Abordagens](#as-três-abordagens)
5. [Como Executar](#como-executar)
6. [Fluxo de Dados e Resultados](#fluxo-de-dados-e-resultados)
7. [Sistemas de Análise](#sistemas-de-análise)
8. [Geração de Relatórios](#geração-de-relatórios)

---

## 📌 Visão Geral do Projeto

Este projeto implementa **três abordagens diferentes** para exploração colaborativa de ambientes desconhecidos por múltiplos agentes de inteligência artificial.

**Objetivo**: Comparar o desempenho de agentes homogéneos vs heterogéneos vs baseline em diferentes cenários:
- **Abordagem A**: Coleta de tesouros evitando bombas
- **Abordagem B**: Exploração completa do ambiente
- **Abordagem C**: Localização de uma bandeira desconhecida

Cada abordagem é simulada **30 vezes** em 3 configurações diferentes (**baseline**, **homogeneous**, **heterogeneous**), totalizando 270 simulações por abordagem completa.

---

## 📁 Estrutura de Ficheiros

```
projeto_ia_agentes/
├── 📄 DOCUMENTACAO_PROJETO.md          # Esta documentação
├── requirements.txt                     # Dependências Python
│
├── 🎯 ABORDAGENS (kernel da IA)
├── abordagem/
│   ├── __init__.py
│   ├── abordagem_a.py                  # Coleta de tesouros (com baselines)
│   ├── abordagem_b.py                  # Exploração BFS (com baselines)
│   └── abordagem_c.py                  # Localização de bandeira A* (com baselines)
│
├── 🔧 EXECUÇÃO DE SIMULAÇÕES
├── run_simulations_a.py                # Script para rodar 30x3=90 sims (Abordagem A)
├── run_simulations_b.py                # Script para rodar 30x3=90 sims (Abordagem B)
├── run_simulations_c.py                # Script para rodar 30x3=90 sims (Abordagem C)
├── test_abordagem_b.py                 # Testes unitários Abordagem B
│
├── 📊 ANÁLISE E VISUALIZAÇÃO
├── analise/
│   ├── __init__.py
│   ├── comparative_analysis.py         # DataStorage + MetricsCalculator + Visualizadores
│   └── gui_integration.py              # Integração da análise com a GUI
│
├── 🖥️ INTERFACE GRÁFICA
├── gui.py                              # Interface Tkinter completa
│
├── 📈 RELATÓRIOS
├── relatorio.py                        # Gerador automático de relatórios markdown
├── analise_dados.py                    # Script de análise exploratória
├── postprocess_all_results.py          # Pós-processamento de resultados
│
├── 📂 RESULTADOS (criadas durante execução)
├── simulation_results/
│   ├── all_results.json               # Base de dados JSON com TODOS os resultados
│   ├── init_all_results.py            # Script para inicializar o JSON
│   └── cleanup_results.py             # Script para limpar/resetar resultados
│
├── 📂 DADOS EXPORTADOS
├── data/
│   ├── dados_simulacao_*.csv          # Exportações CSV dos resultados
│   └── RELATORIO_ANALISE.md           # Relatórios markdown gerados
│
└── FicheirosR/                         # Análises em R (opcional)
    └── RELATORIO_ANALISE.md

```

---

## ▶️ Ficheiros Executáveis Principais

Este projeto tem 5 ficheiros Python principais que você deve executar, conforme o caso de uso:

### **1. 🖥️ gui.py - Interface Gráfica Interativa**

**Localização**: `gui.py` (raiz do projeto)

**Função**: Fornece uma interface gráfica completa (Tkinter) para executar simulações manualmente, visualizar resultados em tempo real e fazer análises comparativas.

**Como Executar**:
```bash
python gui.py
```

**O que oferece**:

| Funcionalidade | Descrição |
|---|---|
| **Simulação Manual** | Execute uma simulação por vez com parâmetros customizados |
| **Seleção de Abordagem** | Escolha A (tesouros), B (exploração) ou C (bandeira) |
| **Configuração** | Ajuste nº agentes, proporção bombas, nº tesouros, max passos |
| **Visualização Grid** | Veja o ambiente em tempo real durante a simulação |
| **Métricas em Tempo Real** | Acompanhe progresso (agentes, objetivos alcançados, etc) |
| **Gráficos** | Boxplot, scatter, linha de tendência entre métricas |
| **Batch Simulations** | Botão para executar 30 simulações automaticamente |
| **Exportação** | Salve gráficos e dados em ficheiro |
| **Gestão de Dados** | Ver estatísticas, exportar CSV, limpar resultados |

**Interface Principais**:
```
┌─────────────────────────────────────────────────────┐
│          PROJETO IA - EXPLORAÇÃO COLABORATIVA       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [ABORDAGEM A/B/C] [Nº Agentes: 4] [Bombas: 30%]   │
│  [Tesouros: 12]    [Max Passos: 300]               │
│                                                     │
│  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │   GRID 10x10         │  │   MÉTRICAS           │ │
│  │  [T][B][L][L]...     │  │  Tesouro: 85%        │ │
│  │  [L][L][B][T]...     │  │  Agentes Vivos: 4    │ │
│  │  ...                 │  │  Passos: 150         │ │
│  └──────────────────────┘  └──────────────────────┘ │
│                                                     │
│  [▶ SIMULAR] [⏸ PARAR] [📊 COMPARAR] [💾 EXPORT]   │
│  [🔬 BATCH] [📈 GRÁFICO] [🗑️ LIMPAR]              │
│                                                     │
│  LOG:                                               │
│  ✅ Simulação iniciada                             │
│  📍 Agente 1 em (2,3)                              │
│  ⚠️ Bomba encontrada em (4,5)                      │
│  🏆 Tesouro coletado! (1/12)                       │
│  ✅ Simulação completa em 150 passos               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Botões Disponíveis**:
- **▶ SIMULAR**: Inicia uma simulação com parâmetros selecionados
- **📊 COMPARAR**: Abre janela de comparação entre abordagens/grupos
- **🔬 BATCH**: Executa 30 simulações automaticamente em background
- **📈 GRÁFICO**: Escolhe tipo de gráfico (boxplot, scatter, etc)
- **💾 EXPORT**: Exporta dados/gráficos para CSV ou PNG
- **🗑️ LIMPAR**: Reset dos resultados armazenados

**Tempo de Execução**: ~2-3 segundos por simulação manual

---

### **2. 🔄 run_simulations_a.py - Executar 90 Simulações (Abordagem A)**

**Localização**: `run_simulations_a.py` (raiz do projeto)

**Função**: Executa automaticamente 90 simulações da Abordagem A (coleta de tesouros) em 3 grupos diferentes. Ideal para experimentos em batch.

**Como Executar**:
```bash
python run_simulations_a.py
```

**O que faz**:
```
├── 30 simulações BASELINE
│   └── Estratégia greedy (sem aprendizado ML)
├── 30 simulações HOMOGENEOUS  
│   └── Todos agentes usam MESMO modelo ML
└── 30 simulações HETEROGENEOUS
    └── Cada agente usa modelo ML DIFERENTE
    
TOTAL: 90 simulações salvas em simulation_results/all_results.json
```

**Parâmetros Padrão** (configurável no código):
```python
num_agents = 4           # Número de agentes
bomb_ratio = 0.3         # 30% do grid são bombas
treasure_count = 12      # 12 tesouros a recolher
max_steps = 300          # Máximo 300 passos
```

**Output Console**:
```
[A] Saved run 1/30 for group baseline
[A] Saved run 2/30 for group baseline
...
[A] Saved run 30/30 for group baseline
[A] Saved run 1/30 for group homogeneous
...
[A] Saved run 30/30 for group heterogeneous
```

**Tempo de Execução**: ~2-3 minutos (para 90 sims)

**Métricas Salvas**:
- `treasure_percentage`: % de tesouros recolhidos
- `success_rate`: Taxa de sucesso
- `exploration_efficiency`: Eficiência de exploração
- `reward_risk_ratio`: Razão recompensa-risco
- `avg_steps_to_treasure`: Passos médios até tesouro

**Ficheiro de Saída**: `simulation_results/all_results.json` (append aos dados existentes)

---

### **3. 🔄 run_simulations_b.py - Executar 90 Simulações (Abordagem B)**

**Localização**: `run_simulations_b.py` (raiz do projeto)

**Função**: Executa automaticamente 90 simulações da Abordagem B (exploração completa) em 3 grupos diferentes.

**Como Executar**:
```bash
python run_simulations_b.py
```

**O que faz**:
```
├── 30 simulações BASELINE
│   └── BFS puro (sem aprendizado ML)
├── 30 simulações HOMOGENEOUS  
│   └── BFS + ML colaborativo (mesmo modelo)
└── 30 simulações HETEROGENEOUS
    └── BFS + ML diferenciado (modelos diversos)
    
TOTAL: 90 simulações salvas em simulation_results/all_results.json
```

**Parâmetros Padrão**:
```python
num_agents = 4           # Número de agentes
bomb_ratio = 0.3         # 30% do grid são bombas
max_steps = 300          # Máximo 300 passos
# Nota: Abordagem B NÃO usa treasure_count (sem tesouros)
```

**Output Console**:
```
[B] Saved run 1/30 for group baseline
[B] Saved run 2/30 for group baseline
...
[B] Saved run 30/30 for group heterogeneous
```

**Tempo de Execução**: ~2-3 minutos (para 90 sims)

**Métricas Salvas**:
- `explored_percentage`: % do ambiente explorado
- `agents_alive`: Quantos agentes sobrevivem
- `cells_per_step`: Eficiência (células/passo)
- `safety_coverage_score`: Taxa exploração com segurança

**Ficheiro de Saída**: `simulation_results/all_results.json` (append aos dados existentes)

---

### **4. 🔄 run_simulations_c.py - Executar 90 Simulações (Abordagem C)**

**Localização**: `run_simulations_c.py` (raiz do projeto)

**Função**: Executa automaticamente 90 simulações da Abordagem C (localização de bandeira) em 3 grupos diferentes.

**Como Executar**:
```bash
python run_simulations_c.py
```

**O que faz**:
```
├── 30 simulações BASELINE
│   └── A* puro (sem aprendizado ML)
├── 30 simulações HOMOGENEOUS  
│   └── A* + ML colaborativo (mesmo modelo)
└── 30 simulações HETEROGENEOUS
    └── A* + ML diferenciado (modelos diversos)
    
TOTAL: 90 simulações salvas em simulation_results/all_results.json
```

**Parâmetros Padrão**:
```python
num_agents = 4           # Número de agentes
bomb_ratio = 0.3         # 30% do grid são bombas
treasure_count = 10      # 10 tesouros (opcionais, como pistas)
max_steps = 300          # Máximo 300 passos
```

**Output Console**:
```
[C] Saved run 1/30 for group baseline
[C] Saved run 2/30 for group baseline
...
[C] Saved run 30/30 for group heterogeneous
```

**Tempo de Execução**: ~2-3 minutos (para 90 sims)

**Métricas Salvas**:
- `success`: Bandeira encontrada? (1=sim, 0=não)
- `success_rate`: % de execuções com sucesso
- `avg_steps_to_flag`: Passos médios até bandeira
- `path_efficiency`: Passos reais / Caminho ótimo

**Ficheiro de Saída**: `simulation_results/all_results.json` (append aos dados existentes)

---

### **5. 📊 relatorio.py - Gerador de Relatório Automático**

**Localização**: `relatorio.py` (raiz do projeto)

**Função**: Lê os dados em `all_results.json` e gera um relatório markdown completo com análise estatística por abordagem.

**Como Executar - Opção 1 (Direto)**:
```bash
python relatorio.py
```

Isso geralmente gera um relatório e o salva automaticamente.

**Como Executar - Opção 2 (Com controlo)**:
```python
from relatorio import RelatorioGerador

# Inicializar
gerador = RelatorioGerador('simulation_results/all_results.json')

# Gerar relatório completo
relatorio_text = gerador.gerar_relatorio_completo()

# Salvar em ficheiro markdown
gerador.salvar_relatorio('RELATORIO_ANALISE.md')

# Ou visualizar em consola
print(relatorio_text)
```

**O que faz**:

Analisa dados agrupa por **abordagem** e **grupo** (baseline/homogeneous/heterogeneous) e responde a 6 perguntas por abordagem:

**Para Abordagem A (Coleta de Tesouros)**:
1. ✅ Qual é a taxa média de sucesso em cada grupo?
2. ✅ Qual grupo recolhe mais tesouros em média?
3. ✅ Qual é a eficiência de exploração?
4. ✅ Qual é a razão recompensa-risco?
5. ✅ Qual grupo é mais rápido em encontrar tesouros?
6. ✅ Há diferença significativa entre grupos?

**Para Abordagem B (Exploração)**:
1. ✅ Qual é a percentagem média explorada?
2. ✅ Qual grupo mantém mais agentes vivos?
3. ✅ Qual é a velocidade de exploração?
4. ✅ Qual é a cobertura segura?
5. ✅ Qual grupo é mais eficiente?
6. ✅ Qual estratégia é mais robusta?

**Para Abordagem C (Bandeira)**:
1. ✅ Qual é a taxa de sucesso?
2. ✅ Qual grupo é mais rápido?
3. ✅ Qual é a eficiência do caminho?
4. ✅ Há diferença em usar tesouros como pistas?
5. ✅ Qual estratégia é mais confiável?
6. ✅ Ranking geral de desempenho

**Output Exemplo** (no ficheiro markdown):
```markdown
# 📊 RELATÓRIO DE ANÁLISE - PROJETO IA

## Abordagem A: Coleta de Tesouros

### 1️⃣ Taxa Média de Sucesso
| Grupo | Sucesso (%) | Intervalo |
|-------|------------|-----------|
| Baseline | 82.45 ± 8.34 | 71-95% |
| Homogeneous | 88.92 ± 6.12 | 78-98% |
| Heterogeneous | 91.23 ± 5.45 | 82-99% |

**Conclusão**: Heterogeneous é 8.78% melhor que Baseline.

### 2️⃣ Coleta de Tesouros
...

### 3️⃣ Eficiência de Exploração
...
```

**Tempo de Execução**: ~1-2 segundos (leitura e análise)

**Ficheiros de Saída**: 
- `RELATORIO_ANALISE.md` (ou outro nome escolhido)
- Contém tabelas, gráficos em ASCII, conclusões

**Requisitos**: 
- Ter dados em `simulation_results/all_results.json`
- Mínimo: 30 simulações por grupo para análise significativa

---

### **6. 🔧 postprocess_all_results.py - Pós-Processamento de Dados**

**Localização**: `postprocess_all_results.py` (raiz do projeto)

**Função**: Processa dados brutos de `all_results.json` para limpeza, normalização, remoção de outliers e geração de resumos.

**Como Executar - Opção 1 (Direto)**:
```bash
python postprocess_all_results.py
```

Executa processamento padrão e salva resultados processados.

**Como Executar - Opção 2 (Com controlo)**:
```python
from postprocess_all_results import PostProcessor

# Inicializar
processor = PostProcessor('simulation_results/all_results.json')

# Remover outliers (opcional)
processor.remove_outliers(approach='A', metric='treasure_percentage', threshold=2.5)

# Normalizar métricas
processor.normalize_metrics(approach='A')

# Gerar resumo agregado
summary = processor.generate_summary()

# Salvar dados processados
processor.save_processed_results('simulation_results/all_results_processed.json')
```

**O que faz**:

| Operação | Descrição |
|----------|-----------|
| **Validação** | Verifica integridade dos dados (timestamps, tipos, valores) |
| **Limpeza** | Remove entradas duplicadas ou incompletas |
| **Outliers** | Identifica e opcionalmente remove valores extremos |
| **Normalização** | Normaliza métricas para escala 0-1 para comparação |
| **Agregação** | Agrupa por abordagem, grupo, parâmetros |
| **Resumo** | Calcula média, desvio, mediana, quartis |
| **Correlações** | Analisa relações entre métricas |

**Output Exemplo**:
```
POSTPROCESSAMENTO INICIADO
====================================
✅ Dados carregados: 270 registros

VALIDAÇÃO:
  - Entradas válidas: 268/270 (99.3%)
  - Timestamps: OK
  - Tipos de dados: OK

LIMPEZA:
  - Duplicatas removidas: 2
  - Entradas incompletas: 0

OUTLIERS (2.5σ):
  - Abordagem A - treasure_percentage: 1 outlier removido
  - Abordagem B - explored_percentage: 0 outliers
  - Abordagem C - success_rate: 0 outliers

NORMALIZAÇÃO:
  ✅ Abordagem A: 87 métricas normalizadas
  ✅ Abordagem B: 89 métricas normalizadas
  ✅ Abordagem C: 88 métricas normalizadas

RESUMO POR GRUPO:
┌──────────────────────────────────┐
│ Abordagem A - BASELINE           │
├──────────────────────────────────┤
│ Treasure %: 75.45 ± 8.23         │
│ Success Rate: 85.00 ± 10.12      │
│ Efficiency: 0.0245 ± 0.0034      │
└──────────────────────────────────┘
...

====================================
✅ Postprocessamento completo!
📁 Dados salvos em: all_results_processed.json
```

**Tempo de Execução**: ~2-5 segundos (depende do volume)

**Ficheiros de Saída**:
- `simulation_results/all_results_processed.json` (dados limpos)
- `data/postprocessing_report.txt` (relatório de limpeza)

**Requisitos**:
- Ter dados em `simulation_results/all_results.json`

---

## 🎯 Fluxo Recomendado de Execução

### **Cenário 1: Experimento Novo (Início do Zero)**

```bash
# 1. Inicializar JSON vazio
python simulation_results/init_all_results.py

# 2. Executar as 3 abordagens
python run_simulations_a.py  # ~3 min
python run_simulations_b.py  # ~3 min
python run_simulations_c.py  # ~3 min
# Total: ~9 minutos para 270 simulações

# 3. Pós-processar dados
python postprocess_all_results.py

# 4. Gerar relatório markdown
python -c "from relatorio import RelatorioGerador; g = RelatorioGerador(); g.salvar_relatorio('RELATORIO_FINAL.md')"

# 5. Visualizar e analisar na GUI
python gui.py
```

---

### **Cenário 2: Análise Rápida (Dados Já Existem)**

```bash
# 1. Gerar relatório
python relatorio.py

# 2. Abrir GUI para exploração visual
python gui.py

# 3. (Opcional) Exportar para CSV
python -c "from analise.comparative_analysis import DataStorage; DataStorage().export_to_csv()"
```

---

### **Cenário 3: Executar Uma Única Abordagem**

```bash
# Se só quer Abordagem A
python run_simulations_a.py

# Analisar essa abordagem na GUI
python gui.py
# → Selecionar "Approach A" e comparar grupos
```

---

### **Cenário 4: Limpeza Total (Reset dos Dados)**

```bash
# Limpar todos os resultados
python simulation_results/cleanup_results.py

# Reiniciar JSON vazio
python simulation_results/init_all_results.py

# Re-executar simulações desde zero
python run_simulations_a.py
python run_simulations_b.py
python run_simulations_c.py
```

---

## 📝 Resumo Rápido de Ficheiros Executáveis

| Ficheiro | Comando | Tempo | Saída | Uso |
|----------|---------|-------|-------|-----|
| **gui.py** | `python gui.py` | Contínuo | Interativo | Interface gráfica |
| **run_simulations_a.py** | `python run_simulations_a.py` | ~3 min | 90 sims | Abordagem A em batch |
| **run_simulations_b.py** | `python run_simulations_b.py` | ~3 min | 90 sims | Abordagem B em batch |
| **run_simulations_c.py** | `python run_simulations_c.py` | ~3 min | 90 sims | Abordagem C em batch |
| **relatorio.py** | `python relatorio.py` | ~2 seg | Markdown | Gerar relatório |
| **postprocess_all_results.py** | `python postprocess_all_results.py` | ~5 seg | JSON | Limpar dados |
| **analise_dados.py** | `python analise_dados.py` | ~1 seg | Consola | Resumo estatístico |

---

## 🔄 Como O Projeto Funciona

### Fluxo Geral

```
┌─────────────────────────────────────────────────────────────┐
│ 1. ESCOLHER ABORDAGEM E GRUPO (via GUI ou scripts)         │
├─────────────────────────────────────────────────────────────┤
│   • Abordagem: A (tesouros) | B (exploração) | C (bandeira) │
│   • Grupo: baseline | homogeneous | heterogeneous          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. GERAR AMBIENTE                                           │
├─────────────────────────────────────────────────────────────┤
│   • Grid 10x10 com bombas, tesouros/bandeira               │
│   • Garantia de resolubilidade (caminho acessível)         │
│   • Proporção de bombas: 20-80% (configurável)             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. EXECUTAR SIMULAÇÃO                                       │
├─────────────────────────────────────────────────────────────┤
│   • 4 agentes (padrão, configurável)                        │
│   • Máx 300 passos (configurável)                          │
│   • Agentes usam ML (K-NN, Naive Bayes, Random Forest)     │
│   • Comunicação entre agentes (memória partilhada)         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. CALCULAR MÉTRICAS                                        │
├─────────────────────────────────────────────────────────────┤
│   • Taxa de sucesso, eficiência, exploração               │
│   • Recompensa-risco, passos até objetivo                  │
│   • Células exploradas, agentes vivos                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ARMAZENAR RESULTADOS                                     │
├─────────────────────────────────────────────────────────────┤
│   • Salvar em: simulation_results/all_results.json         │
│   • Incluir: timestamp, parâmetros, métricas               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. ANÁLISE E COMPARAÇÃO                                     │
├─────────────────────────────────────────────────────────────┤
│   • GUI: comparar abordagens, grupos e parâmetros          │
│   • Gráficos: boxplot, scatter, linha de tendência        │
│   • Exportar para CSV e Markdown                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 As Três Abordagens

### **ABORDAGEM A: Coleta de Tesouros** 
**Ficheiro**: [abordagem/abordagem_a.py](abordagem/abordagem_a.py)

#### Objetivo
Recolher o máximo número de tesouros evitando bombas.

#### Ambiente
- Grid 10x10
- **Tesouros** (T): ~12 distribuídos em quadrantes
- **Bombas** (B): proporção configurável (20-80%)
- **Células livres** (L): passáveis (≥20% garantido)

#### Estratégia
1. **Exploração ML**: Cada agente aprende a identificar padrões de bombas
2. **Cooperação**: Compartilham locais descobertos numa memória partilhada
3. **Classificação**: K-NN, Naive Bayes ou Random Forest (configurável)

#### Métricas
| Métrica | Descrição |
|---------|-----------|
| `treasure_percentage` | % de tesouros recolhidos |
| `success_rate` | Taxa de sucesso (todos tesouros coletados?) |
| `exploration_efficiency` | Tesouros/passos totais |
| `reward_risk_ratio` | Tesouros recolhidos / Bombas encontradas |
| `avg_steps_to_treasure` | Passos médios para cada tesouro |

#### Simulações
```
run_simulations_a.py
├── baseline (no learning, greedy)
├── homogeneous (ML idêntico em todos agentes)
└── heterogeneous (ML diferente por agente)
    cada com 30 execuções
```

---

### **ABORDAGEM B: Exploração Completa**
**Ficheiro**: [abordagem/abordagem_b.py](abordagem/abordagem_b.py)

#### Objetivo
Explorar o máximo do ambiente sem colidir com bombas.

#### Ambiente
- Grid 10x10
- **SEM Tesouros** (apenas bombas)
- **Bombas** (B): proporção configurável
- **Células livres** (L): o máximo possível

#### Estratégia
1. **BFS Colaborativo**: Expande fronteira segura a cada passo
2. **Detecção ML**: Aprende a predizer células com bombas
3. **Coordenação**: Evita que agentes se sobreponham

#### Métricas
| Métrica | Descrição |
|---------|-----------|
| `explored_percentage` | % do ambiente explorado |
| `agents_alive` | Quantos agentes sobrevivem até final |
| `cells_per_step` | Eficiência: células exploradas por passo |
| `safety_coverage_score` | Taxa de células exploradas com segurança |

#### Simulações
```
run_simulations_b.py
├── baseline (BFS puro, sem ML)
├── homogeneous (ML colaborativo)
└── heterogeneous (estratégias diferentes)
    cada com 30 execuções
```

---

### **ABORDAGEM C: Localização de Bandeira**
**Ficheiro**: [abordagem/abordagem_c.py](abordagem/abordagem_c.py)

#### Objetivo
Encontrar a bandeira desconhecida no menor número de passos.

#### Ambiente
- Grid 10x10
- **Bandeira** (F): posição desconhecida, colocada distante
- **Tesouros** (T): opcionais, podem ajudar
- **Bombas** (B): risco calculado
- **Caminho garantido**: de (0,0) até bandeira sem bombas

#### Estratégia
1. **A* Adaptativo**: Otimização colaborativa do caminho
2. **Heurística ML**: Predição de probabilidade de bandeira
3. **Balanceamento**: Custo de movimento vs risco de bomba

#### Métricas
| Métrica | Descrição |
|---------|-----------|
| `success` | Bandeira encontrada? (1=sim, 0=não) |
| `success_rate` | % de execuções com sucesso |
| `avg_steps_to_flag` | Passos médios até bandeira |
| `path_efficiency` | Passos reais / Caminho ótimo |

#### Simulações
```
run_simulations_c.py
├── baseline (A* puro, sem ML)
├── homogeneous (A* + ML colaborativo)
└── heterogeneous (estratégias diversas)
    cada com 30 execuções
```

---

## 🚀 Como Executar

### ✅ Pré-requisitos
```bash
pip install -r requirements.txt
```

**Dependências**:
- matplotlib (plots)
- pandas (análise dados)
- scikit-learn (ML: K-NN, Naive Bayes, Random Forest)

---

### **OPÇÃO 1: Executar via Scripts (Automático)**

#### Abordagem A - Coleta de Tesouros (90 simulações)
```bash
python run_simulations_a.py
```

Executa:
- ✅ 30 simulações baseline
- ✅ 30 simulações homogeneous  
- ✅ 30 simulações heterogeneous

**Duração**: ~2-3 minutos

#### Abordagem B - Exploração (90 simulações)
```bash
python run_simulations_b.py
```

**Duração**: ~2-3 minutos

#### Abordagem C - Bandeira (90 simulações)
```bash
python run_simulations_c.py
```

**Duração**: ~2-3 minutos

---

### **OPÇÃO 2: Executar via GUI (Interativa)**

```bash
python gui.py
```

#### Interface Gráfica oferece:
1. **Simulação Manual**
   - Escolher abordagem (A/B/C)
   - Config: nº agentes, proporção bombas, nº tesouros, max passos
   - Visualizar grid em tempo real
   - Ver métricas ao fim

2. **Batch Simulations** (botão na GUI)
   - Executar 30x3=90 simulações automaticamente
   - Guardar em background

3. **Comparação Visual**
   - Boxplot: distribuição de métricas
   - Scatter: relação entre 2 métricas
   - Linha: tendência de performance
   - Exportar gráficos para ficheiro

4. **Gestão de Dados**
   - Ver stats resumidas
   - Exportar para CSV
   - Limpar/resetar resultados

---

### **OPÇÃO 3: Análise de Dados**

#### Ver resumo dos resultados
```bash
python analise_dados.py
```

**Output**: Estatísticas por abordagem e grupo (média, desvio padrão, ranking)

#### Gerar relatório Markdown automático
```python
from relatorio import RelatorioGerador

gerador = RelatorioGerador()
relatorio = gerador.gerar_relatorio_completo()  
gerador.salvar_relatorio('RELATORIO_FINAL.md')
```

#### Exportar resultados para CSV
```python
from analise.comparative_analysis import DataStorage

storage = DataStorage()
storage.export_to_csv('A')  # Abordagem A
storage.export_to_csv()     # Todas as abordagens
```

---

## 📊 Fluxo de Dados e Resultados

### **1. Onde são guardados os resultados?**

#### Base de Dados Principal
```
simulation_results/
└── all_results.json
```

Estrutura:
```json
{
  "A": [
    {
      "timestamp": "2026-02-11T15:30:45.123456",
      "approach": "A",
      "group_type": "homogeneous",
      "parameters": {
        "num_agents": 4,
        "bomb_ratio": 0.3,
        "treasure_count": 12,
        "max_steps": 300,
        "homogeneous": true
      },
      "metrics": {
        "treasure_percentage": 83.33,
        "success_rate": 1.0,
        "exploration_efficiency": 0.0278,
        "reward_risk_ratio": 2.5,
        "avg_steps_to_treasure": 45.6
      }
    },
    ...
  ],
  "B": [...],
  "C": [...]
}
```

#### Inicializar JSON Vazio
```bash
python simulation_results/init_all_results.py
```

#### Limpar Resultados
```bash
python simulation_results/cleanup_results.py
```

---

### **2. Exportações em CSV**

Localização: `data/`

Nomes: `approach_A_20260211_090417.csv`, `all_20260211_090417.csv`

Colunas: timestamp, approach, group_type, param_*, metric_*

---

### **3. Como são Gerados os Dados**

```
┌──────────────────────────┐
│ run_simulations_X.py     │ (X = A, B ou C)
│ ou GUI Batch Simulations │
└───────────┬──────────────┘
            │
            ↓
┌──────────────────────────┐
│ ApproachXSimulation      │
│ BaselineX, EnvironmentX  │
│ (abordagem_x.py)         │
└───────────┬──────────────┘
            │
            ↓ (run/run_simulation, retorna metrics dict)
            
┌──────────────────────────┐
│ MetricsCalculator        │
│ .calculate_approach_x_   │
│  metrics(sim)            │
│ (comparative_analysis)   │
└───────────┬──────────────┘
            │
            ↓ (processado metrics dict)
            
┌──────────────────────────┐
│ DataStorage.save_result( │
│  approach, group_type,   │
│  metrics, parameters)    │
│ (comparative_analysis)   │
└───────────┬──────────────┘
            │
            ↓ (JSON with timestamp)

└────► all_results.json
```

---

## 📈 Sistemas de Análise

### **1. DataStorage** 
**Ficheiro**: `analise/comparative_analysis.py`

Responsabilidades:
- Carregar/salvar resultados JSON
- Filtrar por abordagem, grupo, timestamp
- Exportar para CSV
- Remover entradas antigas

**Métodos principais**:
```python
storage = DataStorage()

# Salvar resultado
storage.save_result(approach, group_type, metrics, parameters)

# Carregar resultados
results_a = storage.get_results_by_approach('A')

# Exportar CSV
filename = storage.export_to_csv('A')  

# Remover resultados antigos
removed_count = storage.remove_results(approach='A', keep_last=30)
```

---

### **2. MetricsCalculator**
**Ficheiro**: `analise/comparative_analysis.py`

Responsabilidades:
- Compilar dados brutos da simulação em métricas processadas
- Garantir consistência entre abordagens

**Métodos principais**:
```python
calc = MetricsCalculator()

# Abordagem A
metrics_a = calc.calculate_approach_a_metrics(sim)
# Retorna: treasure_percentage, success_rate, exploration_efficiency, etc.

# Abordagem B
metrics_b = calc.calculate_approach_b_metrics(sim)
# Retorna: explored_percentage, agents_alive, cells_per_step, etc.

# Abordagem C
metrics_c = calc.calculate_approach_c_metrics(sim)
# Retorna: success, success_rate, avg_steps_to_flag, path_efficiency
```

---

### **3. ComparativeAnalyzer**
**Ficheiro**: `analise/comparative_analysis.py`

Responsabilidades:
- Análise estatística (média, desvio, mediana)
- Testes estatísticos (significância)
- Agrupamento por parâmetros

**Métodos principais**:
```python
analyzer = ComparativeAnalyzer(storage)

# Estatísticas por grupo
stats = analyzer.compute_summary_statistics(approach='A')
# Retorna: {group: {metric: {mean, std, median, ...}}}

# Comparação pairwise
comparison = analyzer.pairwise_comparison(approach='A', metric='treasure_percentage')
# Retorna: tabela com p-values de testes

# Variação paramétrica
params_effect = analyzer.parameter_variation_analysis(approach='A')
```

---

### **4. ComparisonVisualizer**
**Ficheiro**: `analise/comparative_analysis.py`

Responsabilidades:
- Gerar gráficos comparativos
- Box plots, scatter plots, histogramas
- Integração com Matplotlib + Tkinter

**Tipos de gráficos**:

| Gráfico | Comando | Descrição |
|---------|---------|-----------|
| Box Plot | `plot_boxplot()` | Distribuição por grupo |
| Scatter | `plot_scatter()` | Relação 2 métricas |
| Histograma | `plot_histogram()` | Distribuição métrica |
| Linha | `plot_line()` | Tendência temporal |

---

### **5. GUI Integration** 
**Ficheiro**: `analise/gui_integration.py`

Responsabilidades:
- Integrar análise na GUI existente
- Auto-salvar simulações
- Abrir janela de comparação

**Classe**: `ComparativeExtension`

```python
# Na GUI
from analise.gui_integration import integrate_comparison_system

self.comparison_ext = integrate_comparison_system(self)

# Métodos disponíveis
self.comparison_ext.save_current_simulation()
self.comparison_ext.open_comparison_window()
self.comparison_ext.open_simulation_runner()
```

---

## 📄 Geração de Relatórios

### **1. Relatório Automático (Markdown)**
**Ficheiro**: [relatorio.py](relatorio.py)

O script `RelatorioGerador` lê `all_results.json` e gera análise em Markdown.

#### Responde a 6 perguntas por abordagem:

**ABORDAGEM A - Coleta de Tesouros**
1. Qual é a taxa média de sucesso em cada grupo?
2. Qual grupo recolhe mais tesouros em média?
3. Qual é a eficiência de exploração de cada grupo?
4. Qual é a razão recompensa-risco de cada grupo?
5. Qual grupo é mais rápido em encontrar tesouros?
6. Há diferença significativa entre grupos?

**ABORDAGEM B - Exploração**
1. Qual é a percentagem média explorada por grupo?
2. Qual grupo mantém mais agentes vivos?
3. Qual é a velocidade de exploração (células/passo)?
4. Qual é a cobertura segura de cada grupo?
5. Qual grupo é mais eficiente globalmente?
6. Qual estratégia é mais robusta?

**ABORDAGEM C - Bandeira**
1. Qual é a taxa de sucesso em localizar bandeira?
2. Qual grupo é mais rápido?
3. Qual é a eficiência do caminho?
4. Há diferença em usar tesouros como pistas?
5. Qual estratégia é mais confiável?
6. Ranking geral de desempenho

#### Usar
```python
from relatorio import RelatorioGerador

gerador = RelatorioGerador('simulation_results/all_results.json')
md_text = gerador.gerar_relatorio_completo()
gerador.salvar_relatorio('RELATORIO_FINAL.md')
```

---

### **2. Análise Exploratória**
**Ficheiro**: [analise_dados.py](analise_dados.py)

Script simples que:
- Lê all_results.json
- Calcula média e desvio padrão por grupo
- Mostra ranking de abordagens
- Analisa variação com nº de agentes

```bash
python analise_dados.py
```

**Output padrão**:
```
========================================
ANÁLISE DETALHADA POR ABORDAGEM
========================================

--- ABORDAGEM A: COLETA DE TESOUROS ---

BASELINE:
  Tesouro coletado: 75.45% (±8.23)
  Taxa sucesso: 85.00% (±10.12)
  Eficiência exploração: 0.0245 (±0.0034)
  Razão recompensa-risco: 2.34 (±0.67)

HOMOGENEOUS:
  ...

HETEROGENEOUS:
  ...

--- ABORDAGEM B: EXPLORAÇÃO COMPLETA ---
...

--- ABORDAGEM C: LOCALIZAÇÃO DE BANDEIRA ---
...

========================================
RANKING GERAL (melhor grupo por métrica)
========================================

Abordagem A - Melhor em coleta de tesouro:
  heterogeneous: 92.34%

Abordagem B - Melhor em exploração:
  homogeneous: 76.89%

Abordagem C - Melhor em encontrar bandeira:
  baseline: 98.34%
```

---

### **3. Pós-processamento**
**Ficheiro**: [postprocess_all_results.py](postprocess_all_results.py)

Funções utilitárias:
- Remover outliers
- Agregar dados por critérios
- Interpolação/normalização
- Geração de resumos

---

## 🔍 Casos de Uso Prácticos

### **Cenário 1: Rodar novo experimento completo**

```bash
# 1. Limpar resultados antigos
python simulation_results/cleanup_results.py

# 2. Executar todos as simulações
python run_simulations_a.py
python run_simulations_b.py
python run_simulations_c.py

# 3. Gerar relatório
python -c "from relatorio import RelatorioGerador; g = RelatorioGerador(); g.salvar_relatorio('RELATORIO_NOVO.md')"

# 4. Ver resultados na GUI
python gui.py
```

---

### **Cenário 2: Comparar apenas 2 grupos em Abordagem A**

```python
from analise.comparative_analysis import DataStorage, ComparativeAnalyzer

storage = DataStorage()
analyzer = ComparativeAnalyzer(storage)

# Filtrar Abordagem A
results_a = storage.get_results_by_approach('A')

# Separar homogeneous vs heterogeneous
homo = [r for r in results_a if r['group_type'] == 'homogeneous']
hetero = [r for r in results_a if r['group_type'] == 'heterogeneous']

# Extrair métrica
treasure_homo = [r['metrics']['treasure_percentage'] for r in homo]
treasure_hetero = [r['metrics']['treasure_percentage'] for r in hetero]

print(f"Homogeneous: {np.mean(treasure_homo):.2f}%")
print(f"Heterogeneous: {np.mean(treasure_hetero):.2f}%")
```

---

### **Cenário 3: Exportar dados para análise externa**

```bash
# Exportar Abordagem A para CSV
python -c "from analise.comparative_analysis import DataStorage; s = DataStorage(); s.export_to_csv('A')"

# CSV criado em: data/approach_A_20260211_*.csv
# Pode importar em Excel, R, Python para análise customizada
```

---

## 📝 Notas Importantes

### **Garantias de Resolubilidade**

✅ **Abordagem A**: 
- Todos os tesouros são alcançáveis (BFS verifica)
- Células livres ≥ 20% do grid

✅ **Abordagem B**:
- (0,0) nunca fica encurralado
- ≥ 50% das células livres acessíveis de (0,0)

✅ **Abordagem C**:
- Caminho garantido de (0,0) até bandeira SEM bombas
- Bandeira sempre alcançável

---

### **Configuração de Parâmetros**

| Parâmetro | Padrão | Mín | Máx | Descrição |
|-----------|--------|-----|-----|-----------|
| num_agents | 4 | 1 | 10 | Nº de agentes |
| bomb_ratio | 0.3 | 0.2 | 0.8 | Proporção de bombas |
| treasure_count | 12 | 1 | 20 | Nº de tesouros |
| max_steps | 300 | 50 | 1000 | Máximo passos simulação |
| grid_size | 10 | 5 | 20 | Tamanho grid (NxN) |

---

### **Modelos ML Utilizados**

- **K-Nearest Neighbors** (K-NN): Classificação por vizinhança
- **Naive Bayes Gaussiano**: Classificação probabilística
- **Random Forest**: Ensemble de árvores de decisão

Cada agente pode usar modelo diferente (heterogeneous) ou igual (homogeneous).

---

## 🎓 Conclusão

Este projeto oferece:
- ✅ 3 abordagens diferentes de IA colaborativa
- ✅ Sistema completo de simulação e análise
- ✅ Interface gráfica interativa
- ✅ Armazenamento estructurado de resultados
- ✅ Geração automática de relatórios
- ✅ Visualizações comparativas avançadas

Tudo integrado num pipeline de investigação robustus e reproduzível! 🚀
