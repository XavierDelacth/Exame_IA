#  Guia Rápido de Execução

##  Ficheiros Executáveis do Projeto

Este guia mostra como executar os 5 ficheiros principais do projeto e o que cada um faz.

---

## 1️⃣ **gui.py** - Interface Gráfica Interativa

### Como Executar
```bash
python gui.py
```

### O que faz 
Interface gráfica Tkinter que permite executar simulações manualmente com parâmetros customizados, visualizar grids em tempo real e fazer análises comparativas com gráficos (boxplot, scatter, linha).
Também oferece botão para executar batch simulations (30 sims automaticamente) e exportar dados para CSV/PNG.

---

## 2️⃣ **run_simulations_a.py** - Abordagem A (Coleta de Tesouros)

### Como Executar
```bash
python run_simulations_a.py
```

### O que faz 
Executa automaticamente 90 simulações da Abordagem A (30 baseline + 30 homogeneous + 30 heterogeneous) onde agentes competem para recolher o máximo de tesouros evitando bombas.
Todos os resultados são salvos em `simulation_results/all_results.json` com métricas de sucesso, eficiência e recompensa-risco.

### Tempo Estimado
⏱️ **~2-3 minutos** para 90 simulações

### Output Esperado
```
[A] Saved run 1/30 for group baseline
[A] Saved run 2/30 for group baseline
...
[A] Saved run 30/30 for group heterogeneous
```

---

## 3️⃣ **run_simulations_b.py** - Abordagem B (Exploração Completa)

### Como Executar
```bash
python run_simulations_b.py
```

### O que faz 
Executa automaticamente 90 simulações da Abordagem B (30 baseline + 30 homogeneous + 30 heterogeneous) onde agentes usam BFS colaborativo para explorar o máximo do ambiente sem bater em bombas.
Todos os resultados são salvos em `simulation_results/all_results.json` com métricas de exploração, agentes vivos e eficiência por passo.

### Tempo Estimado
   **~2-3 minutos** para 90 simulações

### Output Esperado
```
[B] Saved run 1/30 for group baseline
[B] Saved run 2/30 for group baseline
...
[B] Saved run 30/30 for group heterogeneous
```

---

## 4️⃣ **run_simulations_c.py** - Abordagem C (Localização de Bandeira)

### Como Executar
```bash
python run_simulations_c.py
```

### O que faz 
Executa automaticamente 90 simulações da Abordagem C (30 baseline + 30 homogeneous + 30 heterogeneous) onde agentes precisam localizar uma bandeira desconhecida usando A* adaptativo e ML colaborativo.
Todos os resultados são salvos em `simulation_results/all_results.json` com métricas de sucesso, velocidade até bandeira e eficiência do caminho.

### Tempo Estimado
 **~2-3 minutos** para 90 simulações

### Output Esperado
```
[C] Saved run 1/30 for group baseline
[C] Saved run 2/30 for group baseline
...
[C] Saved run 30/30 for group heterogeneous
```

---

## 5️⃣ **postprocess_all_results.py** - Pós-Processamento de Dados

### Como Executar
```bash
python postprocess_all_results.py
```

### O que faz 
Processa dados brutos do `all_results.json` para validação, limpeza de duplicatas, remoção de outliers e normalização de métricas.
Gera um relatório de pós-processamento e salva dados limpos em `all_results_processed.json` para análise posterior.

### Tempo Estimado
 **~2-5 segundos** (depende do volume de dados)

### Output Esperado
```
POSTPROCESSAMENTO INICIADO
====================================
 Dados carregados: 270 registros

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

====================================
 Postprocessamento completo!
```

---

##  Ficheiros Relacionados (Execução Secundária)

### **relatorio.py** - Gerador de Relatórios
```bash
python relatorio.py
```
Gera um relatório markdown completo com análise estatística de todos os dados em `all_results.json`.
Salva em `RELATORIO_ANALISE.md` com tabelas, conclusões e rankings.

### **analise_dados.py** - Análise Exploratória Rápida
```bash
python analise_dados.py
```
Script rápido que mostra estatísticas básicas (média, desvio padrão) de todos os dados em consola.

---

##  Fluxo Recomendado Completo

### **Para Executar Tudo do Zero:**

```bash
# 1. Limpar dados antigos (opcional)
python simulation_results/cleanup_results.py

# 2. Executar as 3 abordagens (30 min total de espera)
python run_simulations_a.py
python run_simulations_b.py
python run_simulations_c.py

# 3. Processar dados
python postprocess_all_results.py

# 4. Gerar relatório
python relatorio.py

# 5. Analisar na interface gráfica
python gui.py
```

**Tempo Total**: ~15 minutos (incluindo 9 min de simulações)

---

### **Para Análise Rápida (Dados Já Existem):**

```bash
# 1. Ver análise rápida em consola
python analise_dados.py

# 2. Gerar relatório markdown
python relatorio.py

# 3. Explorar visualmente na GUI
python gui.py
```

**Tempo Total**: ~10 segundos

---

### **Para Rodar Uma Única Abordagem:**

```bash
# Só Abordagem A
python run_simulations_a.py

# Analisar na GUI
python gui.py
```

**Tempo Total**: ~3 minutos

---

##  Configuração (Opcional)

### Modificar Parâmetros nos Scripts

Se quiser alterar parâmetros, edite o ficheiro correspondente:

#### **run_simulations_a.py**
```python
def main(runs_per_group=30, num_agents=4, bomb_ratio=0.3, treasure_count=12, max_steps=300):
    # Altere os valores por defeito aqui
```

#### **run_simulations_b.py**
```python
def main(runs_per_group=30, num_agents=4, bomb_ratio=0.3, max_steps=300):
    # Altere os valores por defeito aqui
```

#### **run_simulations_c.py**
```python
def main(runs_per_group=30, num_agents=4, bomb_ratio=0.3, treasure_count=10, max_steps=300):
    # Altere os valores por defeito aqui
```

---

##  Pré-requisitos

Antes de executar qualquer ficheiro, certifique-se que tem dependências instaladas:

```bash
pip install -r requirements.txt
```

**Dependências Necessárias:**
- `matplotlib` - para gráficos
- `pandas` - para análise de dados
- `scikit-learn` - para modelos ML (K-NN, Naive Bayes, Random Forest)

---

## Ficheiros de Entrada/Saída

| Script                         | Entrada            | Saída                                    |
|--------------------------------|--------------------|------------------------------------------|
| **gui.py**                     | Interativo         | Gráficos PNG (opcional)                  |
| **run_simulations_a.py**       | Nenhuma            | `simulation_results/all_results.json`    |
| **run_simulations_b.py**       | Nenhuma            | `simulation_results/all_results.json`    |
| **run_simulations_c.py**       | Nenhuma            | `simulation_results/all_results.json`    |
| **postprocess_all_results.py** | `all_results.json` | `all_results_processed.json` + relatório |
| **relatorio.py**               | `all_results.json` | `RELATORIO_ANALISE.md`                   |
| **analise_dados.py**           | `all_results.json` | Consola (texto)                          |

---

## 🆘 Troubleshooting

### **Erro: "ModuleNotFoundError: No module named 'matplotlib'"**
```bash
pip install matplotlib pandas scikit-learn
```

### **Erro: "FileNotFoundError: simulation_results/all_results.json"**
```bash
# Inicializar JSON vazio primeiro
python simulation_results/init_all_results.py
```

### **GUI não abre ou tem erro Tkinter**
Em Windows:
```bash
pip install tk
```

Em Linux:
```bash
sudo apt-get install python3-tk
```

### **Simulações muito lentas**
- Reduza `runs_per_group` nos scripts (de 30 para 10)
- Reduza `max_steps` (de 300 para 100)
- Reduza `num_agents` (de 4 para 2)



