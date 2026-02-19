# Rascunho de Artigo — Exploração Autônoma com Risco

## Resumo
Este trabalho investiga estratégias de exploração autônoma em ambientes discretos com perigos latentes (p.ex. minas/bombas). Implementámos três abordagens (A, B e C) com ambientes balanceados e memórias partilhadas entre agentes. Realizámos experimentos Monte‑Carlo (30 execuções por grupo) para três configurações de equipa (homogeneous, heterogeneous, baseline) e medimos cobertura, eficiência de exploração e métricas de segurança (bombs triggered, survival rate). Apresentamos comparação entre políticas, análise estatística e propostas para trabalho futuro.

## Introdução
- Contexto: robótica autónoma e exploração em ambientes perigosos; importância de trade‑off entre descoberta e segurança.
- Motivação: explorar de forma eficiente enquanto minimizamos exposição a riscos.
- Contribuições: (i) implementações de três abordagens com balanceamento automático do ambiente; (ii) pipeline de experimentos com métricas padronizadas; (iii) análise comparativa das políticas.

## Trabalhos relacionados
- Frontier-based exploration; Probabilistic Robotics (Thrun et al.).
- Safe Reinforcement Learning e Constrained MDPs.
- Técnicas clássicas: BFS/A* e potential fields.

## Metodologia

Ambientes e abordagens
- Abordagem A: ambiente com tesouros distribuídos, zonas seguras criadas em torno de cada tesouro; agentes procuram maximizar recolha de tesouros. Ambiente garantido balanceado (bomb_ratio entre 20% e 80%, células livres >=20%).
- Abordagem B: exploração pura (sem tesouros), constraints para garantir que pelo menos 50% das células livres são acessíveis a partir de (0,0); agentes usam BFS/heurísticas seguras.
- Abordagem C: objetivo é encontrar uma bandeira desconhecida; cria-se um caminho garantido (sem bombas) da origem até à bandeira e agentes usam variantes A*/otimização de caminho.

Memória partilhada e perceção
- Cada abordagem define uma estrutura de `SharedMemory` para partilhar conhecimento sobre células (explorado, seguro, bomba identificada, risco estimado).
- Estimativas de risco locais são atualizadas quando bombas são descobertas; células não exploradas podem ser marcadas como inseguras até prova em contrário (abordagem conservadora).

Políticas de agentes
- Baselines: BFS puro (B), A* (C) e variantes simples (A).
- Políticas colaborativas: homogeneous (mesma política por agente) e heterogeneous (mix de políticas).

Métricas calculadas
- Abordagem A: `treasure_percentage`, `treasures_per_second`, `bombs_triggered`, `exploration_efficiency`, `explored_percentage`.
- Abordagem B: `explored_percentage`, `safe_exploration_rate`, `survival_rate`, `cells_per_second`.
- Abordagem C: `avg_steps_to_flag`, `path_efficiency`, `bombs_triggered`, `success_rate`, `overall_score`.

## Experimentos
- Configuração padrão nos scripts: 30 runs por grupo; `num_agents=4`; `bomb_ratio=0.3`; `max_steps=300`; `treasure_count` variando por abordagem (A:12, C:10).
- Grupos testados: `homogeneous`, `heterogeneous`, `baseline`.
- Armazenamento e pós‑processamento: resultados agregados em `simulation_results/all_results.json` e processados por `analise/comparative_analysis.py`.

Protocolo experimental
- Repetições: 30 runs por combinação (permite estimar média e desvio‑padrão).
- Medidas: coletar métricas por run, depois calcular média, std, min/max e usar testes estatísticos quando necessário.

## Resultados (descrição e interpretação)
- Observação geral: existe trade‑off entre cobertura e exposição ao risco; políticas mais agressivas tendem a recolher mais tesouros/mais cobertura mas com maiores `bombs_triggered`.
- Como ler os resultados: usar `DataStorage.export_to_csv()` para obter tabelas por abordagem e grupo; usar `ComparativeAnalyzer` para sumarizar médias e identificar o melhor grupo segundo a métrica principal (p.ex. `treasure_percentage` para A, `explored_percentage` para B, `success_rate` para C).
- Exemplos de insights que podem ser extraídos (a partir dos ficheiros de resultados):
  - Diferença média de `treasure_percentage` entre `homogeneous` e `heterogeneous` em A.
  - Taxa média de `bombs_triggered` por agente e por run; correlação negativa entre `survival_rate` e `explored_percentage`.
  - Eficiência de caminho em C: `path_efficiency` e `avg_steps_to_flag` mostram quanto as políticas se aproximam do ótimo criado pelo caminho garantido.

## Discussão
- Interpretação das métricas: discutir se ganhos de cobertura justificam maior risco.
- Limitações: simulações usam sensores ideais / lógica determinística em algumas partes; ambiente modelado (grid) não captura complexidades físicas reais.
- Riscos e ética: reforçar que os resultados são simulações e não instruções para operações reais com dispositivos perigosos; considerar aspetos de segurança humana e uso responsável.

## Conclusão e trabalho futuro
- Resumo: incorporar modelos probabilísticos de risco e penalizações explícitas na função‑objetivo melhora segurança sem sacrificar totalmente eficiência (observação qualitativa dos nossos resultados).
- Futuro: validar com modelos de sensor ruidosos, incluir aprendizagem online para estimativa de risco, testar políticas multi‑agente com divisão explícita de risco, explorar abordagens baseadas em RL com restrições (Constrained MDPs).

## Recomendações práticas para o relatório
- Incluir tabelas com médias±std para cada métrica e grupo.
- Traçar boxplots por métrica (A,B,C) divididos por grupo.
- Mostrar mapas de exemplo (mapa de risco antes/depois) e trajetórias representativas.

## Referências sugeridas
- Thrun, S., Burgard, W., Fox, D. — Probabilistic Robotics (2005).
- Yamauchi, B. — A frontier-based approach for autonomous exploration (1997).
- Altman, E. — Constrained Markov Decision Processes (1999).
- Mihatsch, O., Neuneier, R. — Risk-sensitive reinforcement learning (2002).

---
Arquivo gerado automaticamente a partir do repositório; para incorporar valores numéricos reais execuções: exporte `all_results.json` para CSV e atualize a secção `Resultados` com médias e gráficos.
