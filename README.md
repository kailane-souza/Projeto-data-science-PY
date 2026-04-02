# Projeto-data-science-PY

1.1. Objetivo
O objetivo deste projeto é desenvolver um modelo preditivo capaz de estimar o desempenho acadêmico de estudantes do ensino médio, identificando antecipadamente aqueles com maior risco de reprovação. A importância está em permitir que instituições de ensino atuem de forma preventiva, direcionando suporte pedagógico a quem mais precisa antes que o problema se agrave.

1.2. Fontes de Dados
O dataset utilizado é o Student Performance Dataset, disponibilizado pelo UCI Machine Learning Repository. Ele contém dados de 395 estudantes de escolas públicas de Portugal, organizados em três grupos de variáveis:
Dados Acadêmicos: notas dos períodos anteriores (G1 e G2), nota final (G3), número de reprovações passadas, horas de estudo semanais e frequência às aulas.
Dados Socioeconômicos: escolaridade dos pais, tipo de emprego dos responsáveis e acesso a recursos educativos como internet e aulas de reforço.
Dados Comportamentais: consumo de álcool, tempo livre, atividades extracurriculares e relacionamentos afetivos.

1.3. Métodos Planejados
O projeto seguiu um pipeline estruturado em quatro fases:
Análise Exploratória de Dados (EDA): estatísticas descritivas, análise de correlação entre variáveis e geração de gráficos para identificar padrões relevantes.
Pré-processamento: tratamento de valores ausentes, codificação de variáveis categóricas e normalização dos dados numéricos com StandardScaler.
Modelagem Preditiva: treinamento e comparação de três algoritmos — Regressão Logística, Árvore de Decisão e Random Forest — utilizando o Scikit-learn.
Avaliação: uso de métricas como Acurácia, F1-Score, Precisão, Recall e Matriz de Confusão para selecionar o modelo com melhor desempenho.

1.4. Impacto Esperado
Espera-se que os resultados permitam identificar os fatores que mais influenciam o desempenho dos estudantes, gerando insumos concretos para o desenvolvimento de estratégias de intervenção educacional baseadas em dados — contribuindo para a redução dos índices de reprovação e evasão escolar.
