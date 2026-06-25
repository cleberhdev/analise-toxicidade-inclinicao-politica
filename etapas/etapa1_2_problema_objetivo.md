# Etapas 1 e 2 — Definição do Problema e Objetivo da Solução

Projeto: Análise computacional comparada da linguagem política brasileira em dois registros — toxicidade no discurso cidadão (tweets) e sinal ideológico no discurso institucional (discursos parlamentares). Deep Learning para demandas reais.

## Etapa 1 — Definição do Problema

### 1.1 Contexto e motivação
A esfera pública brasileira se expressa em registros linguísticos muito diferentes. De um lado, o **cidadão comum** em redes sociais — texto curto, informal, com gírias, ironia e, com frequência, toxicidade. De outro, o **agente institucional** (o parlamentar) — texto longo, formal, protocolar, mas carregado de sinal ideológico. Compreender automaticamente esses sinais de comportamento político tem demanda real: moderação de plataformas, observatórios de discurso de ódio, análise de polarização, transparência legislativa e pesquisa em ciência política computacional.

O problema de fundo é que ferramentas de Processamento de Linguagem Natural costumam ser desenvolvidas e avaliadas dentro de **um único registro**, e silenciosamente assume-se que o que funciona em um vale para o outro. Essa suposição raramente é testada — e é justamente o ponto cego que este projeto ataca.

### 1.2 O problema concreto
O sistema precisa resolver dois problemas de **classificação de texto** em português brasileiro, cada um em seu registro, e em seguida investigar a relação entre eles:

- **Tarefa T1 — Toxicidade (registro cidadão/informal).** Dado um tweet, decidir se ele é tóxico. Função alvo: `f1: texto → {tóxico, não-tóxico}`.
- **Tarefa T2 — Inclinação ideológica (registro institucional/formal).** Dado um discurso parlamentar, decidir a orientação ideológica do autor. Função alvo: `f2: texto → {esquerda, direita}`.

Ambas são formuladas como **classificação binária supervisionada**, escolha que maximiza a saúde estatística das classes (ver Etapa 3) e a clareza de avaliação.

### 1.3 Pergunta de pesquisa central
Mais do que entregar dois classificadores, o projeto responde a uma pergunta que conecta as duas tarefas:

> Representações de linguagem política aprendidas em um registro **transferem** para o outro? Um modelo de Deep Learning com parâmetros compartilhados entre os dois registros **ajuda** (transferência positiva) ou **atrapalha** (*negative transfer*)?

Essa pergunta transforma a heterogeneidade das bases — que seria uma fragilidade num projeto mal estruturado — no próprio objeto de estudo.

### 1.4 Por que é um problema de Deep Learning
Três propriedades do problema justificam Deep Learning, em vez de abordagens rasas (bag-of-words + classificador linear):

A toxicidade depende fortemente de **contexto, ironia e construções não literais** ("que coisa linda, hein"), que modelos lineares de superfície capturam mal; representações contextuais (Transformers) são o estado da arte exatamente por modelar essas dependências. A inclinação ideológica em discurso formal é um sinal **difuso e de longo alcance** — não está numa palavra-chave, mas no enquadramento, no léxico e nas correferências ao longo de parágrafos —, o que demanda codificadores capazes de representar contexto amplo. E a **supervisão distante** do rótulo político (derivado do partido) introduz ruído que modelos com representações ricas e regularização absorvem melhor do que classificadores frágeis. Por fim, a própria pergunta de transferência só é formulável em termos de **representações compartilhadas**, um conceito nativo de Deep Learning (encoders pré-treinados, multi-task learning).

### 1.5 Escopo e não-escopo
Está **no escopo**: classificação binária nas duas tarefas; uso de modelos de linguagem pré-treinados em português; avaliação intra-registro e teste de transferência entre registros; uma demonstração funcional. Está **fora do escopo**: detecção fina dos seis subtipos de toxicidade (tratada apenas como extensão exploratória), predição de postura sobre alvos específicos (inviabilizada pela indisponibilidade textual do UStanceBR — ver Etapa 3), análise multimodal ou de rede social, e qualquer pretensão de inferir ideologia de cidadãos a partir de tweets (não temos rótulo para isso, e seria eticamente sensível).

## Etapa 2 — Objetivo da Solução

### 2.1 Objetivo geral
Desenvolver, documentar e demonstrar uma solução de Deep Learning, em português brasileiro, capaz de (i) classificar toxicidade em discurso cidadão e (ii) classificar inclinação ideológica em discurso institucional, e (iii) **medir empiricamente** se há transferência de representações entre esses dois registros.

### 2.2 Objetivos específicos
1. Construir, de forma reprodutível, as duas bases textuais abertas (ToLD-Br e discursos da Câmara), com pipeline de pré-processamento comum e *splits* estratificados.
2. Treinar modelos *baseline* fortes e independentes para cada tarefa, usando codificadores monolíngues em português (BERTimbau e/ou BERTabaporu), com seleção de hiperparâmetros documentada.
3. Implementar uma arquitetura de **Multi-task Learning** com encoder compartilhado e duas cabeças, treinada sobre as bases disjuntas, como teste explícito da hipótese de transferência entre registros.
4. Avaliar todos os modelos com protocolo rigoroso (métrica primária macro-F1, validação, análise de erros) e comparar *baselines* independentes vs. multi-tarefa.
5. Entregar uma **demonstração funcional** que receba um texto e retorne as predições apropriadas, com indicação de confiança.
6. Documentar decisões, limitações e ameaças à validade ao longo de todas as etapas.

### 2.3 Hipóteses
- **H1 — Monolíngue vence multilíngue.** Codificadores pré-treinados em português superarão alternativas multilíngues genéricas nas duas tarefas (consistente com o próprio paper do ToLD-Br).
- **H2 — Transferência entre registros é fraca ou negativa.** Por causa do contraste de registro (informal/curto vs. formal/longo), o encoder compartilhado **não** superará — e possivelmente piorará — os *baselines* independentes. Confirmar ou refutar H2 é o resultado científico central do projeto.
- **H3 — Aplicação cruzada falha.** Um modelo treinado em um registro terá desempenho ruim quando aplicado diretamente ao outro, evidenciando que "linguagem política" não é um sinal homogêneo entre domínios.

### 2.4 Critérios de sucesso
O sucesso **não** se define apenas por acurácia alta. Define-se por: (a) *baselines* por tarefa com macro-F1 competitivo frente à literatura (referência ToLD-Br: ~0,76 macro-F1 no binário); (b) uma comparação *baseline* vs. multi-tarefa metodologicamente limpa, com conclusão clara sobre H2 — **inclusive se a conclusão for negativa**; (c) reprodutibilidade ponta a ponta (dados, código, sementes, versões); e (d) uma demonstração que funcione sobre texto novo. Um resultado negativo bem fundamentado (H2/H3 confirmadas) é considerado sucesso pleno, pois responde à pergunta de pesquisa.

### 2.5 Entregáveis
Pipeline de dados reprodutível; modelos treinados (baselines + multi-tarefa); relatório de avaliação com análise de erros; demonstração funcional; e documentação técnica completa das sete etapas.

---

### Fontes
- ToLD-Br: Leite, J. A. et al. (2020). *Toxic Language Detection in Social Media for Brazilian Portuguese*. AACL-IJCNLP. https://github.com/JAugusto97/ToLD-Br
- API de dados abertos da Câmara dos Deputados: https://dadosabertos.camara.leg.br
- Classificação ideológica de partidos: Bolognesi, B.; Ribeiro, E.; Codato, A. (2023). *A New Measure of Party Ideology in Brazil*.
