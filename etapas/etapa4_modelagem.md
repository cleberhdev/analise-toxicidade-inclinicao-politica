# Etapa 4 — Modelagem em Deep Learning

Projeto: Análise computacional comparada da linguagem política brasileira em dois registros. Esta etapa define as arquiteturas, os codificadores, os hiperparâmetros e o protocolo experimental que respondem à pergunta de pesquisa (transferência entre registros — H2/H3 da Etapa 2).

## 4.1 Estratégia geral de modelagem

A modelagem segue três blocos, em ordem crescente de ambição, para que cada conclusão se apoie na anterior:

1. **Baseline raso (referência mínima).** TF-IDF + Regressão Logística por tarefa. Serve para quantificar *quanto* o Deep Learning realmente agrega — sem essa referência, ganhos de DL não têm contexto.
2. **Baselines profundos independentes (modelo principal).** Um Transformer pré-treinado *fine-tunado* por tarefa, cada um com o codificador mais adequado ao seu registro. É o coração do sistema.
3. **Multi-task com encoder compartilhado (o experimento).** Um único codificador com duas cabeças, treinado sobre as duas bases, para testar diretamente a hipótese de transferência (H2).

A justificativa dessa escalada é metodológica: ela transforma "treinar um modelo" em "demonstrar, com controles, por que este modelo e não outro".

## 4.2 Por que Transformers (e não LSTM/CNN)

Consideramos três famílias. CNNs sobre embeddings capturam bem *n-gramas* locais, mas a toxicidade frequentemente depende de ironia e contexto não local ("nossa, que pessoa *maravilhosa*…"), que janelas convolucionais fixas modelam mal. LSTMs/BiLSTMs modelam sequência, mas degradam em dependências muito longas — exatamente o caso do discurso parlamentar, onde o sinal ideológico é difuso ao longo de parágrafos. **Transformers pré-treinados** resolvem os dois pontos: atenção global captura dependências de longo alcance, e o pré-treino em grandes corpora em português injeta conhecimento linguístico que nossas bases (dezenas de milhares de exemplos) não ensinariam do zero. Para o tamanho de dados que temos, *fine-tuning* de um modelo pré-treinado é a escolha de melhor custo-benefício frente a treinar uma rede do zero.

## 4.3 Codificadores: escolha por registro

A decisão mais importante da etapa, e a que justifica os dois modelos separados, é **casar o codificador ao registro** de cada tarefa:

| Tarefa | Registro | Codificador | Pré-treino | Por quê |
|---|---|---|---|---|
| T1 — Toxicidade | Tweets, informal | **BERTabaporu** (`pablocosta/bertabaporu-base-uncased`) | ~237 mi de tweets em PT | Domínio idêntico ao da base; aprende gíria, grafia de rede social, emoji |
| T2 — Inclinação | Discurso parlamentar, formal | **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) | BrWaC (texto web formal) | Domínio formal e *cased* preserva nomes próprios/siglas, relevantes em discurso político |

Esse alinhamento codificador↔registro é também o argumento técnico central do projeto: **não existe um único codificador ótimo para os dois registros**. É por isso que esperamos que o encoder compartilhado (4.6) não vença os baselines especializados — a hipótese H2 nasce desta tabela.

Como controle de H1 (monolíngue > multilíngue), cada tarefa será também treinada com mBERT (`bert-base-multilingual-cased`) para comparação.

## 4.4 Arquitetura dos baselines profundos

Arquitetura padrão de *fine-tuning* para classificação de sequência: codificador Transformer → representação do token `[CLS]` (pooling do primeiro token) → `dropout` (0,1) → camada linear de classificação → `softmax`. Treina-se **o codificador inteiro** (full fine-tuning), não apenas a cabeça: com dezenas de milhares de exemplos rotulados, liberar todas as camadas supera o *feature-extraction* congelado, pois permite ao modelo adaptar as representações ao domínio e à tarefa.

**Tratamento de texto longo (T2).** Tweets cabem folgadamente no limite de 512 *tokens* do BERT; discursos não. Como o sinal ideológico é difuso, truncar perde informação. Estratégia principal: **janelamento** — fragmentar o discurso em janelas de 512 *tokens* (com sobreposição), classificar cada janela e **agregar** por média das probabilidades (ou voto). Estratégia *fallback* (mais barata): truncamento **cabeça+cauda** (primeiros 256 + últimos 256 *tokens*), que preserva abertura e conclusão do discurso. A escolha entre as duas será decidida empiricamente na Etapa 5; documentamos ambas para reprodutibilidade.

## 4.5 Função de perda e desbalanceamento

T1 (toxicidade) tem ~19% de positivos; T2 (inclinação) tem desbalanceamento por volume de discursos entre partidos. Em ambos os casos usa-se **entropia cruzada com pesos de classe** inversamente proporcionais à frequência (`class_weight balanced`), para que a classe minoritária não seja ignorada. A métrica primária de avaliação é **macro-F1**, coerente com a perda ponderada e robusta a desbalanceamento (a Etapa 5 detalha). Como alternativa para T1, registra-se a opção de *focal loss*, útil quando há muitos exemplos fáceis — a ser testada apenas se o macro-F1 da classe positiva ficar aquém.

## 4.6 Arquitetura Multi-task (o experimento de transferência)

O modelo multi-tarefa usa **hard parameter sharing**: um único codificador compartilhado e **duas cabeças** independentes (uma por tarefa). Como **não há corpus rotulado conjuntamente** (nenhum texto tem rótulo de toxicidade *e* de ideologia — limitação herdada das bases disjuntas), o treino é feito por **batches alternados em round-robin**: cada *batch* vem de uma base, atravessa o codificador e atualiza apenas a cabeça correspondente, com **mascaramento da perda** da outra tarefa.

Decisões e justificativas:

- **Codificador compartilhado:** adota-se o BERTimbau como *backbone* compartilhado por ser o mais geral entre os dois registros (e, opcionalmente, repete-se com BERTabaporu para checar sensibilidade). Por definição, esse codificador único não pode ser ótimo para ambos os registros — esse é o mecanismo que pode gerar *negative transfer*.
- **Amostragem de tarefas:** como as bases têm tamanhos diferentes, usa-se amostragem proporcional com **temperatura** (suavizando a base maior) para evitar que uma tarefa domine o gradiente.
- **Pesos de perda por tarefa:** escalares ajustáveis (λ₁, λ₂) para balancear as magnitudes das perdas das duas cabeças.

Este bloco existe para **testar H2**, não para necessariamente vencer. Se ele empatar ou perder para os baselines de 4.4, isso é um resultado válido e esperado, e será reportado como tal.

## 4.7 Protocolo do teste de transferência entre registros (H3)

Além do multi-task, executa-se um teste direto de transferência *zero-shot* entre registros: treinar o classificador no registro A e avaliá-lo no registro B (e vice-versa), naquilo que é comparável. Como os rótulos diferem (toxicidade vs. ideologia), a transferência é avaliada onde faz sentido — por exemplo, aplicando o modelo de toxicidade a discursos para medir queda de desempenho/ativação, e analisando qualitativamente. O objetivo é evidenciar (H3) que "linguagem política" não é um sinal homogêneo entre domínios. Resultados alimentam a discussão da Etapa 5 e 6.

## 4.8 Hiperparâmetros (ponto de partida, com justificativa)

Valores iniciais seguem as boas práticas de *fine-tuning* de BERT (Devlin et al.); o ajuste fino vem por busca em grade documentada na Etapa 5.

| Hiperparâmetro | T1 (toxicidade) | T2 (inclinação) | Justificativa |
|---|---|---|---|
| Otimizador | AdamW | AdamW | Padrão para Transformers; desacopla *weight decay* |
| Taxa de aprendizado | 2e-5 | 2e-5 | Faixa segura (2e-5–5e-5) para não destruir o pré-treino |
| *Warmup* | 10% dos passos | 10% | Estabiliza o início do *fine-tuning* |
| *Weight decay* | 0,01 | 0,01 | Regularização padrão |
| Épocas | 3–4 (early stopping) | 3 (early stopping) | BERT converge rápido; mais épocas → *overfitting* |
| *Batch size* | 32 | 8 | T2 tem sequências longas → *batch* menor por memória |
| `max_len` | 128 | 512 (+ janelamento) | Tweets são curtos; discursos exigem contexto longo |
| *Dropout* | 0,1 | 0,1 | Valor herdado do pré-treino; regulariza a cabeça |
| Perda | CE ponderada | CE ponderada | Trata desbalanceamento (4.5) |
| Critério de *early stopping* | macro-F1 (val.) | macro-F1 (val.) | Alinhado à métrica primária |

## 4.9 Controle de sobreajuste e reprodutibilidade

Contra *overfitting*: *early stopping* na macro-F1 de validação, *weight decay*, *dropout* e número baixo de épocas; opcionalmente *layer-wise learning rate decay* (camadas inferiores com LR menor). Para reprodutibilidade: sementes fixas (`random`, `numpy`, `torch`), modo determinístico quando viável, *split* estratificado e versionado, e registro das versões de `transformers`, `torch`, `scikit-learn` e `pandas`. Todo treino loga hiperparâmetros e métricas por época.

## 4.10 Resumo das decisões e o fio condutor

Cada decisão desta etapa serve à pergunta de pesquisa: codificadores casados ao registro (4.3) estabelecem baselines fortes e justos; o multi-task (4.6) e o teste de transferência (4.7) medem se há ponte entre os registros; e os controles (4.5, 4.8, 4.9) garantem que a comparação baseline-vs-multitask seja limpa o bastante para que a conclusão sobre H2 — positiva **ou** negativa — seja confiável.

---

### Fontes
- BERTimbau: Souza, F.; Nogueira, R.; Lotufo, R. (2020). *BERTimbau: Pretrained BERT Models for Brazilian Portuguese*. BRACIS. (`neuralmind/bert-base-portuguese-cased`)
- BERTabaporu: da Costa, P. B. et al. (2023). *BERTabaporu: assessing a genre-specific language model for Portuguese NLP*. RANLP. (`pablocosta/bertabaporu-base-uncased`)
- *Fine-tuning* BERT: Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
- ToLD-Br: Leite, J. A. et al. (2020). AACL-IJCNLP. https://github.com/JAugusto97/ToLD-Br
