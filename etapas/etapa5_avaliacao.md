# Etapa 5, Avaliação dos Resultados

Projeto: Análise computacional comparada da linguagem política brasileira em dois registros. Esta etapa define **como** medir, comparar e interpretar os modelos da Etapa 4, de modo que as conclusões sobre as hipóteses H1, H3 (Etapa 2) sejam confiáveis, inclusive quando forem negativas.

## 5.1 Métricas e justificativa

A métrica **primária** é a **macro-F1**: média dos F1 por classe, sem ponderar por frequência. Ela é escolhida porque ambas as tarefas têm classes desbalanceadas (toxicidade ~19% positivos; volume desigual de discursos por ideologia), e a acurácia, nesse cenário, é enganosa, um modelo que sempre prevê a classe majoritária pode ter acurácia alta e F1 baixo na classe de interesse.

Reporta-se, como métricas secundárias: precisão, revocação e F1 **por classe** (essenciais para entender o comportamento na classe minoritária); a **matriz de confusão**; e, para a toxicidade, cujo uso é moderação, a curva **precisão, revocação** e o F1 da classe positiva, porque o custo de falsos negativos/positivos é assimétrico e dependente do limiar. A acurácia é reportada apenas como referência contextual, nunca como critério de decisão.

## 5.2 Protocolo de validação (e o risco de vazamento)

**Toxicidade (T1).** Usa-se o *split* oficial `3annotator` do ToLD-Br (treino/validação/teste) para comparabilidade direta com a literatura, após deduplicação de textos idênticos.

**Inclinação (T2), ponto crítico.** O *split* precisa ser **agrupado por deputado** (*group split*): nenhum deputado pode aparecer em treino e teste ao mesmo tempo. Sem isso, o modelo aprende o **estilo idiossincrático** de cada parlamentar e "vaza" essa identidade para o teste, inflando a métrica e medindo *quem fala* em vez de *qual ideologia*. Essa é exatamente a lição do *split* r3 do UStanceBR (usuários disjuntos), aqui transposta para deputados. O *split* é também **estratificado por ideologia** para preservar a proporção de classes.

**Robustez estatística.** Como uma única partição produz números frágeis, cada configuração é treinada com **múltiplas sementes** (mínimo 5) e reporta-se **média ± desvio-padrão**. Onde o orçamento computacional permitir, valida-se com *k-fold* agrupado (T2) para reduzir a variância da estimativa.

## 5.3 Comparações experimentais (cada uma testa uma hipótese)

A avaliação não é uma lista de números soltos: cada comparação responde a uma pergunta. As tabelas abaixo são os **gabaritos de resultado** a preencher após o treino.

**Desempenho por tarefa (macro-F1, média ± dp sobre sementes):**

| Modelo | T1, Toxicidade | T2, Inclinação |
|---|---|---|
| Baseline raso (TF-IDF + Reg. Logística) |, |, |
| mBERT (multilíngue) |, |, |
| Monolíngue especializado (BERTabaporu / BERTimbau) |, |, |
| Multi-task (encoder compartilhado) |, |, |

Leitura dirigida às hipóteses:

- **H1 (monolíngue > multilíngue):** comparar a linha *monolíngue especializado* vs. *mBERT*. Espera-se vantagem do monolíngue.
- **H2 (transferência fraca/negativa):** comparar *multi-task* vs. *monolíngue especializado*. Se o multi-task **não** superar (ou piorar), H2 se confirma, resultado central e plenamente válido.
- **Valor do Deep Learning:** comparar qualquer Transformer vs. *baseline raso*. Quantifica o ganho real de DL.

## 5.4 Significância estatística

Diferenças entre modelos só são afirmadas com suporte estatístico. Sobre as sementes, reporta-se intervalo de confiança da média. Para comparar dois classificadores **no mesmo conjunto de teste**, usa-se teste de **McNemar** (erros pareados) ou *bootstrap* da diferença de macro-F1 (reamostragem do teste, com IC de 95%). Uma diferença dentro da faixa de ruído entre sementes **não** é tratada como melhoria, princípio que protege especialmente a leitura de H2.

## 5.5 Análise de erros (qualitativa e por fatia)

Métrica agregada não explica *por que* o modelo erra. Faz-se análise dirigida:

Para **toxicidade**, inspecionam-se os erros típicos do domínio: ironia e sarcasmo ("que pessoa *maravilhosa*…"), termos reapropriados, e os subtipos raros (racismo, xenofobia) onde a escassez de dados (Etapa 3) deve produzir mais falhas. Fatia-se o desempenho por comprimento do texto e por subtipo de toxicidade.

Para **inclinação**, o foco é o ruído da supervisão distante: discursos protocolares/neutros rotulados como políticos, e deputados que falam contra a linha do próprio partido. Fatia-se o desempenho **por partido** e analisa-se a matriz de confusão por sigla, para distinguir erro do modelo de ruído do rótulo.

## 5.6 Calibração e limiar (para a demonstração)

Como a Etapa 7 expõe predições ao usuário, a **confiança** precisa ser informativa. Avalia-se a calibração (curva de confiabilidade / ECE) e, se necessário, aplica-se *temperature scaling*. Para a toxicidade, o **limiar de decisão** é ajustado conforme o uso de moderação: prioriza-se revocação alta (não deixar passar conteúdo tóxico) ou precisão alta (não punir conteúdo legítimo) de forma explícita, mostrando a curva precisão, revocação em vez de fixar 0,5 por inércia.

## 5.7 Referência da literatura (sanidade)

Para T1, o *baseline* BERT do paper original do ToLD-Br atinge **~0,76 de macro-F1** no caso binário, alvo de sanidade: nossos modelos devem ficar nessa vizinhança. Para T2 não há comparador direto na literatura (a tarefa foi construída neste projeto), então a referência é o ganho sobre o *baseline* raso e a consistência entre sementes.

## 5.8 Ameaças à validade e mitigações

**Validade de construto:** o rótulo de ideologia vem do partido (proxy), não de anotação por fala, mitigação: análise de erros por partido e cautela nas conclusões. **Validade interna (vazamento):** risco de o modelo aprender identidade do autor, mitigação: *group split* por deputado (5.2). **Validade externa:** ambas as bases refletem períodos específicos (Twitter 2017-2020; legislatura coletada), mitigação: declarar o escopo temporal e não extrapolar. **Ruído de anotação (T1):** rótulo por maioria de 3 anotadores, mitigação: reportar também o *split* oficial para comparabilidade.

## 5.9 Critério de decisão (o que conta como projeto bem-sucedido)

O projeto é considerado bem avaliado se: (a) os modelos profundos superam o *baseline* raso com significância; (b) T1 fica na vizinhança do estado da arte (~0,76 macro-F1); (c) a comparação multi-task vs. especializado conclui sobre H2 de forma estatisticamente sustentada, **mesmo que a conclusão seja "não há transferência"**; e (d) a análise de erros distingue limitação do modelo de ruído do rótulo. Reforçando a Etapa 2: um resultado negativo bem fundamentado é sucesso, porque responde à pergunta de pesquisa.

---

### Fontes
- ToLD-Br (referência ~0,76 macro-F1 binário): Leite, J. A. et al. (2020). *Toxic Language Detection in Social Media for Brazilian Portuguese*. AACL-IJCNLP. https://github.com/JAugusto97/ToLD-Br
- Lição de *split* por usuário/grupo: Pereira, C. et al. (2023). *UstanceBR* (split r3, usuários disjuntos). arXiv:2312.06374.
- McNemar para comparação de classificadores: Dietterich, T. (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*. Neural Computation.
