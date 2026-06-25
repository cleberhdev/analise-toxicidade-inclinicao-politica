# Etapa 6, Aplicabilidade Real

Projeto: Análise computacional comparada da linguagem política brasileira em dois registros. Esta etapa conecta os modelos a **demandas reais**, define como entregariam valor, e, com igual peso, onde **não** devem ser usados.

## 6.1 Demandas reais atendidas

O projeto endereça três demandas concretas, uma por frente e uma pela conclusão metodológica:

A frente de **toxicidade** responde à demanda de **moderação em escala**: plataformas, gestores de comunidade e observatórios de discurso de ódio recebem volume de texto que nenhuma equipe humana revisa manualmente. Um classificador de toxicidade prioriza a fila de revisão, leva ao humano primeiro o que tem maior probabilidade de ser tóxico.

A frente de **inclinação ideológica** responde à demanda de **análise e transparência do discurso institucional**: ciência política computacional, jornalismo de dados e monitoramento legislativo precisam mapear tendências ideológicas em grandes volumes de discursos parlamentares, algo inviável de ler à mão.

A **conclusão sobre transferência entre registros** (H2/H3) atende a uma demanda do próprio praticante de ML: a evidência de que um modelo de linguagem política não se reaproveita entre registros é um aviso prático que evita desperdício e erro em projetos futuros.

## 6.2 Como entregaria valor (cenários de implantação)

**Toxicidade, assistente de moderação com humano no circuito.** O modelo não decide sozinho: ele ordena e sinaliza, com o limiar calibrado (Etapa 5) ao custo de erro do contexto, e exibe a confiança da predição. A decisão final de remover/punir permanece humana. É assim que se captura o ganho de escala sem terceirizar a um modelo uma decisão sensível.

**Inclinação, ferramenta analítica de nível agregado.** O valor está em medir **tendências** (ex.: distribuição ideológica de discursos sobre um tema, ao longo do tempo, por bancada), não em carimbar um discurso individual como verdade. Reporta-se sempre com incerteza e em agregado, onde o ruído da supervisão distante se dilui.

## 6.3 Limites de aplicabilidade (declarados)

Três limites condicionam qualquer uso real:

**Deriva de domínio e período.** Os modelos aprendem a gíria, os temas e os atores de um recorte temporal (Twitter 2017-2020; legislatura coletada). Linguagem política muda rápido; sem **retreino periódico**, o desempenho degrada. O script reprodutível de coleta (Etapa 3) é justamente o mecanismo para atualizar a base.

**Natureza do rótulo político.** T2 mede *linguagem alinhada a partido*, não convicção pessoal. Usá-lo para afirmar a ideologia "real" de um indivíduo extrapola o que o rótulo suporta.

**Não transferência entre registros (H3).** Não se aplica o modelo de toxicidade a texto formal, nem o de ideologia a comentários de redes sociais, a própria pesquisa do projeto evidencia que esse reaproveitamento falha.

## 6.4 Riscos e ética

Por lidar com toxicidade **e** política, o projeto exige cuidado ético explícito.

**Viés e justiça (toxicidade).** Modelos de toxicidade tendem a marcar em excesso dialetos minoritários e termos reapropriados pelos próprios grupos, podendo silenciar quem deveriam proteger. Mitigações: auditoria de viés por fatia, revisão humana das remoções e transparência sobre as limitações, nunca moderação totalmente automática.

**Sensibilidade política e risco de vigilância.** Classificar orientação ideológica é tecnologia sensível: aplicada a cidadãos comuns, pode alimentar vigilância, perfilamento e efeito inibitório sobre a liberdade de expressão. Por isso o projeto delimita, desde a Etapa 1, que **não** infere ideologia de cidadãos, T2 opera sobre discurso **público de agentes públicos** (parlamentares no exercício do mandato), cujo escrutínio é legítimo e esperado em democracia, e apenas em nível agregado.

**Privacidade e LGPD.** Discursos parlamentares são públicos por natureza. Os tweets do ToLD-Br já vêm com menções anonimizadas e sob licença de pesquisa. Nenhum dado pessoal sensível de indivíduos privados é processado.

**Responsabilização.** Em ambas as frentes, mantém-se humano no circuito, possibilidade de contestação da decisão e documentação das limitações, o modelo é ferramenta de apoio, não autoridade final.

## 6.5 Manutenção e ciclo de vida

Uma solução real não termina no treino. Prevê-se: **monitoramento de deriva** (queda de métrica e mudança de distribuição de entrada ao longo do tempo), **retreino periódico** com bases atualizadas pelo script de coleta reprodutível, versionamento de modelos e dados, e registro de decisões. O pipeline reprodutível da Etapa 3 é o que torna esse ciclo viável sem refazer o projeto do zero.

## 6.6 Síntese

O projeto entrega valor real em moderação assistida e em análise de discurso institucional, desde que usado **com humano no circuito, em nível agregado e dentro do registro para o qual cada modelo foi treinado**. As mesmas limitações que a avaliação revela (Etapa 5) são as que delimitam o uso responsável aqui, coerência que a documentação técnica preserva de ponta a ponta.

---

### Fontes
- Viés em detecção de toxicidade (sobre-marcação de dialetos minoritários): Sap, M. et al. (2019). *The Risk of Racial Bias in Hate Speech Detection*. ACL.
- ToLD-Br (licença e anonimização): Leite, J. A. et al. (2020). AACL-IJCNLP. https://github.com/JAugusto97/ToLD-Br
- Dados públicos: Portal de Dados Abertos da Câmara dos Deputados, https://dadosabertos.camara.leg.br
