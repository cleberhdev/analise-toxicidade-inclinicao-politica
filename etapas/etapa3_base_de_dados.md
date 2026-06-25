# Etapa 3, Base de Dados

Projeto: Análise computacional comparada da linguagem política brasileira em dois registros, toxicidade no discurso cidadão (tweets) e sinal ideológico no discurso institucional (discursos parlamentares). Deep Learning para demandas reais.

## 3.1 Visão geral

A tese do projeto é comparar como **sinais de comportamento político** se manifestam, e se transferem, entre **dois registros linguísticos distintos**. Não se trata de moderar comentários de um único domínio, e sim de estudar duas faces da linguagem política brasileira:

1. **Toxicidade / ofensividade** no discurso **cidadão e informal** (tweets), detectar se um texto é tóxico (ofensivo, insultuoso, de ódio).
2. **Inclinação ideológica** no discurso **institucional e formal** (discursos parlamentares), classificar a orientação (esquerda / direita).

A diferença de registro entre as duas bases é, portanto, um **eixo deliberado de análise**, não uma inconsistência: ela permite investigar se representações aprendidas em um registro transferem para o outro (questão central da Etapa 4, via teste do multi-task com encoder compartilhado).

Cada frente é alimentada por uma base distinta, escolhida com dois critérios inegociáveis para um projeto cujo mérito inclui *reprodutibilidade* e *documentação técnica*: (i) o texto precisa estar disponível de forma aberta (não apenas como identificadores), e (ii) a licença precisa permitir reuso para pesquisa.

A tabela abaixo resume as duas bases selecionadas.

| Frente | Base | Origem | Texto aberto? | Licença | Rótulo usado |
|---|---|---|---|---|---|
| Toxicidade | ToLD-Br | Twitter PT-BR | Sim (CSV com texto) | CC BY-SA 4.0 | Binário: tóxico / não-tóxico |
| Inclinação política | Discursos da Câmara dos Deputados | API de dados abertos da Câmara | Sim (texto integral via API) | Dados públicos / dados abertos gov.br | Binário: esquerda / direita (via partido) |

## 3.2 Base 1, Toxicidade (ToLD-Br)

### Origem e características
ToLD-Br (*Toxic Language Dataset for Brazilian Portuguese*, Leite et al., 2020) é uma coleção de tweets em português brasileiro anotados quanto a aspectos de toxicidade. Foi escolhida por ser a maior base aberta de toxicidade em PT-BR que **distribui o texto** (sob CC BY-SA 4.0), o que a torna diretamente utilizável e reprodutível.

### Schema e tamanho (verificado)
O arquivo principal `ToLD-BR.csv` contém **21.000 exemplos**, com a coluna `text` e seis colunas de categoria de toxicidade: `homophobia`, `obscene`, `insult`, `racism`, `misogyny`, `xenophobia`. Cada categoria recebe um valor de 0 a 3 representando **quantos dos três anotadores** marcaram aquele aspecto. O repositório também fornece um recorte binário pronto (`experiments/data/3annotator/ptbr_{train,validation,test}_3annotator.csv`, colunas `text,toxic`).

### Distribuição de classes (medida diretamente sobre o CSV)

| Recorte | Positivos | Proporção |
|---|---|---|
| Binário, ≥1 voto em alguma categoria | 9.255 | 44,1% |
| Binário, ≥2 votos em alguma categoria | 4.063 | 19,3% |
| obscene (≥2 votos) | 2.403 | 11,4% |
| insult (≥2 votos) | 1.869 | 8,9% |
| homophobia (≥2 votos) | 176 | 0,8% |
| misogyny (≥2 votos) | 133 | 0,6% |
| xenophobia (≥2 votos) | 42 | 0,2% |
| racism (≥2 votos) | 33 | 0,2% |

### Decisão de granularidade e justificativa
Adotamos o **rótulo binário** (tóxico / não-tóxico). A justificativa é empírica: quatro das seis categorias de toxicidade têm menos de 200 exemplos positivos no limiar de ≥2 votos (racism com 33, xenophobia com 42, misogyny com 133, homophobia com 176). Com essa escassez, uma cabeça de classificação multi-rótulo memorizaria ruído nessas classes e produziria métricas instáveis, na prática o multi-rótulo colapsa para "obscene + insult", as duas únicas categorias com massa estatística. O recorte binário, por outro lado, é saudável (≈19% de positivos no limiar ≥2; ≈44% no limiar ≥1), o que viabiliza um classificador estável e bem avaliável.

O multi-rótulo (restrito a obscene + insult) fica registrado como **extensão opcional** para discussão na avaliação de resultados, não como objetivo principal.

### Pré-processamento previsto
As menções a usuários já vêm anonimizadas como `@user` na base. O pipeline padronizará: normalização de hashtags, emojis e URLs; manutenção de caixa conforme o tokenizador escolhido; e o limiar de rotulagem binária fixado em ≥2 votos (maioria simples dos 3 anotadores), ou, alternativamente, o split oficial `3annotator` para comparabilidade com o paper original.

## 3.3 Base 2, Inclinação política (Discursos da Câmara dos Deputados)

### Por que não usamos o UStanceBR
A proposta inicial previa o UStanceBR (postura político em Twitter PT-BR). Ao inspecionar o pacote oficial, constatamos que **todo o corpus é distribuído apenas como identificadores de tweets** (`Tweet_ID;Polarity`), inclusive as timelines, não há texto. O próprio README é explícito: por política de privacidade do Twitter/X, para obter o texto é necessário (i) permissão da plataforma e (ii) reidratação via API. Em 2026 isso é inviável de forma reprodutível: a API do X não oferece *lookup* em volume no nível gratuito, e os tweets são de 2018-2020 (alta taxa de remoção/suspensão após ~6 anos), resultando em recuperação parcial e enviesada. O mesmo problema afeta outras bases de stance PT-BR baseadas em Twitter (ex.: Zé Lensky). Por isso o UStanceBR foi descartado como fonte primária.

### Por que os discursos da Câmara
Optou-se por **reenquadrar** a tarefa de "postura sobre um alvo" para "inclinação ideológica do autor", usando os discursos parlamentares da Câmara dos Deputados, obtidos pela **API de dados abertos** (`dadosabertos.camara.leg.br`). Vantagens decisivas: o texto integral é público e aberto (sem o problema de IDs), a coleta é totalmente reprodutível por qualquer pessoa, o volume é alto (dezenas de milhares de discursos) e o conteúdo é genuinamente político em português brasileiro.

### Schema da fonte
A coleta usa dois endpoints encadeados:

- `GET /api/v2/deputados?idLegislatura={N}` → lista de deputados, com `id` e `siglaPartido`.
- `GET /api/v2/deputados/{id}/discursos` → discursos do deputado, incluindo o campo `transcricao` (texto integral), além de `dataHoraInicio`, `siglaTipoDiscurso` e metadados.

O dataset final terá, no mínimo, as colunas: `texto` (transcrição), `partido`, `ideologia` (rótulo derivado), `id_deputado`, `data`.

### Estratégia de rotulagem (supervisão distante)
O rótulo de ideologia (esquerda / direita) é atribuído **pelo partido do deputado**, mapeado para o eixo esquerda, direita. Como referência de posicionamento recomenda-se a classificação por survey de especialistas de **Bolognesi, Ribeiro & Codato (2023), "A New Measure of Party Ideology in Brazil"** (escala contínua 1-10), que deve ser consultada para fixar os escores exatos. A tabela abaixo é um mapeamento ternário de trabalho (a ser confirmado contra a fonte antes da execução):

| Tendência | Exemplos de partidos |
|---|---|
| Esquerda | PT, PSOL, PCdoB, PSB, PDT, Rede |
| Centro | MDB, PSD, Cidadania, PV, Solidariedade, Avante, Podemos |
| Direita | PL, PP, Republicanos, União Brasil, Novo, PSC, PSDB (centro-direita) |

Para a tarefa **binária** esquerda/direita, a faixa de centro é descartada (ou tratada como classe separada apenas em análise exploratória), o que reduz ruído e produz um problema melhor definido.

### Riscos da supervisão distante (declarados)
O rótulo vem do partido, não de anotação por fala, logo há **ruído de rótulo**: um deputado pode discursar contra a linha do próprio partido, e discursos protocolares/neutros recebem rótulo político ainda que pouco ideológicos. Mitigações previstas: (i) filtrar discursos muito curtos; (ii) opcionalmente restringir a tipos de discurso mais substantivos; (iii) reportar a incerteza e analisar erros por partido. Essas limitações serão tratadas explicitamente na Etapa 5 (Avaliação).

## 3.4 Comparabilidade e alinhamento entre as bases

As duas bases diferem em **domínio linguístico**: tweets curtos e informais (toxicidade) versus discurso parlamentar formal e longo (inclinação política). Esse contraste é relevante para a modelagem (Etapa 4): ele enfraquece a hipótese de um encoder compartilhado em Multi-task Learning (risco de *negative transfer*), e por isso o desenho principal será de **dois modelos independentes**, com o MTL avaliado como hipótese explicitamente testada. O alinhamento operacional comum entre as bases se dá no nível de pipeline: mesma rotina de limpeza textual configurável, mesmos formatos de split (treino/validação/teste estratificados), mesma semente aleatória e mesmas métricas de avaliação (macro-F1 como métrica primária, por robustez a desbalanceamento).

## 3.5 Reprodutibilidade

A reprodutibilidade é garantida por: (i) ToLD-Br versionado e baixável diretamente do repositório oficial sob CC BY-SA; (ii) um **script de coleta** parametrizável para os discursos da Câmara (legislatura, intervalo de datas, filtros), que reconstrói o dataset político a partir da API pública, entregue como artefato do projeto; (iii) a **tabela de mapeamento partido→ideologia** versionada junto ao código, com a fonte (Bolognesi et al., 2023) citada; (iv) fixação de sementes aleatórias e registro das versões de bibliotecas (`transformers`, `torch`, `scikit-learn`, `pandas`). Observação de ambiente: a coleta dos discursos deve ser executada na máquina do usuário, pois requer acesso ao domínio `dadosabertos.camara.leg.br`.

## 3.6 Limitações e ameaças à validade

A base de toxicidade reflete o Twitter de um período específico e categorias raras (racismo, xenofobia) são subrepresentadas, limitando a detecção desses subtipos. A base política usa supervisão distante por partido, com o ruído já descrito (o rótulo capta tanto ideologia quanto jargão partidário). A diferença de registro entre as duas bases **não** é uma limitação, mas o objeto de estudo do projeto, ainda assim, ela impõe que conclusões de um registro não sejam extrapoladas ao outro sem o teste de transferência previsto na Etapa 4. Essas restrições são assumidas conscientemente em troca de **dados abertos e reprodutíveis**, e serão revisitadas na avaliação e na discussão de aplicabilidade real.

---

### Fontes
- ToLD-Br: Leite, J. A.; Silva, D.; Bontcheva, K.; Scarton, C. (2020). *Toxic Language Detection in Social Media for Brazilian Portuguese: New Dataset and Multilingual Analysis*. AACL-IJCNLP. Repositório: https://github.com/JAugusto97/ToLD-Br
- UStanceBR (analisado e descartado como fonte primária): Pereira, C. et al. (2023/2025). *UstanceBR: a social media language resource for stance prediction*. arXiv:2312.06374.
- API de dados abertos da Câmara dos Deputados: https://dadosabertos.camara.leg.br
- Classificação ideológica de partidos: Bolognesi, B.; Ribeiro, E.; Codato, A. (2023). *A New Measure of Party Ideology in Brazil*.
