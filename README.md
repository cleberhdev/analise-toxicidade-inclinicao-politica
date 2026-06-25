# Toxicidade e Inclinação Política em Dois Registros da Linguagem Brasileira

Projeto de **Deep Learning para Demandas Reais** — Atividade Prática.

Análise computacional comparada da linguagem política brasileira em **dois registros**:
toxicidade no discurso **cidadão/informal** (tweets) e inclinação ideológica no discurso
**institucional/formal** (discursos da Câmara dos Deputados), com um teste explícito de
**transferência entre registros**.

---

## Início rápido (rodar a página em 4 passos)

> Pré-requisito: **Python 3** instalado no Windows. (Detalhes completos na Seção 3.)

1. Abra a pasta `analise-toxicidade-inclinicao-politica` e dê **duplo-clique em `Iniciar_demo.bat`**
   (na 1ª vez ele cria o ambiente e instala as dependências — alguns minutos).
2. Vão abrir a janela **"Servidor de Modelos (NAO FECHE)"** e a página no Chrome. **Deixe a janela do servidor aberta.**
3. Na página, **role até "Analisador"**; deve aparecer o selo verde **"Modelo real conectado"**.
4. Digite um texto e clique em **Analisar**. A toxicidade mostra **"modelo rede neural"** (modelo real).
   Se aparecer "Servidor offline", aguarde uns segundos e tecle **Ctrl+F5**.

Para encerrar: feche a janela "Servidor de Modelos".

---

## 1. Objetivo do projeto

1. **Classificar toxicidade** em texto informal (tóxico / não-tóxico);
2. **Classificar inclinação ideológica** em discurso formal (esquerda / direita);
3. **Medir** se representações aprendidas em um registro transferem para o outro.

---

## 2. Resultados (treinados de verdade)

**Toxicidade** (ToLD-Br, 20.813 exemplos, split 70/15/15, macro-F1 como métrica primária):

| Modelo | Tipo | macro-F1 | ROC-AUC | F1 tóxico |
|---|---|---|---|---|
| TF-IDF + Regressão Logística | ML raso (baseline) | 0,70 | 0,77 | 0,52 |
| TF-IDF + SVD + MLP | Deep Learning | 0,67 | 0,74 | 0,46 |
| **BERT (BERTabaporu)** | **DL estado da arte** | **0,77** | **0,87** | **0,63** |

**Inclinação política** (discursos da Câmara, texto limpo, balanceado, split por deputado):

| Modelo | macro-F1 |
|---|---|
| Baseline TF-IDF + LogReg | 0,77 |
| Rede neural (TF-IDF + SVD + MLP) | 0,78 |
| BERTimbau (truncado, 256 tokens) | 0,73 |
| **BERTimbau + janelamento (documento inteiro)** | **0,77** |

**Multi-task** (encoder BERTimbau compartilhado, mesmo encoder nas três configurações):

| Configuração | macro-F1 toxicidade | macro-F1 política |
|---|---|---|
| Single-task | 0,70 | 0,73 |
| Multi-task | 0,68 | 0,77 |
| Δ (multi − single) | −1,9 p.p. | +4,0 p.p. |

Três achados metodológicos: (a) na toxicidade o **BERT vence** com folga; (b) na política, a
**truncagem** era o gargalo — o janelamento recupera o BERTimbau; (c) a transferência entre
registros é **assimétrica** (negativa na toxicidade, positiva na política). Detalhes na
`DOCUMENTACAO_TECNICA.docx`. (Durante o desenvolvimento, descobrimos e corrigimos um **vazamento
de rótulo** na frente política — a sigla do partido no cabeçalho do discurso inflava o resultado.)

---

## 3. Como executar a PÁGINA WEB com os modelos reais (passo a passo)

A página (`site_projeto.html`) não roda os modelos sozinha — quem os carrega e responde é um
servidor local (`app.py`). O atalho `Iniciar_demo.bat` cuida de tudo.

### Pré-requisito
- **Python 3** instalado no Windows (o projeto foi testado com Python 3.13).

### Passos
1. Abra a pasta `analise-toxicidade-inclinicao-politica`.
2. Dê **duplo-clique em `Iniciar_demo.bat`**. Na **primeira vez** ele cria um ambiente virtual
   isolado (`.venv`) e instala as dependências (flask, scikit-learn, joblib) — leva alguns minutos.
   Nas próximas vezes é quase imediato.
3. Vão abrir uma janela preta **"Servidor de Modelos (NAO FECHE)"** e a página no Chrome.
   **Deixe a janela do servidor aberta** — é ela que mantém os modelos no ar.
4. Na página, **role até "Analisador"**. Se aparecer o selo verde **"Modelo real conectado"**,
   está tudo certo. Digite um texto e clique em **Analisar**.
   - Toxicidade: mostra **"modelo rede neural"** (ou "modelo BERT") → modelo real respondendo.
   - Se mostrar "Servidor offline", espere uns segundos e aperte **Ctrl+F5**.
5. Para **encerrar**: feche a janela "Servidor de Modelos".

### Ativar também o modelo de INCLINAÇÃO (BERTimbau) — opcional
Por padrão a inclinação usa uma heurística. Para usar o modelo real:

1. Rode o notebook **`treinar_bertimbau_colab.ipynb`** no Google Colab (com GPU). A última célula
   gera e baixa `modelo_bertimbau_politica.zip`.
2. **Descompacte** o zip dentro de `analise-toxicidade-inclinicao-politica`, de modo a ficar a pasta
   `modelo_bertimbau_politica` ao lado do `app.py`.
3. Instale Torch e Transformers no ambiente (download grande, ~2 GB):
   ```
   .venv\Scripts\python -m pip install torch transformers
   ```
4. Reinicie o `Iniciar_demo.bat`. O `app.py` detecta a pasta sozinho; ao iniciar, a janela do
   servidor mostra `Toxicidade: rede neural | Inclinacao: BERTimbau`. Na página, a inclinação
   passa a exibir **"modelo BERTimbau"**.

> Observação honesta: o BERTimbau foi treinado em **discursos longos**; em frases curtas ele pode
> errar ou ficar inseguro — mas analisa o sentido (contexto/negação), não conta palavras.

---

## Como (re)treinar os modelos (reprodutibilidade)

Todos os modelos são reprodutíveis. A rede neural de toxicidade treina localmente (CPU); os
modelos **BERT/BERTimbau** treinam no **Google Colab** com GPU gratuita.

### Rede neural de toxicidade (local) → gera `modelo_rede_neural.joblib`

```bash
git clone https://github.com/JAugusto97/ToLD-Br.git
pip install -r requirements_treino.txt
python treinar_rede_neural.py
```

Saída: o modelo `.joblib`, as figuras (curva de loss, matriz de confusão) e as métricas.
As versões usadas no desenvolvimento estão fixadas em **`requirements_treino.txt`** (Python 3.13)
— use-o para reproduzir o mesmo ambiente.

> **Artefatos pesados não estão versionados.** Por ultrapassarem o limite de 100 MB do GitHub,
> a pasta `modelo_bertimbau_politica/` (e o `.zip` correspondente, com os pesos `model.safetensors`)
> estão no `.gitignore` e **não vêm no clone**. Para recriá-los, rode o notebook
> `treinar_bertimbau_colab.ipynb` no Colab (a última célula baixa o `.zip`) e descompacte-o ao lado
> do `app.py`. A rede neural de toxicidade (`modelo_rede_neural.joblib`, leve) **está** versionada.

### Modelos BERT (Google Colab, GPU) — os mesmos 4 passos para os 4 notebooks

1. Abra o notebook no [Google Colab](https://colab.research.google.com).
2. `Ambiente de execução → Alterar o tipo de ambiente de execução → GPU`.
3. `Ambiente de execução → Executar tudo`.
4. As células finais **baixam os resultados** (figuras, JSONs e, no BERTimbau, o `.zip` do modelo).

| Notebook | Treina | Saída principal |
|---|---|---|
| `treinar_bert_colab.ipynb` | BERT (BERTabaporu) — toxicidade | métricas + figuras |
| `treinar_bertimbau_colab.ipynb` | BERTimbau — inclinação (coleta + treina) | `modelo_bertimbau_politica.zip` (usado pela página) |
| `treinar_bertimbau_janelamento_colab.ipynb` | BERTimbau + janelamento | métricas + figuras |
| `treinar_multitask_colab.ipynb` | Multi-task (encoder compartilhado) | comparação single vs multi |

Reprodutibilidade garantida: sementes fixas, *splits* documentados (split por deputado na
inclinação) e limpeza de texto (remoção do cabeçalho com a sigla do partido).

---

## 🔭 Próximos passos / Como continuar este trabalho

> Esta seção é para quem pega o repositório para **dar continuidade** (ex.: próxima turma de
> Tópicos Especiais de Programação — TADS). O *porquê* de cada decisão está nas `etapas/` e na
> `DOCUMENTACAO_TECNICA.docx` (Seção 10 traz limitações e trabalhos futuros em detalhe).

**Por onde começar (em ordem de esforço/retorno):**

1. **Servir o BERTimbau com janelamento na API.** Hoje o `app.py` trunca a inclinação em 256 tokens;
   o janelamento (que recupera o macro-F1 de 0,73 → 0,77) só roda nos notebooks. Levar essa lógica
   de `demo_inferencia.py` para dentro do `app.py` é a melhoria de maior impacto e menor risco.
2. **Mais dados rotulados para a inclinação.** O rótulo atual vem do partido do deputado (supervisão
   distante, com ruído). Anotação por fala, ou uma classe **"neutro/other"**, tende a melhorar a
   qualidade — e permite reavaliar o multi-task com base maior.
3. **Confirmar a transferência assimétrica.** O achado (compartilhar encoder **piora** toxicidade e
   **melhora** política) saiu de um único experimento. Repetir com mais dados, variando o encoder
   compartilhado e o esquema de amostragem, fecharia H2/H3 com mais segurança.
4. **Calibração e limiar por custo de erro.** Ajustar probabilidades e o limiar de decisão conforme
   o custo de falso-positivo/negativo na moderação.
5. **Auditoria de viés/justiça na toxicidade.** Risco conhecido: sobre-marcação de dialetos
   minoritários e termos reapropriados. Avaliar por subgrupo antes de qualquer uso real.
6. **Empacotamento para produção.** Conteinerizar (Docker) o `app.py` + modelos e adicionar
   monitoramento de *drift* para uma implantação de verdade.

**Mapa rápido para se localizar no código:**

- Treino reprodutível → `treinar_rede_neural.py` (local) e os 4 notebooks `treinar_*_colab.ipynb`.
- Coleta dos discursos → `coletar_discursos_camara.py` (API de dados abertos da Câmara).
- Inferência com janelamento → `demo_inferencia.py` (a lógica a portar para o `app.py`).
- Pipeline de referência (esqueleto comentado das decisões das Etapas 4 e 5) → `pipeline.py`.
- Métricas e figuras de cada modelo → `figuras/` (JSONs `resultados_*.json` + PNGs).

**Antes de continuar:** instale o ambiente de treino com `pip install -r requirements_treino.txt`
e lembre que os pesos pesados do BERTimbau **não estão no repositório** (veja o aviso na seção de
treino acima — recrie pelo Colab).

---

## 4. Estrutura de arquivos (o que é cada coisa)

### Documentação
| Arquivo | O que é |
|---|---|
| `DOCUMENTACAO_TECNICA.docx` | Documentação técnica completa (7 etapas + resultados + histórico) |
| `README.md` | Este guia (resumo, execução, treino, arquivos, notebooks) |
| `etapas/` | As 7 etapas detalhadas, uma a uma, em markdown |

### Demonstração (página web e API)
| Arquivo | O que é |
|---|---|
| `Iniciar_demo.bat` | **Atalho principal**: cria o ambiente, sobe o servidor e abre a página |
| `site_projeto.html` | Página web interativa (a demonstração principal) |
| `demo_visual.html` | Demonstração simples, autônoma (heurística, sem servidor) |
| `app.py` | Servidor local (Flask) que carrega os modelos e responde à página |
| `requirements_api.txt` | Dependências do servidor (flask, flask-cors, scikit-learn, joblib) |
| `.venv/` | Ambiente virtual com as bibliotecas (criado pelo `Iniciar_demo.bat`) |

### Modelos treinados e dados
| Arquivo | O que é |
|---|---|
| `modelo_rede_neural.joblib` | Rede neural de **toxicidade** já treinada (pronta para uso) |
| `modelo_bertimbau_politica/` | Pasta do BERTimbau de **inclinação** (baixar do Colab — opcional) |
| `mapeamento_partidos_ideologia.csv` | Tabela partido → ideologia (esquerda/direita) |
| `figuras/` | Gráficos e métricas (matriz de confusão, ROC, curva de loss, JSONs de resultado) |
| `Bert_bertimbau/` | Saídas brutas baixadas do Colab (discursos, figuras, resultados) |

### Código de treino e inferência (executáveis)
| Arquivo | O que é |
|---|---|
| `treinar_rede_neural.py` | Treina a rede neural de toxicidade (TF-IDF → SVD → MLP) |
| `requirements_treino.txt` | Dependências (com versões fixadas) para **retreinar** os modelos |
| `pipeline.py` | Pipeline de treino/avaliação dos modelos (esqueleto) |
| `demo_inferencia.py` | Inferência sobre texto novo (com janelamento para texto longo) |
| `coletar_discursos_camara.py` | Coleta reprodutível dos discursos da Câmara (API de dados abertos) |

### Notebooks do Google Colab (treino com GPU)
| Notebook | O que treina |
|---|---|
| `treinar_bert_colab.ipynb` | **BERT (BERTabaporu)** para **toxicidade** — baixa o ToLD-Br, treina, avalia e salva |
| `treinar_bertimbau_colab.ipynb` | **BERTimbau** para **inclinação** (truncado em 256 tokens) — coleta os discursos da Câmara, limpa, treina e **salva a pasta `modelo_bertimbau_politica`** usada pela página |
| `treinar_bertimbau_janelamento_colab.ipynb` | **BERTimbau + janelamento** — lê o **documento inteiro** (fatiando em janelas); recupera o desempenho (0,77) |
| `treinar_multitask_colab.ipynb` | **Multi-task** — um encoder compartilhado com duas cabeças; compara single-task vs multi-task (H2/H3) |

> Para a **página web**, o notebook relevante é o `treinar_bertimbau_colab.ipynb`, pois é ele que
> gera a pasta com o nome exato (`modelo_bertimbau_politica`) que o `app.py` carrega.

---

## 5. Bases de dados

| Base | Frente | Origem | Licença |
|---|---|---|---|
| **ToLD-Br** | Toxicidade | [github.com/JAugusto97/ToLD-Br](https://github.com/JAugusto97/ToLD-Br) | CC BY-SA 4.0 |
| **Discursos da Câmara** | Inclinação | [dadosabertos.camara.leg.br](https://dadosabertos.camara.leg.br) | Dados públicos |

> O UStanceBR (proposta inicial) foi descartado por ser distribuído só como IDs de tweets, sem
> texto — inviável de reidratar de forma reprodutível. Detalhes na Seção 3 da documentação.

---

## 6. Tecnologias

Python 3; PyTorch + Hugging Face Transformers (BERTimbau, BERTabaporu); scikit-learn; pandas;
numpy; matplotlib; Flask (servidor da página); HTML/CSS/JavaScript (página web).

## 7. Ética e uso responsável

Ferramenta de apoio com **humano no circuito**, análise em **nível agregado**, e cada modelo
aplicado **apenas ao registro** para o qual foi treinado. O modelo de inclinação opera sobre
**discurso público de agentes públicos**, nunca inferindo ideologia de cidadãos.
