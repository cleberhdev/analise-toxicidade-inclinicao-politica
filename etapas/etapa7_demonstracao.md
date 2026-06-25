# Etapa 7 — Demonstração Funcional

Projeto: Análise computacional comparada da linguagem política brasileira em dois registros. Esta etapa entrega a prova de funcionamento: um pipeline executável ponta a ponta e uma demonstração de inferência das duas frentes.

## 7.1 O que a demonstração mostra

A demonstração recebe um texto em português e devolve, para a frente apropriada ao registro:

- **Toxicidade:** rótulo (tóxico / não-tóxico) com probabilidade.
- **Inclinação ideológica:** rótulo (esquerda / direita) com probabilidade.

Cada predição vem acompanhada de **confiança** e de um aviso de registro: o modelo de toxicidade é aplicável a texto informal; o de inclinação, a discurso formal. Esse aviso materializa, no produto, a conclusão de não transferência entre registros (H3) — a demo não esconde a limitação, ela a comunica.

## 7.2 Artefatos entregues

1. **`pipeline.py`** — código executável que cobre o ciclo completo: carregamento e preparação das duas bases, treino de um Transformer por tarefa (com perda ponderada e *early stopping*), avaliação por macro-F1, e salvamento dos modelos. Inclui o *split* agrupado por deputado (T2) e o janelamento de texto longo (Etapa 4/5).
2. **`demo_inferencia.py`** — carrega os modelos treinados e expõe uma função `prever(texto, tarefa)` para inferência sobre texto novo, com a probabilidade calibrável.
3. **`demo_visual.html`** — protótipo visual interativo que simula a inferência das duas frentes, para apresentação da ideia sem necessidade de GPU.
4. **`coletar_discursos_camara.py`** e **`mapeamento_partidos_ideologia.csv`** (Etapa 3) — reconstroem a base política de forma reprodutível.

## 7.3 Como executar (ambiente real, com GPU)

```bash
# 1. Dependências
pip install torch transformers scikit-learn pandas numpy

# 2. Base de toxicidade: clonar ToLD-Br
git clone https://github.com/JAugusto97/ToLD-Br.git

# 3. Base política: coletar discursos (na sua máquina, acessa a API da Câmara)
python coletar_discursos_camara.py --legislatura 57 \
    --data-inicio 2023-02-01 --data-fim 2023-12-31 \
    --min-palavras 50 --saida discursos_camara.csv

# 4. Treinar
python pipeline.py --tarefa toxicidade --told_csv ToLD-Br/ToLD-BR.csv --saida modelos/tox
python pipeline.py --tarefa inclinacao --camara_csv discursos_camara.csv --saida modelos/pol

# 5. Inferência sobre texto novo
python demo_inferencia.py --modelo modelos/tox --tarefa toxicidade --texto "seu texto aqui"
```

O `pipeline.py` é parametrizável (encoder, `max_len`, *learning rate*, épocas, semente), de modo que todos os experimentos das Etapas 4 e 5 — incluindo o controle mBERT e a variação base/large — sejam reproduzíveis pela linha de comando.

## 7.4 Observações de reprodutibilidade e ambiente

A coleta dos discursos roda na máquina do usuário (a API `dadosabertos.camara.leg.br` exige acesso direto). O treino exige GPU para tempo razoável; sem GPU, recomenda-se reduzir o tamanho da amostra (`--max-por-deputado`) e usar o encoder `base`. Sementes, versões de biblioteca e hiperparâmetros são logados a cada execução. A **demo visual** (`demo_visual.html`) funciona em qualquer navegador e serve para comunicar o conceito; ela usa uma heurística leve apenas para ilustrar a interface — os números reais vêm dos modelos treinados pelo `pipeline.py`.

## 7.5 Fechamento

Com as sete etapas, o projeto cobre o arco completo exigido: problema bem estruturado e reenquadrado em torno de uma pergunta de pesquisa real (Etapas 1–2), bases abertas e reprodutíveis com decisões justificadas por dados medidos (Etapa 3), modelagem em Deep Learning com codificadores casados ao registro e um experimento de transferência (Etapa 4), avaliação rigorosa com controle de vazamento e significância (Etapa 5), aplicabilidade real com ética e ciclo de vida (Etapa 6) e, agora, a demonstração funcional (Etapa 7).
