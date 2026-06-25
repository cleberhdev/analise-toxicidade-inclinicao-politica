#!/usr/bin/env python3
"""
demo_inferencia.py — Inferencia das duas frentes sobre texto novo.

Carrega um modelo treinado pelo pipeline.py e prediz:
  - toxicidade -> {nao-toxico, toxico}
  - inclinacao -> {esquerda, direita}

Para discursos longos (inclinacao), aplica JANELAMENTO + agregacao por media
de probabilidades (Etapa 4): o texto e fatiado em janelas de max_len tokens
com sobreposicao, cada janela e classificada, e as probabilidades sao mediadas.

Uso:
  python demo_inferencia.py --modelo modelos/tox --tarefa toxicidade \
      --texto "seu texto aqui"
"""

import argparse

ROTULOS = {
    "toxicidade": {0: "nao-toxico", 1: "toxico"},
    "inclinacao": {0: "esquerda", 1: "direita"},
}


def _janelas(input_ids, max_len, stride):
    """Gera janelas de ids com sobreposicao (stride)."""
    if len(input_ids) <= max_len:
        return [input_ids]
    janelas, i = [], 0
    while i < len(input_ids):
        janelas.append(input_ids[i:i + max_len])
        if i + max_len >= len(input_ids):
            break
        i += max_len - stride
    return janelas


def prever(texto, modelo_dir, tarefa, max_len=None, stride=64):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

    tok = AutoTokenizer.from_pretrained(modelo_dir)
    model = AutoModelForSequenceClassification.from_pretrained(modelo_dir).eval()
    max_len = max_len or (128 if tarefa == "toxicidade" else 512)

    # tokeniza sem padding p/ obter a sequencia real e janelar
    ids = tok(texto, add_special_tokens=False)["input_ids"]
    cls, sep = tok.cls_token_id, tok.sep_token_id
    corpo = max_len - 2  # espaco para [CLS] e [SEP]

    probs_acc = None
    n = 0
    with torch.no_grad():
        for jan in _janelas(ids, corpo, stride):
            seq = [cls] + jan + [sep]
            attn = [1] * len(seq)
            # padding ate max_len
            while len(seq) < max_len:
                seq.append(tok.pad_token_id)
                attn.append(0)
            batch = {
                "input_ids": torch.tensor([seq]),
                "attention_mask": torch.tensor([attn]),
            }
            logits = model(**batch).logits
            p = torch.softmax(logits, dim=-1)[0]
            probs_acc = p if probs_acc is None else probs_acc + p
            n += 1

    probs = (probs_acc / n).tolist()
    idx = int(max(range(len(probs)), key=lambda k: probs[k]))
    return {
        "rotulo": ROTULOS[tarefa][idx],
        "confianca": round(probs[idx], 4),
        "probabilidades": {ROTULOS[tarefa][k]: round(probs[k], 4)
                           for k in range(len(probs))},
        "n_janelas": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, help="diretorio do modelo treinado")
    ap.add_argument("--tarefa", required=True, choices=["toxicidade", "inclinacao"])
    ap.add_argument("--texto", required=True)
    ap.add_argument("--max_len", type=int, default=None)
    args = ap.parse_args()

    r = prever(args.texto, args.modelo, args.tarefa, args.max_len)
    print(f"Tarefa:     {args.tarefa}")
    print(f"Predicao:   {r['rotulo']}  (confianca {r['confianca']:.1%})")
    print(f"Detalhe:    {r['probabilidades']}  | janelas={r['n_janelas']}")


if __name__ == "__main__":
    main()
