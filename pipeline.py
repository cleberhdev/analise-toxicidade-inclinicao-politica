#!/usr/bin/env python3
"""
pipeline.py — Treino e avaliacao das duas frentes do projeto:
  - toxicidade  (ToLD-Br, registro informal)
  - inclinacao  (discursos da Camara, registro formal)

Implementa as decisoes das Etapas 4 e 5:
  - encoder casado ao registro (BERTabaporu p/ tweets, BERTimbau p/ discurso)
  - perda ponderada por classe (desbalanceamento)
  - split agrupado por deputado na inclinacao (evita vazamento de identidade)
  - macro-F1 como metrica primaria, com early stopping
  - sementes fixas para reprodutibilidade

Requer: torch, transformers, scikit-learn, pandas, numpy.
Este e um ESQUELETO executavel: rode em ambiente com GPU e com as bases disponiveis.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd

# Imports de DL ficam dentro das funcoes para o arquivo poder ser importado/lintado
# mesmo sem torch/transformers instalados.

ENCODERS_PADRAO = {
    "toxicidade": "pablocosta/bertabaporu-base-uncased",  # dominio: tweets PT
    "inclinacao": "neuralmind/bert-base-portuguese-cased",  # dominio: texto formal
}


# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #
def fixar_sementes(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Carregamento das bases
# --------------------------------------------------------------------------- #
CATS_TOX = ["homophobia", "obscene", "insult", "racism", "misogyny", "xenophobia"]


def carregar_toldbr(csv_path: str, limiar_votos: int = 2) -> pd.DataFrame:
    """ToLD-Br -> DataFrame(texto, label). label=1 se alguma categoria tem
    >= limiar_votos (default 2 = maioria de 3 anotadores)."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["label"] = (df[CATS_TOX].max(axis=1) >= limiar_votos).astype(int)
    out = df[["text", "label"]].rename(columns={"text": "texto"})
    return out


def carregar_camara(csv_path: str) -> pd.DataFrame:
    """Discursos da Camara -> DataFrame(texto, label, grupo).
    label: 0=esquerda, 1=direita. grupo=id_deputado (para split agrupado)."""
    df = pd.read_csv(csv_path)
    df = df[df["ideologia"].isin(["esquerda", "direita"])].copy()
    df["label"] = (df["ideologia"] == "direita").astype(int)
    df = df.rename(columns={"texto": "texto", "id_deputado": "grupo"})
    return df[["texto", "label", "grupo"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def split_estratificado(df, seed=42, val=0.1, test=0.1):
    """Split simples estratificado por label (toxicidade)."""
    from sklearn.model_selection import train_test_split
    tr, tmp = train_test_split(df, test_size=val + test, stratify=df["label"],
                               random_state=seed)
    rel = test / (val + test)
    va, te = train_test_split(tmp, test_size=rel, stratify=tmp["label"],
                              random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def split_agrupado(df, seed=42, val=0.15, test=0.15):
    """Split agrupado por 'grupo' (deputado): nenhum deputado em dois conjuntos.
    Evita que o modelo aprenda a identidade do autor (Etapa 5)."""
    from sklearn.model_selection import GroupShuffleSplit
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val + test, random_state=seed)
    idx_tr, idx_tmp = next(gss1.split(df, df["label"], groups=df["grupo"]))
    tr, tmp = df.iloc[idx_tr], df.iloc[idx_tmp]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test / (val + test), random_state=seed)
    idx_va, idx_te = next(gss2.split(tmp, tmp["label"], groups=tmp["grupo"]))
    va, te = tmp.iloc[idx_va], tmp.iloc[idx_te]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Dataset / metrica / treino
# --------------------------------------------------------------------------- #
def construir_dataset(df, tokenizer, max_len):
    import torch

    class DS(torch.utils.data.Dataset):
        def __init__(self, frame):
            self.textos = frame["texto"].astype(str).tolist()
            self.labels = frame["label"].tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            enc = tokenizer(self.textos[i], truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
            return item

    return DS(df)


def metricas(eval_pred):
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average=None,
                                                 labels=[0, 1], zero_division=0)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_classe0": f[0], "f1_classe1": f[1],
    }


def pesos_classe(df):
    """Pesos inversamente proporcionais a frequencia (perda ponderada)."""
    import torch
    n = len(df)
    n1 = df["label"].sum()
    n0 = n - n1
    w0 = n / (2 * n0) if n0 else 1.0
    w1 = n / (2 * n1) if n1 else 1.0
    return torch.tensor([w0, w1], dtype=torch.float)


def treinar(tarefa, df_tr, df_va, df_te, encoder, max_len, lr, epocas,
            batch_size, seed, saida):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(encoder)
    model = AutoModelForSequenceClassification.from_pretrained(encoder, num_labels=2)

    pesos = pesos_classe(df_tr)

    class TrainerPonderado(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=pesos.to(out.logits.device))
            loss = loss_fct(out.logits, labels)
            return (loss, out) if return_outputs else loss

    args = TrainingArguments(
        output_dir=saida,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epocas,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = TrainerPonderado(
        model=model, args=args,
        train_dataset=construir_dataset(df_tr, tokenizer, max_len),
        eval_dataset=construir_dataset(df_va, tokenizer, max_len),
        compute_metrics=metricas,
    )
    trainer.train()

    print("\n[AVALIACAO NO TESTE]")
    res = trainer.evaluate(construir_dataset(df_te, tokenizer, max_len))
    for k, v in res.items():
        if k.startswith("eval_"):
            print(f"  {k.replace('eval_',''):12s}: {v:.4f}")

    trainer.save_model(saida)
    tokenizer.save_pretrained(saida)
    print(f"[OK] Modelo salvo em {saida}")
    return res


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefa", required=True, choices=["toxicidade", "inclinacao"])
    ap.add_argument("--told_csv", help="ToLD-BR.csv (tarefa toxicidade)")
    ap.add_argument("--camara_csv", help="discursos_camara.csv (tarefa inclinacao)")
    ap.add_argument("--encoder", default=None, help="HF model id; default casado ao registro")
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epocas", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--saida", default="modelo_saida")
    args = ap.parse_args()

    fixar_sementes(args.seed)
    encoder = args.encoder or ENCODERS_PADRAO[args.tarefa]

    if args.tarefa == "toxicidade":
        assert args.told_csv, "--told_csv obrigatorio"
        df = carregar_toldbr(args.told_csv)
        tr, va, te = split_estratificado(df, seed=args.seed)
        max_len = args.max_len or 128
        batch = args.batch_size or 32
    else:
        assert args.camara_csv, "--camara_csv obrigatorio"
        df = carregar_camara(args.camara_csv)
        tr, va, te = split_agrupado(df, seed=args.seed)
        max_len = args.max_len or 512
        batch = args.batch_size or 8

    print(f"[i] Tarefa={args.tarefa} | encoder={encoder} | max_len={max_len}")
    print(f"[i] Tamanhos: treino={len(tr)} val={len(va)} teste={len(te)}")
    print(f"[i] Distribuicao treino: {tr['label'].value_counts().to_dict()}")

    treinar(args.tarefa, tr, va, te, encoder, max_len, args.lr,
            args.epocas, batch, args.seed, args.saida)


if __name__ == "__main__":
    main()
