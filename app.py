#!/usr/bin/env python3
# API local que serve os modelos do projeto para a pagina web (site_projeto.html).
#
# Frente TOXICIDADE (/api/toxicidade):
#   usa o BERT (pasta ./modelo_bert_toxicidade, baixada do Colab) se existir;
#   caso contrario, usa a rede neural ./modelo_rede_neural.joblib (so scikit-learn).
# Frente INCLINACAO (/api/inclinacao):
#   usa o BERTimbau (pasta ./modelo_bertimbau_politica, baixada do Colab) se existir.
#
# Rodar:
#   pip install -r requirements_api.txt   (para BERT, descomente torch/transformers)
#   python app.py                          ->  http://127.0.0.1:5000
import os, joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

TOX_BERT = "modelo_bert_toxicidade"
TOX_JOBLIB = "modelo_rede_neural.joblib"
POL_BERT = "modelo_bertimbau_politica"

tox_backend = None   # (nome, funcao->prob_toxico, limiar)
pol_backend = None   # (nome, funcao->prob_direita)

def _carregar_bert(pasta, max_len=128):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(pasta)
    mdl = AutoModelForSequenceClassification.from_pretrained(pasta).eval()
    def pred(texto):
        e = tok(texto, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(mdl(**e).logits, dim=-1)[0].tolist()
        return p[1]  # prob da classe 1
    return pred

def carregar():
    global tox_backend, pol_backend
    # toxicidade
    if os.path.isdir(TOX_BERT):
        tox_backend = ("BERT", _carregar_bert(TOX_BERT, 128), 0.5)
    elif os.path.exists(TOX_JOBLIB):
        d = joblib.load(TOX_JOBLIB)
        vec, svd, sc, model = d["vectorizer"], d["svd"], d["scaler"], d["model"]
        lim = d.get("limiar", 0.5)
        def pred(texto):
            X = sc.transform(svd.transform(vec.transform([texto])))
            return float(model.predict_proba(X)[0, 1])
        tox_backend = ("rede neural", pred, lim)
    # inclinacao (so se a pasta do BERTimbau existir)
    if os.path.isdir(POL_BERT):
        pol_backend = ("BERTimbau", _carregar_bert(POL_BERT, 256))
    print("Toxicidade:", tox_backend[0] if tox_backend else "INDISPONIVEL",
          "| Inclinacao:", pol_backend[0] if pol_backend else "indisponivel (treine o BERTimbau)")

@app.route("/api/health")
def health():
    return jsonify(ok=True,
                   toxicidade=tox_backend[0] if tox_backend else None,
                   inclinacao=pol_backend[0] if pol_backend else None)

@app.route("/api/toxicidade")
def toxicidade():
    if not tox_backend:
        return jsonify(error="modelo de toxicidade indisponivel"), 503
    texto = request.args.get("texto", "")
    if not texto.strip():
        return jsonify(error="texto vazio"), 400
    prob = tox_backend[1](texto); lim = tox_backend[2]
    tox = prob >= lim
    return jsonify(modelo=tox_backend[0], toxico=bool(tox),
                   prob_toxico=round(prob, 4),
                   confianca=round(prob if tox else 1 - prob, 4))

@app.route("/api/inclinacao")
def inclinacao():
    if not pol_backend:
        return jsonify(error="modelo de inclinacao indisponivel (treine o BERTimbau)"), 503
    texto = request.args.get("texto", "")
    if not texto.strip():
        return jsonify(error="texto vazio"), 400
    prob = pol_backend[1](texto)  # prob da DIREITA (classe 1)
    direita = prob >= 0.5
    return jsonify(modelo=pol_backend[0], direita=bool(direita),
                   prob_direita=round(prob, 4),
                   confianca=round(prob if direita else 1 - prob, 4))

if __name__ == "__main__":
    carregar()
    if not tox_backend:
        raise SystemExit("Nenhum modelo de toxicidade encontrado (rode treinar_rede_neural.py ou baixe o BERT).")
    print("Servidor em http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)
