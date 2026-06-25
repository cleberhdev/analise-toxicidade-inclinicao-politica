#!/usr/bin/env python3
"""
Coleta de discursos da Camara dos Deputados (API de dados abertos) e rotulagem
por inclinacao ideologica (esquerda/direita) via partido do deputado.

Gera a base de "inclinacao politica" do projeto, com TEXTO INTEGRAL e licenca aberta.

Uso tipico:
    python coletar_discursos_camara.py \
        --legislatura 57 \
        --data-inicio 2023-02-01 --data-fim 2023-12-31 \
        --min-palavras 50 \
        --mapa mapeamento_partidos_ideologia.csv \
        --saida discursos_camara.csv

Observacoes:
- Rode na sua maquina (precisa acessar dadosabertos.camara.leg.br).
- A API e paginada; o script segue o link "next" automaticamente.
- Politica de boa vizinhanca: ha um pequeno sleep entre chamadas.
"""

import argparse
import csv
import sys
import time
import urllib.parse
import urllib.request
import json

API = "https://dadosabertos.camara.leg.br/api/v2"
HEADERS = {"Accept": "application/json", "User-Agent": "projeto-dl-pt/1.0"}


def get_json(url, tentativas=4, pausa=1.0):
    """GET com retry exponencial simples."""
    for i in range(tentativas):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i == tentativas - 1:
                raise
            time.sleep(pausa * (2 ** i))
    return None


def listar_deputados(legislatura):
    """Retorna lista de dicts {id, nome, siglaPartido}."""
    deputados = []
    url = f"{API}/deputados?idLegislatura={legislatura}&ordem=ASC&ordenarPor=nome&itens=100"
    while url:
        data = get_json(url)
        deputados.extend(data.get("dados", []))
        url = _proximo_link(data)
        time.sleep(0.2)
    return deputados


def coletar_discursos(id_dep, data_inicio=None, data_fim=None):
    """Generator de discursos (dicts) de um deputado."""
    params = {"itens": 100, "ordenarPor": "dataHoraInicio", "ordem": "ASC"}
    if data_inicio:
        params["dataInicio"] = data_inicio
    if data_fim:
        params["dataFim"] = data_fim
    url = f"{API}/deputados/{id_dep}/discursos?" + urllib.parse.urlencode(params)
    while url:
        data = get_json(url)
        for d in data.get("dados", []):
            yield d
        url = _proximo_link(data)
        time.sleep(0.2)


def _proximo_link(data):
    for l in data.get("links", []):
        if l.get("rel") == "next":
            return l.get("href")
    return None


def carregar_mapa(caminho):
    """Le o CSV partido->ideologia. Usa a coluna eixo_binario (vazia = descartar)."""
    mapa = {}
    with open(caminho, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eixo = (row.get("eixo_binario") or "").strip()
            if eixo:
                mapa[row["partido"].strip().upper()] = eixo
    return mapa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legislatura", type=int, default=57,
                    help="57 = 2023-2027 (atual). Use 56 p/ 2019-2023, etc.")
    ap.add_argument("--data-inicio", default=None, help="AAAA-MM-DD")
    ap.add_argument("--data-fim", default=None, help="AAAA-MM-DD")
    ap.add_argument("--min-palavras", type=int, default=50,
                    help="Descarta transcricoes mais curtas (reduz ruido).")
    ap.add_argument("--max-por-deputado", type=int, default=0,
                    help="0 = sem limite. Util para amostra rapida.")
    ap.add_argument("--mapa", default="mapeamento_partidos_ideologia.csv")
    ap.add_argument("--saida", default="discursos_camara.csv")
    args = ap.parse_args()

    mapa = carregar_mapa(args.mapa)
    print(f"[i] Partidos mapeados (eixo binario): {len(mapa)}", file=sys.stderr)

    deputados = listar_deputados(args.legislatura)
    print(f"[i] Deputados na legislatura {args.legislatura}: {len(deputados)}", file=sys.stderr)

    n_total, n_gravados = 0, 0
    by_label = {"esquerda": 0, "direita": 0}

    with open(args.saida, "w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["texto", "partido", "ideologia", "id_deputado", "data"])

        for dep in deputados:
            partido = (dep.get("siglaPartido") or "").strip().upper()
            ideologia = mapa.get(partido)
            if not ideologia:
                continue  # partido de centro ou nao mapeado -> ignora
            n_dep = 0
            for disc in coletar_discursos(dep["id"], args.data_inicio, args.data_fim):
                n_total += 1
                texto = (disc.get("transcricao") or "").strip()
                if len(texto.split()) < args.min_palavras:
                    continue
                w.writerow([texto, partido, ideologia, dep["id"],
                            disc.get("dataHoraInicio", "")])
                n_gravados += 1
                by_label[ideologia] += 1
                n_dep += 1
                if args.max_por_deputado and n_dep >= args.max_por_deputado:
                    break
            print(f"  - {dep.get('nome')} ({partido}/{ideologia}): {n_dep} discursos",
                  file=sys.stderr)

    print(f"\n[OK] Discursos vistos: {n_total} | gravados: {n_gravados}", file=sys.stderr)
    print(f"[OK] Distribuicao: {by_label}", file=sys.stderr)
    print(f"[OK] Arquivo: {args.saida}", file=sys.stderr)


if __name__ == "__main__":
    main()
