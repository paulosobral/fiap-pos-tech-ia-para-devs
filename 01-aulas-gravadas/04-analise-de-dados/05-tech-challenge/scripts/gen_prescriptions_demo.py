#!/usr/bin/env python3
"""Gera o CSV de demonstração "sem anomalias" da aba Prescrições.

Produz ``data/prescricoes_sinteticas_normal.csv``, um histórico de
prescrições para os mesmos 3 pacientes (mesmos nomes/medicamentos) do
dataset sintético existente ``data/prescricoes_sinteticas.csv``, mas
construído para NÃO conter nenhum dos 3 padrões que o prompt de revisão
via AWS Bedrock (``prescriptions.bedrock_review.build_review_prompt``)
pede para detectar:

- mudança abrupta de dose: cada medicamento mantém a MESMA dose em todo o
  histórico do paciente (dose constante).
- interação medicamentosa: nenhum paciente toma a combinação de risco
  conhecida usada no dataset anômalo (Warfarina + Aspirina).
- dose sem justificativa: decorre diretamente de dose constante — não há
  alteração de dose para exigir justificativa.

O dataset atual (``data/prescricoes_sinteticas.csv``) permanece como o
par "com anomalias" (Paciente B tem uma mudança abrupta de dose; Paciente
C toma Warfarina + Aspirina).

Diferente do gerador de sinais vitais (``scripts/gen_vital_signs_demo.py``),
este script NÃO chama o AWS Bedrock para autoverificar "zero findings":
a revisão é uma chamada de rede real e não-determinística (LLM), fora do
orçamento deste projeto para rodar a cada geração do dataset. A garantia
aqui é sobre a CONSTRUÇÃO do dataset (ausência dos padrões-alvo), não
sobre a resposta do modelo em uma execução real.

Determinismo: os pacientes, medicamentos, doses e datas são fixos
(constantes abaixo); não há aleatoriedade.

Como rodar (a partir da raiz do projeto):

    venv/bin/python scripts/gen_prescriptions_demo.py
"""
from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "data" / "prescricoes_sinteticas_normal.csv"

COLUMNS = ["paciente", "medicamento", "dose", "data"]

# Mesmos pacientes/medicamentos do dataset anômalo, dose constante em cada
# um (sem mudança abrupta), sem a combinação de risco Warfarina+Aspirina.
ROWS = [
    ("Paciente A", "Losartana 50mg", "50mg", "2026-01-05"),
    ("Paciente A", "Losartana 50mg", "50mg", "2026-01-19"),
    ("Paciente A", "Losartana 50mg", "50mg", "2026-02-02"),
    ("Paciente A", "Losartana 50mg", "50mg", "2026-02-16"),
    ("Paciente B", "Metformina 500mg", "500mg", "2026-01-03"),
    ("Paciente B", "Metformina 500mg", "500mg", "2026-01-17"),
    ("Paciente B", "Metformina 500mg", "500mg", "2026-01-31"),
    ("Paciente B", "Metformina 500mg", "500mg", "2026-02-14"),
    ("Paciente C", "Losartana 50mg", "50mg", "2026-01-10"),
    ("Paciente C", "Losartana 50mg", "50mg", "2026-01-24"),
    ("Paciente C", "Losartana 50mg", "50mg", "2026-02-07"),
    ("Paciente C", "Losartana 50mg", "50mg", "2026-02-21"),
]


def main() -> int:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        writer.writerows(ROWS)

    print(f"Escrito {OUTPUT_CSV} com {len(ROWS)} linha(s), "
          f"{len({row[0] for row in ROWS})} paciente(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
