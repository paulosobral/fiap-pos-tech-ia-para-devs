#!/usr/bin/env python3
"""Gera os dois CSVs de demonstração da aba Sinais Vitais.

Produz, a partir da amostra MIMIC-III real do projeto
(``data/vital_signs_sample.csv``), dois arquivos didáticos:

- ``data/vital_signs_sample_normal.csv`` — trecho estável, SEM anomalias em
  nenhuma das duas camadas de detecção (rolling z-score e Isolation Forest);
  toda leitura sai como ``normal`` ao ser processada por
  ``vital_signs.analysis.analyze``.
- ``data/vital_signs_sample_anomalias.csv`` — o mesmo trecho com alguns
  episódios clínicos plausíveis injetados (taquicardia, hipoxemia, pico
  hipertensivo), grandes/agudos o bastante para que AMBAS as camadas
  concordem (``alta_confianca``) nas leituras de pico, mantendo as leituras
  ao redor ``normal``.

Como rodar (a partir da raiz do projeto):

    venv/bin/python scripts/gen_vital_signs_demo.py

Ao final o script roda ``analyze`` sobre cada CSV nos parâmetros PADRÃO do
app e imprime a contagem de rótulos; sai com código != 0 se o arquivo normal
tiver qualquer leitura não-``normal`` ou o anômalo não tiver ao menos uma
``alta_confianca``. Assim os CSVs são reproduzíveis e a garantia
"limpo vs. anômalo" é checável só rodando o script.

--------------------------------------------------------------------------
COMO O TRECHO NORMAL É CONSTRUÍDO (critério de D1)
--------------------------------------------------------------------------
O detector cria uma tensão: no z-score com ``DEFAULT_WINDOW=13``, QUALQUER
leitura isolada que destoe do resto da sua janela alcança ``|z| ~ sqrt(12)
≈ 3.46 > DEFAULT_THRESHOLD=3.0`` e é marcada — ou seja, patamares
constantes com degraus disparam o z-score nas bordas. Já o Isolation Forest
tem ``contamination=0.05`` fixo e, em qualquer série contínua "bem
comportada", marca ~5% das linhas como outlier; ele só marca ZERO quando os
escores de anomalia empatam no limiar, o que exige que muitas linhas
(vetores das 5 colunas) se REPITAM.

Nenhum trecho contíguo bruto do MIMIC satisfaz as duas camadas ao mesmo
tempo (o IF sempre marca ~6 de 120 linhas). A saída, mínima e documentada,
é derivar um pequeno POOL de vetores-leitura reais do trecho estável
(percentis 0.30..0.70 de cada coluna, dentro de faixas fisiológicas do
próprio MIMIC) e percorrê-lo num padrão "vai-e-volta" (zigue-zague) suave:

- Poucos vetores distintos que se repetem muitas vezes → escores do IF
  empatam → IF marca ZERO.
- A oscilação é gradual e balanceada (cada leitura fica perto das vizinhas
  na janela) → nenhum ``|z|`` cruza o limiar → z-score marca ZERO.

O resultado é uma série que oscila suavemente dentro das faixas reais do
paciente, inequivocamente "sem alertas" nas duas camadas — verificado
chamando ``analyze`` nos parâmetros padrão (não no olho).

--------------------------------------------------------------------------
EVENTOS INJETADOS NO ARQUIVO ANÔMALO (D2)
--------------------------------------------------------------------------
Sobre o trecho normal, injetamos 3 leituras de pico agudo, espaçadas por
mais de uma janela, cada uma um quadro clínico plausível. São picos de uma
leitura (a "leitura de pico" capturada do episódio): num único ponto muito
acima/abaixo da sua janela, o z-score alcança ``sqrt(12) ≈ 3.46 > 3.0`` E o
IF classifica o ponto como outlier → ``alta_confianca``. As leituras ao
redor permanecem ``normal``.

Determinismo: as posições, o trecho, o pool e os valores injetados são
fixos (constantes abaixo); não há aleatoriedade. Os parâmetros de detecção
NÃO são hardcoded — são importados de ``vital_signs.analysis``
(``DEFAULT_WINDOW`` / ``DEFAULT_THRESHOLD``), então este gerador continua
correto se os padrões do app mudarem.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Rodar a partir da raiz do projeto; garante que os pacotes do app sejam
# importáveis independentemente do diretório de invocação.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vital_signs.analysis import (  # noqa: E402
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    analyze,
    load_vital_signs_csv,
)

# --- Parâmetros do gerador (fixos → determinístico) -----------------------
BASE_CSV = PROJECT_ROOT / "data" / "vital_signs_sample.csv"
NORMAL_CSV = PROJECT_ROOT / "data" / "vital_signs_sample_normal.csv"
ANOMALIAS_CSV = PROJECT_ROOT / "data" / "vital_signs_sample_anomalias.csv"

# Colunas de sinais vitais presentes na amostra (na ordem do arquivo base).
SIGNAL_COLUMNS = [
    "heart_rate",
    "spo2",
    "resp_rate",
    "systolic_bp",
    "diastolic_bp",
]

# Trecho contíguo estável escolhido da amostra (índice 0-based, 120 linhas
# = 5 dias horários). O pool de leituras é derivado deste trecho.
SLICE_START = 96
SLICE_LEN = 120

# Pool de vetores-leitura: percentis de cada coluna neste trecho estável.
# Poucos vetores (repetição → IF marca zero) tirados de dentro das faixas
# reais do paciente (realismo preservado).
POOL_QUANTILES = np.linspace(0.30, 0.70, 6)

# Timestamps sintéticos, horários e contíguos, para o trecho gerado.
DEMO_START_TS = "2147-10-20 00:00:00"

# Episódios clínicos injetados no arquivo anômalo. Cada um é uma leitura de
# pico (linha, coluna→valor), espaçados por mais de uma janela entre si.
INJECTED_EVENTS = [
    (30, {"heart_rate": 165.0}),                          # taquicardia
    (60, {"spo2": 85.0}),                                 # hipoxemia
    (90, {"systolic_bp": 210.0, "diastolic_bp": 128.0}),  # pico hipertensivo
]


def _build_normal_frame() -> pd.DataFrame:
    """Constrói o trecho normal (oscilação suave sobre um pool real)."""
    base = load_vital_signs_csv(str(BASE_CSV))
    stable = base.iloc[SLICE_START : SLICE_START + SLICE_LEN].reset_index(drop=True)

    # Pool de leituras reais (um vetor por quantil).
    pool = [
        [round(float(stable[col].quantile(q)), 1) for col in SIGNAL_COLUMNS]
        for q in POOL_QUANTILES
    ]

    # Padrão vai-e-volta determinístico: 0,1,...,k-1,k-2,...,1 e repete.
    n_pool = len(pool)
    zigzag = list(range(n_pool)) + list(range(n_pool - 2, 0, -1))
    order = [zigzag[i % len(zigzag)] for i in range(SLICE_LEN)]

    rows = [pool[i] for i in order]
    df = pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
    df.insert(
        0,
        "timestamp",
        pd.date_range(DEMO_START_TS, periods=SLICE_LEN, freq="h"),
    )
    return df


def _build_anomalias_frame(normal_df: pd.DataFrame) -> pd.DataFrame:
    """Injeta os episódios clínicos de pico sobre o trecho normal."""
    df = normal_df.copy()
    for row_index, overrides in INJECTED_EVENTS:
        for column, value in overrides.items():
            df.loc[row_index, column] = value
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")


def _label_counts(path: Path) -> dict:
    """Roda ``analyze`` nos parâmetros PADRÃO e devolve a contagem de rótulos."""
    loaded = load_vital_signs_csv(str(path))
    result = analyze(loaded, window=DEFAULT_WINDOW, threshold=DEFAULT_THRESHOLD)
    counts = result["combined_report"]["agreement"].value_counts().to_dict()
    return counts


def main() -> int:
    normal_df = _build_normal_frame()
    anomalias_df = _build_anomalias_frame(normal_df)

    _write_csv(normal_df, NORMAL_CSV)
    _write_csv(anomalias_df, ANOMALIAS_CSV)

    print(f"Parâmetros de detecção (do app): window={DEFAULT_WINDOW}, "
          f"threshold={DEFAULT_THRESHOLD}")
    print(f"Trecho base: linhas {SLICE_START}..{SLICE_START + SLICE_LEN - 1} "
          f"de {BASE_CSV.name}")

    normal_counts = _label_counts(NORMAL_CSV)
    anomalias_counts = _label_counts(ANOMALIAS_CSV)

    print(f"\n{NORMAL_CSV.name}: {normal_counts}")
    print(f"{ANOMALIAS_CSV.name}: {anomalias_counts}")

    # --- Auto-verificação ---------------------------------------------------
    ok = True
    non_normal = {k: v for k, v in normal_counts.items() if k != "normal"}
    if non_normal:
        print(f"\nFALHA: o arquivo normal tem leituras não-normais: {non_normal}")
        ok = False

    alta = anomalias_counts.get("alta_confianca", 0)
    if alta < 1:
        print("\nFALHA: o arquivo anômalo não tem nenhuma leitura "
              "'alta_confianca'.")
        ok = False

    if ok:
        print(f"\nOK: arquivo normal 100% 'normal'; arquivo anômalo com "
              f"{alta} leitura(s) 'alta_confianca'.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
