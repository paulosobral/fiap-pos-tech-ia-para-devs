#!/usr/bin/env python3
"""Remove de cada imóvel as fotos genéricas da própria imobiliária.

O CMS da Gonçalves anexa, como foto de ~1/3 dos anúncios, uma colagem das
fachadas do escritório (às vezes como única foto, às vezes no meio/fim das
fotos reais). Ela é servida de /Imoveis/<codigo>/ como qualquer foto, então
nenhum filtro por URL pega: a detecção é por conteúdo (correlação da miniatura
em cinza 32x32, sem margens brancas, com `generic_image_signature.json`).

Fotos que não puderem ser baixadas são MANTIDAS (na dúvida, não apaga).
Imóvel que ficar sem foto sai com `images: []` — o bot então não promete foto.

Uso (entre o merge e o cp para o bot):

    python3 scripts/drop_generic_images.py output/properties_merged.json \\
        -o output/properties_clean.json
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PIL import Image

warnings.filterwarnings("ignore", category=DeprecationWarning)

HERE = Path(__file__).resolve().parent
DEFAULT_SIGNATURE = HERE / "generic_image_signature.json"
DEFAULT_CACHE = HERE.parent / "output" / ".generic_image_cache.json"
PLACEHOLDER_MARKERS = ("/Front/img/house",)  # ícone cinza "sem foto" do Imoview
_WHITE = 245
_SIZE = 32
BROKEN = -2.0  # marca no cache: URL morta (404/410) — não adianta mandar pro Telegram

Scorer = Callable[[str], "float | None"]


def base_url(url: str) -> str:
    return url.split("?", 1)[0]


def is_placeholder_url(url: str) -> bool:
    return any(marker in url for marker in PLACEHOLDER_MARKERS)


def gray_signature(img: Image.Image) -> list[int]:
    """Cinza 32x32 depois de cortar margens brancas (letterbox muda de variação pra variação)."""
    gray = img.convert("L")
    bbox = gray.point(lambda v: 255 if v < _WHITE else 0).getbbox()
    if bbox:
        gray = gray.crop(bbox)
    return list(gray.resize((_SIZE, _SIZE)).tobytes())


def pearson(a: list[int], b: list[int]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def load_signature(path: Path) -> tuple[list[int], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(bytes.fromhex(data["gray32_hex"])), float(data["threshold"])


def download_scorer(signature: list[int], timeout: float = 25.0) -> Scorer:
    def score(url: str) -> float | None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=timeout).read()
            return pearson(signature, gray_signature(Image.open(io.BytesIO(raw)).convert("RGB")))
        except urllib.error.HTTPError as exc:
            return BROKEN if exc.code in (404, 410) else None
        except Exception:
            return None

    return score


def clean(
    properties: list[dict],
    scorer: Scorer,
    threshold: float,
    cache: dict[str, float] | None = None,
    workers: int = 24,
) -> tuple[list[dict], dict[str, int]]:
    cache = cache if cache is not None else {}
    pending = sorted(
        {
            base_url(u)
            for p in properties
            for u in p.get("images") or []
            if not is_placeholder_url(u) and base_url(u) not in cache
        }
    )
    with ThreadPoolExecutor(workers) as pool:
        for url, value in zip(pending, pool.map(scorer, pending)):
            if value is not None:
                cache[url] = round(value, 4)

    stats = {"photos": 0, "dropped_generic": 0, "dropped_placeholder": 0, "dropped_broken": 0,
             "unchecked_kept": 0,
             "properties_changed": 0, "properties_without_photos": 0}
    out: list[dict] = []
    for prop in properties:
        kept: list[str] = []
        for url in prop.get("images") or []:
            stats["photos"] += 1
            if is_placeholder_url(url):
                stats["dropped_placeholder"] += 1
                continue
            value = cache.get(base_url(url))
            if value is None:
                stats["unchecked_kept"] += 1
            elif value == BROKEN:
                stats["dropped_broken"] += 1
                continue
            elif value >= threshold:
                stats["dropped_generic"] += 1
                continue
            kept.append(url)
        if len(kept) != len(prop.get("images") or []):
            stats["properties_changed"] += 1
        if not kept:
            stats["properties_without_photos"] += 1
        out.append({**prop, "images": kept})
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="JSON de imóveis (saída do merge_properties.py)")
    parser.add_argument("-o", "--output", required=True, help="JSON de saída, sem as fotos genéricas")
    parser.add_argument("--signature", default=str(DEFAULT_SIGNATURE))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="cache url->correlação (evita rebaixar)")
    parser.add_argument("--threshold", type=float, default=None, help="sobrescreve o limiar da assinatura")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    signature, default_threshold = load_signature(Path(args.signature))
    threshold = args.threshold if args.threshold is not None else default_threshold
    cache_path = Path(args.cache)
    cache: dict[str, float] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cleaned, stats = clean(data["properties"], download_scorer(signature), threshold, cache, args.workers)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    Path(args.output).write_text(
        json.dumps({**data, "properties": cleaned}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(cleaned)} imóveis | {stats['photos']} fotos | limiar {threshold}")
    for key, value in stats.items():
        if key != "photos":
            print(f"  {key}: {value}")
    if stats["unchecked_kept"]:
        print("  AVISO: fotos que não baixaram foram mantidas; rode de novo para completar o cache.")


if __name__ == "__main__":
    main()
