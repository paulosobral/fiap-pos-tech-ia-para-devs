#!/usr/bin/env python3
"""Mescla múltiplos output/*.json de crawls (uma cidade por arquivo) em um só,
removendo duplicados por "id". Uso:

    python3 scripts/merge_properties.py output/properties.json output/abc/*.json \\
        -o output/properties_merged.json
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="arquivos JSON a mesclar, em ordem de prioridade")
    parser.add_argument("-o", "--output", required=True, help="arquivo JSON de saída")
    args = parser.parse_args()

    seen: set[str] = set()
    merged: list[dict] = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for prop in data["properties"]:
            if prop["id"] in seen:
                continue
            seen.add(prop["id"])
            merged.append(prop)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "properties": merged}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"{len(merged)} imóveis mesclados em {args.output}")


if __name__ == "__main__":
    main()
