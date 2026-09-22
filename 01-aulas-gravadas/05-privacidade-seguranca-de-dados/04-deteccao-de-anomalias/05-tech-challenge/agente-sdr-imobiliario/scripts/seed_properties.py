"""Seed do catálogo sintético de imóveis (PRD FR-11, §8.2).

Regenerável: python scripts/seed_properties.py > data/properties.json
Dados SINTÉTICOS e calibrados — corporativo comercial SP (escritórios/lajes/
conjuntos). Os nomes de condomínios/endereços são fictícios; os tickets seguem
faixas de mercado (aluguel R$ 60-220/m², venda R$ 8-16k/m²).
"""

import json
import random
import sys

random.seed(260911)

REGIONS = [
    "Faria Lima",
    "Paulista",
    "Berrini",
    "Vila Olímpia",
    "Itaim Bibi",
    "Brooklin",
    "Pinheiros",
    "Alphaville",
    "Centro",
    "Chácara Santo Antônio",
    "Moema",
    "Morumbi",
]
NEIGHBOURS = {
    "Faria Lima": ["Itaim Bibi", "Vila Olímpia"],
    "Berrini": ["Brooklin", "Chácara Santo Antônio"],
    "Paulista": ["Bela Vista", "Consolação"],
    "Vila Olímpia": ["Itaim Bibi", "Faria Lima"],
    "Itaim Bibi": ["Faria Lima", "Vila Olímpia"],
    "Brooklin": ["Berrini", "Chácara Santo Antônio"],
    "Pinheiros": ["Itaim Bibi", "Vila Madalena"],
    "Alphaville": ["Tamboré"],
    "Centro": ["República", "Sé"],
    "Chácara Santo Antônio": ["Berrini", "Brooklin"],
    "Moema": ["Vila Nova Conceição"],
    "Morumbi": ["Chácara Santo Antônio"],
}
BUILDINGS = [
    ("Torre Office One", "laje corporativa", "Platinum"),
    ("Corporate Faria Lima", "laje corporativa", "Plus"),
    ("Business Center Berrini", "conjunto comercial", "Standard"),
    ("Edifício Paulista Premium", "escritório", "Platinum"),
    ("Vila Olímpia Executive", "conjunto comercial", "Plus"),
    ("Brooklin Prime Tower", "laje corporativa", "Plus"),
    ("Centro Empresarial Pinheiros", "escritório", "Standard"),
    ("Alphaville Corporate", "galpão corporativo", "Standard"),
    ("Morada Corporate Center", "escritório", "Plus"),
    ("Moema Business Point", "conjunto comercial", "Standard"),
]
TYPE_KEYS = {
    "laje corporativa": "laje",
    "conjunto comercial": "conjunto",
    "escritório": "escritorio",
    "galpão corporativo": "galpao",
}


def make_property(i: int) -> dict:
    name, ptype, class_ = BUILDINGS[i % len(BUILDINGS)]
    region = REGIONS[i % len(REGIONS)]
    area = random.choice(
        [60, 80, 100, 120, 150, 180, 220, 280, 350, 420, 500, 650, 800, 1000]
    )
    mode = "rent" if random.random() < 0.75 else "purchase"
    if mode == "rent":
        per_m2 = random.uniform(60, 220)
        price = round(per_m2 * area)
        price_text = f"R$ {price / 1000:.1f} mil/mês"
        condominio = round(random.uniform(18, 45) * area)
    else:
        per_m2 = random.uniform(8000, 16000)
        price = round(per_m2 * area / 1000) * 1000
        price_text = f"R$ {price / 1e6:.1f} milhão"
        condominio = round(random.uniform(18, 45) * area)
    title = f"{name} {i + 1:02d} — {ptype} em {region}"
    return {
        "id": f"pr-{i + 1:03d}",
        "title": title,
        "type": TYPE_KEYS[ptype],
        "class": "A" if class_ in ("Platinum", "Plus") else "B",
        "region": region,
        "regions": NEIGHBOURS[region],
        "corredor": "Faria Lima/Vila Olímpia/Berrini",
        "area_util": area,
        "area_bruta": round(area * 1.25),
        "laje": ptype == "laje corporativa",
        "condominio_m2": round(condominio / max(area, 1)),
        "vagas": max(1, round(area / 60)),
        "andar": random.randint(1, 24),
        "elevadores": random.choice([2, 3, 4, 5, 6]),
        "entrega": random.choice(["imediata", "30 dias", "60 dias", "90 dias", "junho/2027"]),
        "cep": f"{random.randint(10000, 57999):05d}-{random.randint(100, 999):03d}",
        "disponibilidade": random.choice(["disponível", "reservado", "disponível", "entrega programada"]),
        "mode": mode,
        "price": price,
        "price_text": price_text,
        "description": (
            f"{ptype.title()} de {area} m² em {region}, classe "
            f"{'A' if class_ in ('Platinum', 'Plus') else 'B'}, "
            f"ticket {price_text}. {random.randint(0, 4) + 1} vagas, 24h, próximo a metrô."
        ),
    }


properties = [make_property(i) for i in range(120)]
sys.stdout.write(json.dumps({"version": 1, "properties": properties}, ensure_ascii=False, indent=2) + "\n")