#!/usr/bin/env python3
"""Seed do catálogo de clientes (Base 2 do RAG — PRD §8.2).

Gera 50 clientes sintéticos com status Kanban (pré-atendimento → visita →
proposta → fechamento → pós-venda), região, intenção e ticket — para
simular o CRM e enriquecer a qualificação do lead.
"""
import json
import random

random.seed(260911)

REGIONS = [
    "Faria Lima", "Paulista", "Berrini", "Vila Olímpia", "Itaim Bibi",
    "Brooklin", "Pinheiros", "Alphaville", "Centro", "Chácara Santo Antônio",
    "Moema", "Morumbi",
]
INTENTS = ["rent", "purchase", "investment"]
STATUSES = ["pre-atendimento", "visita", "proposta", "fechamento", "pos-venda"]
NAMES = [
    "TechCorp", "Innova Partners", "Vortex Logística", "Helix Biotecnologia",
    "Zenith Consultoria", "Orbit Capital", "Nexus Engenharia", "Stellar Group",
    "Atlas Imóveis", "Pioneiro Holdings", "Vanguard Coworking", "Meridian Office",
    "Apex Corporate", "Lumen Arquitetura", "Sigma Tecnologia", "Delta Serviços",
    "Cresta Advisory", "Fortis Energia", "Vela Sports", "Aurora Health",
    "Brisa Eventos", "Cobalto Mineração", "Dália Cosméticos", "Éter Telecom",
    "Fênix Recursos", "Gália Alimentos", "Hélios Solar", "Íris Marketing",
    "Jade Design", "Karma Studio", "Líber Fintech", "Magnus Seguros",
    "Nimbus Cloud", "Ônix Transportes", "Prisma Análise", "Quasar Labs",
    "Rigel Manufatura", "Sírius Educação", "Tântalo Indústria", "Úrano Mineração",
    "Vértice Projetos", "Watt Energia", "Xodo Contábil", "Yoga Wellness",
    "Zênith Investimentos", "Ágil Logística", "Brisa Coworking", "Certo Assessoria",
    "Domo Arquitetura", "Eco Sustentável",
]


def make_client(i: int) -> dict:
    region = REGIONS[i % len(REGIONS)]
    intent = INTENTS[i % len(INTENTS)]
    status = STATUSES[i % len(STATUSES)]
    area = random.choice([80, 120, 200, 350, 500, 800, 1000, 1500])
    if intent == "rent":
        ticket = round(random.uniform(60, 220) * area)
        ticket_text = f"R$ {ticket / 1000:.1f} mil/mês"
    else:
        ticket = round(random.uniform(8000, 16000) * area / 1000) * 1000
        ticket_text = f"R$ {ticket / 1e6:.1f} milhão"
    return {
        "id": f"cl-{i + 1:03d}",
        "name": f"{NAMES[i % len(NAMES)]} {i + 1:02d}",
        "region": region,
        "intent": intent,
        "status": status,
        "area_m2": area,
        "ticket": ticket,
        "ticket_text": ticket_text,
        "contact_count": random.randint(1, 8),
        "last_contact_days_ago": random.randint(1, 90),
        "notes": f"Cliente histórico — {region}, {intent}, status {status}.",
    }


clients = [make_client(i) for i in range(50)]
output = {"clients": clients, "generated_at": "2026-09-11T01:30:00Z", "count": len(clients)}
print(json.dumps(output, ensure_ascii=False, indent=2))
