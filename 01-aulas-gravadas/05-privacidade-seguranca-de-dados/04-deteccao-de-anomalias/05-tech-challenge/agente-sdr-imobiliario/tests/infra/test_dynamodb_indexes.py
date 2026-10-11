"""Guarda de infra: todo índice DynamoDB consultado pelo código precisa existir no Terraform.

Os testes unitários usam DynamoDB falso, que aceita qualquer `IndexName`; por isso um índice
inexistente só estourava em produção (`ValidationException: The table does not have the specified
index`). Aqui o código é comparado com as definições de `infra/dynamodb.tf`.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TF = (ROOT / "infra" / "dynamodb.tf").read_text(encoding="utf-8")

# (arquivo que consulta o índice, tabela que ele consulta, índice)
USAGES = [
    ("apps/anomaly-detector/infra/alert_store.py", "alerts", "lead-index"),
    ("apps/conversation-router/service/restriction.py", "alerts", "lead-index"),
    ("apps/conversation-router/infra/session_store.py", "sessions", "lead-index"),
    ("apps/conversation-router/infra/session_store.py", "sessions", "telegram-user-index"),
    ("apps/voice-adapter/infra/session_store.py", "sessions", "telegram-user-index"),
]


def indexes_by_table() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for m in re.finditer(r'resource "aws_dynamodb_table" "(\w+)" \{(.*?)\n\}\n', TF, re.S):
        out[m.group(1)] = set(re.findall(r'global_secondary_index \{\s*name\s*=\s*"([^"]+)"', m.group(2)))
    return out


def test_terraform_declares_the_tables_we_check():
    assert {"sessions", "alerts"} <= set(indexes_by_table())


def test_every_index_queried_by_the_code_exists_on_the_right_table():
    tables = indexes_by_table()
    missing = []
    for path, table, index in USAGES:
        source = (ROOT / path).read_text(encoding="utf-8")
        assert f'"{index}"' in source, f"{path} não usa mais o índice {index}: atualize USAGES"
        if index not in tables[table]:
            missing.append(f"{path} consulta '{index}' na tabela '{table}', que só tem {sorted(tables[table])}")
    assert not missing, "\n".join(missing)


def test_index_keys_are_declared_as_attributes():
    for m in re.finditer(r'resource "aws_dynamodb_table" "(\w+)" \{(.*?)\n\}\n', TF, re.S):
        body = m.group(2)
        attrs = set(re.findall(r'attribute \{\s*name\s*=\s*"([^"]+)"', body))
        keys = set(re.findall(r'(?:hash_key|range_key)\s*=\s*"([^"]+)"', body))
        assert keys <= attrs, f"{m.group(1)}: chave(s) sem attribute: {sorted(keys - attrs)}"


def test_no_unused_usage_entries_hide_new_index_calls():
    """Se alguém passar a consultar um índice novo, este teste obriga a registrar em USAGES."""
    known = {(p, i) for p, _, i in USAGES}
    for path in (ROOT / "apps").rglob("*.py"):
        rel = str(path.relative_to(ROOT))
        if "/tests/" in rel or ".venv" in rel:
            continue
        for index in re.findall(r'IndexName\s*[=:]\s*"([^"]+)"', path.read_text(encoding="utf-8")):
            assert (rel, index) in known, f"{rel} consulta o índice '{index}' que não está em USAGES"
