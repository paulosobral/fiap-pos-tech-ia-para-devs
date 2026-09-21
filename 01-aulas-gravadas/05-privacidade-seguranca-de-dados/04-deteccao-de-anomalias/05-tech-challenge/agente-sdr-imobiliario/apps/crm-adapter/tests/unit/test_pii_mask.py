import re

import pytest

from service.pii_mask import mask_pii


def test_mask_email():
    assert mask_pii("contato ana@empresa.com agora") == "contato [EMAIL] agora"


def test_mask_phone():
    assert mask_pii("ligar +5511999990000 hoje") == "ligar [TELEFONE] hoje"


def test_mask_full_name():
    assert mask_pii("Ana Ribeiro quer sala") == "[NOME] quer sala"


def test_mask_single_name_only_from_controlled_list():
    assert mask_pii("Maria pediu área") == "[NOME] pediu área"
    # Fora da lista controlada não é mascarado (mesmo comportamento da u1).
    assert "Zeca" in mask_pii("Zeca pediu área")


def test_mask_cnpj():
    assert mask_pii("cnpj 12.345.678/0001-90 ok") == "cnpj [CNPJ] ok"


def test_non_pii_text_unchanged():
    assert mask_pii("budget R$ 800.000 area 150 m2") == "budget R$ 800.000 area 150 m2"


def test_mask_idempotent_on_placeholders():
    masked = mask_pii("Ana Ribeiro ana@empresa.com")
    assert mask_pii(masked) == masked
    assert "[EMAIL]" in masked
