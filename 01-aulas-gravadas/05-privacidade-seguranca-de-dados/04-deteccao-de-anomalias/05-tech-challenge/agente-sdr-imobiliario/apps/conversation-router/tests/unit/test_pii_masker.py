from unittest.mock import MagicMock

from service.security_layer import SecurityLayer


class TestPiiMasker:
    def setup_method(self):
        self.layer = SecurityLayer()

    def test_masks_email(self):
        out = self.layer.mask("Meu e-mail é joao@empresa.com")
        assert "joao@empresa.com" not in out
        assert "[EMAIL]" in out

    def test_common_words_not_masked_as_name(self):
        out = self.layer.mask("Podemos enviar a proposta amanhã")
        assert out == "Podemos enviar a proposta amanhã"

    def test_masks_phone(self):
        out = self.layer.mask("Ligue no +55 11 91234-5678")
        assert "91234-5678" not in out
        assert "[TELEFONE]" in out

    def test_masks_local_phone_without_country_code(self):
        out = self.layer.mask("telefone 11979918262")
        assert "11979918262" not in out
        assert "[TELEFONE]" in out

    def test_masks_formatted_local_phone(self):
        out = self.layer.mask("telefone (11) 97991-8262")
        assert "97991-8262" not in out
        assert "[TELEFONE]" in out

    def test_masks_cnpj(self):
        out = self.layer.mask("CNPJ 12.345.678/0001-95")
        assert "12.345.678/0001-95" not in out
        assert "[CNPJ]" in out

    def test_persists_extracted_pii(self):
        store = MagicMock()
        layer = SecurityLayer(pii_store=store)
        layer.mask("e-mail joao@empresa.com", session_id="s1")
        store.save.assert_called_once()

    def test_output_leak_detection(self):
        leak, label = self.layer.check_output_leak("contato joao@empresa.com")
        assert leak is True
        assert label == "EMAIL"

    def test_output_clean_passes(self):
        leak, _ = self.layer.check_output_leak("Segue sua lista de imóveis.")
        assert leak is False

    def test_output_unresolved_pii_placeholder_is_blocked(self):
        leak, label = self.layer.check_output_leak("Ótimo, [NOME]! Envio para [EMAIL].")
        assert leak is True
        assert label == "PLACEHOLDER"

    def test_output_leak_blocks_first_name_from_session(self):
        leak, label = self.layer.check_output_leak(
            "Olá João, segue a proposta.", session_pii={"NOME": ["João"]}
        )
        assert leak is True
        assert label == "NOME"

    def test_output_first_name_not_in_session_is_not_blocked(self):
        """Nome comum que aparece no texto mas não é PII desta sessão (ex.: nome de
        bairro/empreendimento) não deve derrubar a resposta — regressão do falso
        positivo com a lista estática COMMON_FIRST_NAMES."""
        leak, _ = self.layer.check_output_leak(
            "Temos opções na região da Vila Ana.", session_pii={"NOME": ["Paulo"]}
        )
        assert leak is False

    def test_unmask_restores(self):
        masked = self.layer.mask("Meu nome é João Silva")
        restored = self.layer.unmask(masked, {"NOME": ["João Silva"]})
        assert "João Silva" in restored


import pytest as _pytest


class _MemPii:
    def __init__(self):
        self.saved = {}

    def save(self, session_id, extracted):
        self.saved.update(extracted)

    def load(self, session_id):
        return self.saved


@_pytest.mark.parametrize(
    "typed",
    [
        "(11) 9-7991-8262",  # chat real 05/10: não era mascarado nem salvo -> lead sem telefone
        "+55 11 9 7991-8262",
        "meu zap é 11 9 7991 8262",
        "11 9-7991 8262",
        "(11)9 7991-8262",
        "fone 011 9 7991 8262",
        "(11) 97991-8262",
        "11979918262",
    ],
)
def test_phone_typed_in_any_separator_style_is_masked_and_stored(typed):
    pii = _MemPii()
    from service.security_layer import SecurityLayer

    masked = SecurityLayer(pii_store=pii).mask(typed, session_id="s")
    assert "[TELEFONE]" in masked and "7991" not in masked
    assert "TELEFONE" in pii.saved


@_pytest.mark.parametrize(
    "text", ["1000 metros quadrados", "uns 15 mil por mês", "R$ 1.200.000", "cep 04240-140", "preço 450 000 000"]
)
def test_numbers_that_are_not_phones_stay_untouched(text):
    from service.security_layer import SecurityLayer

    pii = _MemPii()
    assert SecurityLayer(pii_store=pii).mask(text, session_id="s") == text
    assert "TELEFONE" not in pii.saved


def test_phone_echoed_by_the_bot_in_that_format_is_blocked():
    from service.security_layer import SecurityLayer

    leaked, label = SecurityLayer().check_output_leak("Encaminhei seu telefone (11) 9-7991-8262 ao corretor")
    assert leaked and label == "TELEFONE"


@_pytest.mark.parametrize(
    "spoken",
    [
        "onze nove sete nove nove um oito dois seis dois",
        "meu telefone é onze, nove, sete, nove, nove, um, oito, dois, seis, dois obrigado",
        "11 nove sete nove nove um oito dois seis dois",
        "onze nove sete nove nove um, oitenta e dois, sessenta e dois",
        "zero onze nove sete nove nove um oito dois seis dois",
        "Onze Nove Sete Nove Nove Um Oito Dois Seis Dois",
        "onze nove sete nove nove um oito dois seis dois dois quartos",
        "vinte e um nove oito sete seis cinco quatro três dois um",
    ],
)
def test_spoken_phone_from_audio_transcription_is_masked_and_stored_as_digits(spoken):
    from service.security_layer import SecurityLayer

    pii = _MemPii()
    masked = SecurityLayer(pii_store=pii).mask(spoken, session_id="s")
    assert "[TELEFONE]" in masked
    assert not any(w in masked.lower() for w in ("nove sete", "oito dois", "sessenta"))
    assert pii.saved["TELEFONE"] and all(n.isdigit() for n in pii.saved["TELEFONE"])


@_pytest.mark.parametrize(
    "text",
    ["dois mil e vinte", "tenho três quartos e duas vagas", "um dois três", "quero cinco, seis",
     "são quinze mil por mês", "uns mil e duzentos metros", "preciso de quarenta pessoas",
     "onze de outubro às dez horas", "quero o primeiro, segundo e terceiro", "vinte e cinco mil reais"],
)
def test_ordinary_number_words_are_not_taken_for_a_phone(text):
    from service.security_layer import SecurityLayer

    pii = _MemPii()
    assert SecurityLayer(pii_store=pii).mask(text, session_id="s") == text
    assert "TELEFONE" not in pii.saved


def test_spoken_phone_echoed_by_the_bot_is_blocked():
    from service.security_layer import SecurityLayer

    leaked, label = SecurityLayer().check_output_leak("Anotei: onze nove sete nove nove um oito dois seis dois")
    assert leaked and label == "TELEFONE"


def test_spoken_phone_in_audio_reaches_the_pii_store_digits_for_the_crm():
    from service.security_layer import SecurityLayer

    pii = _MemPii()
    SecurityLayer(pii_store=pii).mask("onze nove sete nove nove um oito dois seis dois", session_id="s")
    assert pii.saved["TELEFONE"] == ["11979918262"]


class TestOnlyContactChannelsAreMasked:
    """Máscara só para e-mail/telefone/CNPJ. Nome não é adivinhado pelo jeito de escrever:
    chat real 05/10 22:30 — 'Santo André' (colado do bot) virou NOME, foi salvo como PII e o
    bot foi bloqueado ('Não posso ajudar com isso') ao responder citando o lugar."""

    @_pytest.mark.parametrize(
        "text",
        [
            "gostei desse 1. Apartamento de 115.6 m² à venda no Centro de Santo André/SP por R$ 0.4 milhão.",
            "quero em São Bernardo do Campo, Vila Bastos",
            "Meu nome é João Silva e moro em Santo André",
            "sou o André",
            "quero ver a Torre Nova",
        ],
    )
    def test_capitalized_words_are_left_alone(self, text):
        pii = _MemPii()
        from service.security_layer import SecurityLayer

        assert SecurityLayer(pii_store=pii).mask(text, session_id="s") == text
        assert "NOME" not in pii.saved

    def test_email_and_phone_are_still_masked_in_the_same_sentence(self):
        pii = _MemPii()
        from service.security_layer import SecurityLayer

        masked = SecurityLayer(pii_store=pii).mask(
            "Sou João Silva, joao@empresa.com, (11) 9-7991-8262", session_id="s"
        )
        assert masked == "Sou João Silva, [EMAIL], [TELEFONE]"
        assert set(pii.saved) == {"EMAIL", "TELEFONE"}

    def test_bot_reply_citing_a_place_is_never_blocked_for_being_a_name(self):
        from service.security_layer import SecurityLayer

        assert SecurityLayer().check_output_leak("O apartamento em Santo André tem 115 m².") == (False, None)

    def test_profile_name_from_telegram_is_still_protected_in_the_reply(self):
        from service.security_layer import SecurityLayer

        leaked, label = SecurityLayer().check_output_leak(
            "Oi Paulo Sobral, tudo bem?", session_pii={"NOME": ["Paulo Sobral"]}
        )
        assert leaked and label == "NOME"


def test_blocked_humanized_reply_falls_back_to_the_official_text_not_to_a_dead_end():
    from unittest.mock import MagicMock

    from tests.integration.fixtures import make_router, telegram_update

    router, _, telegram = make_router()
    router.security.check_output_leak = lambda text, session_pii=None: (
        ("TELEFONE" in text, "TELEFONE") if "TELEFONE" in text else (False, None)
    )
    router.flow.invoke = lambda s: {
        **s, "current_state": "conversation",
        "response": "resposta reescrita com TELEFONE vazado",
        "official_response": "Texto oficial limpo do turno.",
    }
    router.handle(telegram_update("oi"))
    assert telegram.send_message.call_args[0][1] == "Texto oficial limpo do turno."


def test_when_even_the_official_text_leaks_the_generic_fallback_is_used():
    from tests.integration.fixtures import make_router, telegram_update

    router, _, telegram = make_router()
    router.security.check_output_leak = lambda text, session_pii=None: (True, "TELEFONE")
    router.flow.invoke = lambda s: {**s, "current_state": "conversation", "response": "x", "official_response": "y"}
    router.handle(telegram_update("oi"))
    assert telegram.send_message.call_args[0][1] == "Não posso ajudar com isso"
