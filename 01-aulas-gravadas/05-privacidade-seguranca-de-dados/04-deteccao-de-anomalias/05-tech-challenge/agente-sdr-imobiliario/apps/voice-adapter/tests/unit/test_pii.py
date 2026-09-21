from service.pii import PiiMasker


class TestPiiMasker:
    def test_masks_email(self):
        masked = PiiMasker().mask("Meu e-mail é joao@empresa.com")
        assert "[EMAIL]" in masked
        assert "joao@empresa.com" not in masked

    def test_masks_phone(self):
        masked = PiiMasker().mask("Ligue no +55 11 91234-5678")
        assert "[TELEFONE]" in masked
        assert "91234-5678" not in masked

    def test_masks_name(self):
        masked = PiiMasker().mask("Aqui é João Silva falando")
        assert "[NOME]" in masked
        assert "João Silva" not in masked

    def test_masks_cnpj(self):
        masked = PiiMasker().mask("CNPJ 12.345.678/0001-90")
        assert "[CNPJ]" in masked
        assert "12.345.678/0001-90" not in masked

    def test_plain_text_unchanged(self):
        text = "Quero um espaço para 20 pessoas no centro"
        assert PiiMasker().mask(text) == text

    def test_placeholders_are_not_remasked(self):
        masker = PiiMasker()
        masked = masker.mask("Meu e-mail é joao@empresa.com, sou João Silva")
        assert masker.mask(masked) == masked
