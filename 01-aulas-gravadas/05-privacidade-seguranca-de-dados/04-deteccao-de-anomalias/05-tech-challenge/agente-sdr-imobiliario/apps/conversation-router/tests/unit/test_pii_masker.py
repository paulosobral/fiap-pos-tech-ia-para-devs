from unittest.mock import MagicMock

from service.security_layer import SecurityLayer


class TestPiiMasker:
    def setup_method(self):
        self.layer = SecurityLayer()

    def test_masks_email(self):
        out = self.layer.mask("Meu e-mail é joao@empresa.com")
        assert "joao@empresa.com" not in out
        assert "[EMAIL]" in out

    def test_masks_name(self):
        out = self.layer.mask("Meu nome é João Silva")
        assert "João Silva" not in out
        assert "[NOME]" in out

    def test_masks_single_known_first_name(self):
        out = self.layer.mask("Fale com o Pedro sobre o imóvel")
        assert "Pedro" not in out
        assert "[NOME]" in out

    def test_common_words_not_masked_as_name(self):
        out = self.layer.mask("Podemos enviar a proposta amanhã")
        assert out == "Podemos enviar a proposta amanhã"

    def test_masks_phone(self):
        out = self.layer.mask("Ligue no +55 11 91234-5678")
        assert "91234-5678" not in out
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

    def test_output_leak_blocks_first_name(self):
        leak, label = self.layer.check_output_leak("Olá João, segue a proposta.")
        assert leak is True
        assert label == "NOME"

    def test_unmask_restores(self):
        masked = self.layer.mask("Meu nome é João Silva")
        restored = self.layer.unmask(masked, {"NOME": ["João Silva"]})
        assert "João Silva" in restored
