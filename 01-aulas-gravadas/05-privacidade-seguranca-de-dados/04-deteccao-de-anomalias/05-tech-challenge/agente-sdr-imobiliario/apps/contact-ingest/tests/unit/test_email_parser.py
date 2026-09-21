import base64

from service.email_parser import HeuristicEmailParser, MAX_MESSAGE_CHARS


PORTAL_BODY = (
    "Novo contato recebido pelo portal.\n"
    "Nome: Ana Souza\n"
    "E-mail: ana.souza@empresa.com.br\n"
    "Telefone: (11) 98888-7777\n"
    "Interesse: laje corporativa de 300 m² na Faria Lima\n"
)


def make_mail(**overrides):
    mail = {
        "messageId": "mid-1",
        "source": "no-reply@portalimob.com.br",
        "commonHeaders": {
            "from": ["Portal Imob <no-reply@portalimob.com.br>"],
            "to": ["contato@wlevitt.app"],
            "subject": "Contato via portal",
            "date": "Sat, 20 Sep 2026 12:00:00 +0000",
        },
    }
    mail.update(overrides)
    return mail


def raw_content(body: str) -> str:
    from email.message import EmailMessage

    message = EmailMessage()
    message["Subject"] = "Contato via portal"
    message["From"] = "Portal Imob <no-reply@portalimob.com.br>"
    message["To"] = "contato@wlevitt.app"
    message.set_content(body)
    return base64.b64encode(message.as_bytes()).decode()


class TestHeuristicEmailParser:
    def test_extracts_labeled_fields_from_raw_mime(self):
        parser = HeuristicEmailParser()
        parsed = parser.parse(make_mail(content=raw_content(PORTAL_BODY)))
        assert parsed is not None
        assert parsed.name == "Ana Souza"
        assert parsed.email == "ana.souza@empresa.com.br"
        assert parsed.phone == "(11) 98888-7777"
        assert "Faria Lima" in parsed.message_text

    def test_email_label_wins_over_from_header(self):
        body = "E-mail: lead@empresa.com\nMensagem: quero uma laje"
        parsed = HeuristicEmailParser().parse(make_mail(content=raw_content(body)))
        assert parsed.email == "lead@empresa.com"

    def test_falls_back_to_from_header_email(self):
        mail = make_mail(commonHeaders={"subject": "Contato via portal"})
        parsed = HeuristicEmailParser().parse(mail)
        assert parsed is not None
        assert parsed.email == "no-reply@portalimob.com.br"
        assert parsed.phone is None

    def test_name_from_subject_pattern(self):
        mail = make_mail(
            commonHeaders={
                "from": ["no-reply@portalimob.com.br"],
                "subject": "Novo lead: Bruno Carvalho",
            }
        )
        parsed = HeuristicEmailParser().parse(mail)
        assert parsed.name == "Bruno Carvalho"

    def test_missing_email_returns_none(self):
        mail = make_mail(source="", commonHeaders={"subject": "Ola", "to": ["contato@wlevitt.app"]})
        assert HeuristicEmailParser().parse(mail) is None

    def test_message_text_falls_back_to_subject(self):
        parsed = HeuristicEmailParser().parse(make_mail())
        assert parsed.message_text == "Contato via portal"

    def test_message_text_is_truncated(self):
        body = "x" * (MAX_MESSAGE_CHARS + 500)
        parsed = HeuristicEmailParser().parse(make_mail(content=raw_content(body)))
        assert len(parsed.message_text) == MAX_MESSAGE_CHARS

    def test_phone_without_ddd_is_ignored(self):
        body = "Telefone: 98888-7777\nE-mail: lead@empresa.com"
        parsed = HeuristicEmailParser().parse(make_mail(content=raw_content(body)))
        assert parsed.phone is None

    def test_phone_free_form_from_body(self):
        body = "pode ligar no (11) 3456-7890\nE-mail: lead@empresa.com"
        parsed = HeuristicEmailParser().parse(make_mail(content=raw_content(body)))
        assert parsed.phone == "(11) 3456-7890"

    def test_unreadable_content_does_not_crash(self):
        parsed = HeuristicEmailParser().parse(make_mail(content="@@@not-base64@@@"))
        assert parsed is not None
        assert parsed.email == "no-reply@portalimob.com.br"
