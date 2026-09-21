from service.feature_extractor import (
    FEATURE_HOURS,
    FEATURE_LENGTH,
    FEATURE_SENTIMENT,
    FEATURE_VOLUME,
    ConversationFeatureExtractor,
)


def message(text="", at=None):
    out = {"role": "lead"}
    if text is not None:
        out["text"] = text
    if at is not None:
        out["at"] = at
    return out


def legacy_message(text, ts):
    return {"role": "lead", "text": text, "ts": ts}


def make_extractor():
    return ConversationFeatureExtractor()


class TestConversationFeatureExtractor:
    def test_extracts_volume_and_average_length(self):
        conv = {"messages": [message("Quero ver apartamentos"), message("Bom dia, tem 3 quartos?")]}
        features = make_extractor().extract(conv)
        assert features[FEATURE_VOLUME] == 2
        assert features[FEATURE_LENGTH] == round((len("Quero ver apartamentos") + len("Bom dia, tem 3 quartos?")) / 2, 4)

    def test_negative_sentiment_ratio_by_keywords(self):
        conv = {"messages": [message("Isso é um absurdo"), message("Obrigado!"), message("Que atendimento horrível")]}
        features = make_extractor().extract(conv)
        assert features[FEATURE_SENTIMENT] == round(2 / 3, 4)

    def test_generic_domain_terms_are_not_negative(self):
        conv = {
            "messages": [
                message("qual o processo de compra?"),
                message("quero cancelar a visita de amanhã"),
                message("posso remarcar o horário?"),
            ]
        }
        features = make_extractor().extract(conv)
        assert features[FEATURE_SENTIMENT] == 0.0

    def test_atypical_hours_reads_real_u1_at_field_in_local_timezone(self):
        conv = {
            "messages": [
                message("oi", at="2026-09-20T03:30:00+00:00"),
                message("bom dia", at="2026-09-20T13:00:00+00:00"),
                message("ainda aí?", at="2026-09-20T23:45:00+00:00"),
            ]
        }
        features = make_extractor().extract(conv)
        assert features[FEATURE_HOURS] == round(2 / 3, 4)

    def test_business_window_boundary_19_brt_is_atypical(self):
        conv = {
            "messages": [
                message("bom dia", at="2026-09-20T13:00:00+00:00"),
                message("tchau", at="2026-09-20T22:00:00+00:00"),
            ]
        }
        features = make_extractor().extract(conv)
        assert features[FEATURE_HOURS] == 0.5

    def test_real_u1_message_shape_is_scored(self):
        conv = {
            "messages": [
                {"role": "lead", "text": "tenho interesse", "at": "2026-09-20T13:00:00+00:00"},
                {"role": "agent", "text": "bom dia!", "at": "2026-09-20T13:01:00+00:00"},
            ]
        }
        features = make_extractor().extract(conv)
        assert features[FEATURE_HOURS] == 0.0
        assert features[FEATURE_VOLUME] == 2

    def test_at_field_takes_precedence_over_ts(self):
        conv = {
            "messages": [
                {
                    "role": "lead",
                    "text": "oi",
                    "at": "2026-09-20T13:00:00+00:00",
                    "ts": "2026-09-20T03:30:00+00:00",
                }
            ]
        }
        features = make_extractor().extract(conv)
        assert features[FEATURE_HOURS] == 0.0

    def test_legacy_ts_field_still_supported(self):
        conv = {
            "messages": [
                legacy_message("oi", "2026-09-20T03:30:00+00:00"),
                legacy_message("bom dia", "2026-09-20T13:00:00+00:00"),
            ]
        }
        features = make_extractor().extract(conv)
        assert features[FEATURE_HOURS] == 0.5

    def test_naive_timestamp_is_interpreted_as_utc(self):
        conv = {"messages": [message("texto", at="2026-09-20T03:30:00")]}
        features = make_extractor().extract(conv)
        assert features[FEATURE_HOURS] == 1.0

    def test_empty_conversation_yields_zero_features(self):
        features = make_extractor().extract({"messages": []})
        assert features == {FEATURE_VOLUME: 0, FEATURE_LENGTH: 0.0, FEATURE_SENTIMENT: 0.0, FEATURE_HOURS: 0.0}

    def test_unparseable_timestamps_are_ignored(self):
        conv = {"messages": [message("texto", at="not-a-date"), message("outro", at="")]}
        features = make_extractor().extract(conv)
        assert features[FEATURE_HOURS] == 0.0

    def test_missing_or_non_list_messages_are_tolerated(self):
        assert make_extractor().extract({}) [FEATURE_VOLUME] == 0
        assert make_extractor().extract({"messages": "oops"}) [FEATURE_VOLUME] == 0

    def test_non_string_and_non_dict_messages_do_not_break_extraction(self):
        conv = {"messages": [message("válido"), None, "solto", {"content": None}, {"other": 1}]}
        features = make_extractor().extract(conv)
        assert features[FEATURE_VOLUME] == 5
        assert features[FEATURE_LENGTH] == len("válido")

    def test_extraction_is_deterministic(self):
        conv = {"messages": [message("golpe", at="2026-09-20T02:00:00+00:00")] * 3}
        first = make_extractor().extract(conv)
        second = make_extractor().extract(conv)
        assert first == second
