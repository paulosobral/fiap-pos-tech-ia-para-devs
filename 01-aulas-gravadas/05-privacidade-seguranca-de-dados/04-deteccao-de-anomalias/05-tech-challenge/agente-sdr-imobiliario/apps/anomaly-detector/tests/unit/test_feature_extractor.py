from service.feature_extractor import (
    FEATURE_HOURS,
    FEATURE_LENGTH,
    FEATURE_SENTIMENT,
    FEATURE_VOLUME,
    ConversationFeatureExtractor,
)


def message(text="", ts=None):
    out = {"role": "lead"}
    if text is not None:
        out["text"] = text
    if ts is not None:
        out["ts"] = ts
    return out


def make_extractor():
    return ConversationFeatureExtractor()


class TestConversationFeatureExtractor:
    def test_extracts_volume_and_average_length(self):
        conv = {"messages": [message("Quero ver apartamentos"), message("Bom dia, tem 3 quartos?")]}
        features = make_extractor().extract(conv)
        assert features[FEATURE_VOLUME] == 2
        assert features[FEATURE_LENGTH] == round((len("Quero ver apartamentos") + len("Bom dia, tem 3 quartos?")) / 2, 4)

    def test_negative_sentiment_ratio_by_keywords(self):
        conv = {"messages": [message("Isso é um absurdo, quero cancelar"), message("Obrigado!"), message("Que atendimento horrível")]}
        features = make_extractor().extract(conv)
        assert features[FEATURE_SENTIMENT] == round(2 / 3, 4)

    def test_atypical_hours_outside_business_window(self):
        conv = {
            "messages": [
                message("oi", ts="2026-09-20T03:30:00+00:00"),
                message("bom dia", ts="2026-09-20T10:00:00+00:00"),
                message("ainda aí?", ts="2026-09-20T23:45:00+00:00"),
            ]
        }
        features = make_extractor().extract(conv)
        assert features[FEATURE_HOURS] == round(2 / 3, 4)

    def test_empty_conversation_yields_zero_features(self):
        features = make_extractor().extract({"messages": []})
        assert features == {FEATURE_VOLUME: 0, FEATURE_LENGTH: 0.0, FEATURE_SENTIMENT: 0.0, FEATURE_HOURS: 0.0}

    def test_unparseable_timestamps_are_ignored(self):
        conv = {"messages": [message("texto", ts="not-a-date"), message("outro", ts="")]}
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
        conv = {"messages": [message("golpe", ts="2026-09-20T02:00:00+00:00")] * 3}
        first = make_extractor().extract(conv)
        second = make_extractor().extract(conv)
        assert first == second
