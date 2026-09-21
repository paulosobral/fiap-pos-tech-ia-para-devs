from unittest.mock import MagicMock

import pytest

from infra.dedupe_store import DedupeError, DedupeStore


def conditional_failure():
    return type("ConditionalCheckFailedException", (Exception,), {})


def make_store(put_side_effect=None):
    client = MagicMock()
    client.exceptions.ConditionalCheckFailedException = conditional_failure()
    if put_side_effect is not None:
        client.put_item.side_effect = put_side_effect
    return DedupeStore(client, "sdr-ingest-dedupe"), client


class TestDedupeStore:
    def test_first_put_returns_true_and_marks_message(self):
        store, client = make_store()
        assert store.put_first("mid-1", "portalimob.com.br") is True
        _, kwargs = client.put_item.call_args
        assert kwargs["TableName"] == "sdr-ingest-dedupe"
        assert kwargs["Item"]["message_id"] == {"S": "mid-1"}
        assert kwargs["Item"]["source"] == {"S": "portalimob.com.br"}
        assert "received_at" in kwargs["Item"]
        assert kwargs["ConditionExpression"] == "attribute_not_exists(message_id)"

    def test_duplicate_put_returns_false(self):
        store, client = make_store()
        client.put_item.side_effect = client.exceptions.ConditionalCheckFailedException()
        assert store.put_first("mid-1") is False

    def test_missing_source_omits_field(self):
        store, client = make_store()
        store.put_first("mid-1")
        _, kwargs = client.put_item.call_args
        assert "source" not in kwargs["Item"]

    def test_client_error_raises_dedupe_error(self):
        store, _ = make_store(put_side_effect=RuntimeError("dynamo down"))
        with pytest.raises(DedupeError):
            store.put_first("mid-1")

    def test_delete_removes_mark(self):
        store, client = make_store()
        assert store.delete("mid-1") is True
        client.delete_item.assert_called_once_with(
            TableName="sdr-ingest-dedupe", Key={"message_id": {"S": "mid-1"}}
        )

    def test_delete_error_returns_false(self):
        store, client = make_store()
        client.delete_item.side_effect = RuntimeError("dynamo down")
        assert store.delete("mid-1") is False
