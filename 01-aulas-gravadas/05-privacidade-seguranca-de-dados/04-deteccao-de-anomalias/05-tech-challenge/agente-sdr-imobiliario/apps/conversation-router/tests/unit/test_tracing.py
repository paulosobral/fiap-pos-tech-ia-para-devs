import contextlib
import sys
import types
from types import SimpleNamespace

import pytest

import handler
import server
from service import llm, tracing


class FakeEntity:
    def __init__(self):
        self.annotations = {}

    def put_annotation(self, key, value):
        self.annotations[key] = value


class FakeRecorder:
    def __init__(self):
        self.segments, self.subsegments = [], []
        self._segment, self._sub = None, None

    @contextlib.contextmanager
    def in_segment(self, name):
        self._segment = FakeEntity()
        self.segments.append((name, self._segment))
        yield self._segment
        self._segment = None

    @contextlib.contextmanager
    def in_subsegment(self, name):
        self._sub = FakeEntity()
        self.subsegments.append((name, self._sub))
        try:
            yield self._sub
        finally:
            self._sub = None

    def current_segment(self):
        return self._segment

    def current_subsegment(self):
        return self._sub


@pytest.fixture()
def fresh(monkeypatch):
    monkeypatch.setattr(tracing, "_loaded", False)
    monkeypatch.setattr(tracing, "_recorder", None)


def install_sdk(monkeypatch, recorder, patch=lambda modules: None):
    core = types.ModuleType("aws_xray_sdk.core")
    core.patch = patch

    def forbidden(*args, **kwargs):
        raise AssertionError("patch_all() instrumenta requests/urllib e grava a URL (token do bot) no trace")

    core.patch_all = forbidden
    core.xray_recorder = recorder
    pkg = types.ModuleType("aws_xray_sdk")
    pkg.core = core
    monkeypatch.setitem(sys.modules, "aws_xray_sdk", pkg)
    monkeypatch.setitem(sys.modules, "aws_xray_sdk.core", core)


def remove_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "aws_xray_sdk", None)
    monkeypatch.setitem(sys.modules, "aws_xray_sdk.core", None)


class TestTracingHelpers:
    def test_without_sdk_everything_is_a_noop(self, fresh, monkeypatch):
        remove_sdk(monkeypatch)
        assert tracing.enable() is False
        with tracing.segment("x"), tracing.subsegment("y", a=1):
            tracing.annotate("k", "v")  # não levanta

    def test_with_sdk_patches_libraries_once_and_records(self, fresh, monkeypatch):
        recorder, calls = FakeRecorder(), []
        install_sdk(monkeypatch, recorder, patch=lambda modules: calls.append(modules))
        assert tracing.enable() is True and tracing.enable() is True
        assert calls == [("botocore",)]  # só botocore: o token do bot do Telegram vai na URL
        with tracing.segment("conversation-router"):
            tracing.annotate("route", "POST /webhook/telegram")
            with tracing.subsegment("llm:deepseek", model="deepseek", max_tokens=300):
                tracing.annotate("inside", True)
        assert recorder.segments[0][0] == "conversation-router"
        assert recorder.segments[0][1].annotations == {"route": "POST /webhook/telegram"}
        name, sub = recorder.subsegments[0]
        assert name == "llm:deepseek"
        assert sub.annotations == {"model": "deepseek", "max_tokens": 300, "inside": True}

    def test_missing_segment_only_logs_instead_of_raising(self, fresh, monkeypatch):
        import os

        monkeypatch.delenv("AWS_XRAY_CONTEXT_MISSING", raising=False)
        install_sdk(monkeypatch, FakeRecorder())
        tracing.enable()
        assert os.environ["AWS_XRAY_CONTEXT_MISSING"] == "LOG_ERROR"

    def test_patch_failure_disables_tracing_without_raising(self, fresh, monkeypatch):
        def boom(modules):
            raise RuntimeError("sem daemon")

        install_sdk(monkeypatch, FakeRecorder(), patch=boom)
        assert tracing.enable() is False
        with tracing.segment("x"):
            pass

    def test_annotation_values_are_made_safe(self, fresh, monkeypatch):
        recorder = FakeRecorder()
        install_sdk(monkeypatch, recorder)
        with tracing.segment("s"):
            tracing.annotate("obj", {"a": 1})
            tracing.annotate("none", None)
        assert recorder.segments[0][1].annotations == {"obj": "{'a': 1}"}


class TestServerSegment:
    def test_request_runs_inside_a_segment_with_route_and_status(self, monkeypatch):
        seen = []

        @contextlib.contextmanager
        def fake_segment(name):
            seen.append(("segment", name))
            yield

        monkeypatch.setattr(tracing, "segment", fake_segment)
        monkeypatch.setattr(tracing, "annotate", lambda k, v: seen.append((k, v)))
        monkeypatch.setattr(handler, "handler", lambda event, ctx=None: {"statusCode": 202, "body": "ok"})
        res = server.handle_traced({"routeKey": "POST /webhook/telegram"})
        assert res["statusCode"] == 202
        assert seen == [("segment", "conversation-router"), ("route", "POST /webhook/telegram"), ("status", 202)]

    def test_handler_error_propagates_so_the_server_returns_500(self, monkeypatch):
        def broken(event, ctx=None):
            raise RuntimeError("falhou")

        monkeypatch.setattr(handler, "handler", broken)
        with pytest.raises(RuntimeError):
            server.handle_traced({"routeKey": "GET /health"})


class TestLlmSpans:
    def _record(self, monkeypatch):
        spans, notes = [], []

        @contextlib.contextmanager
        def fake_sub(name, **ann):
            spans.append((name, ann))
            yield

        monkeypatch.setattr(llm.tracing, "subsegment", fake_sub)
        monkeypatch.setattr(llm.tracing, "annotate", lambda k, v: notes.append((k, v)))
        return spans, notes

    def test_each_llm_call_is_a_named_subsegment(self, monkeypatch):
        spans, _ = self._record(monkeypatch)
        monkeypatch.setattr(llm, "_HAS_LITELLM", True)
        monkeypatch.setattr(
            llm, "litellm",
            SimpleNamespace(completion=lambda **kw: {"choices": [{"message": {"content": "oi"}}]}),
        )
        out = llm._completion([{"role": "user", "content": "x"}], "k", "deepseek/deepseek-chat", 40, 0.0)
        assert out == "oi"
        assert spans == [("llm:deepseek/deepseek-chat", {"model": "deepseek/deepseek-chat", "max_tokens": 40})]

    def test_fallback_is_marked_and_both_attempts_are_traced(self, monkeypatch):
        spans, notes = self._record(monkeypatch)
        monkeypatch.setattr(llm, "_HAS_LITELLM", True)

        def completion(**kw):
            if kw["model"].endswith("deepseek-chat"):
                raise RuntimeError("429")
            return {"choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(llm, "litellm", SimpleNamespace(completion=completion))
        out = llm._completion_with_fallback(
            [{"role": "user", "content": "x"}], "k", "deepseek/deepseek-chat", "anthropic/claude-haiku-4.5", 40, 0.0
        )
        assert out == "ok"
        assert [s[0] for s in spans] == ["llm:deepseek/deepseek-chat", "llm:anthropic/claude-haiku-4.5"]
        assert ("llm_fallback", True) in notes
