import sys
import types

import pytest

import tracing


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(tracing, "_enabled", False)


def _install_fake_sdk(monkeypatch, patch):
    core = types.ModuleType("aws_xray_sdk.core")
    core.patch = patch

    def forbidden(*args, **kwargs):
        raise AssertionError("patch_all() instrumenta requests/urllib e grava a URL (token do bot) no trace")

    core.patch_all = forbidden
    pkg = types.ModuleType("aws_xray_sdk")
    pkg.core = core
    monkeypatch.setitem(sys.modules, "aws_xray_sdk", pkg)
    monkeypatch.setitem(sys.modules, "aws_xray_sdk.core", core)


def test_enable_patches_libraries_once(monkeypatch):
    calls = []
    _install_fake_sdk(monkeypatch, lambda modules: calls.append(modules))
    assert tracing.enable() is True
    assert tracing.enable() is True  # idempotente: não instrumenta de novo
    assert calls == [("botocore",)]  # só botocore: nunca requests/urllib (vazaria o token do bot)


def test_missing_segment_only_logs_instead_of_raising(monkeypatch):
    monkeypatch.delenv("AWS_XRAY_CONTEXT_MISSING", raising=False)
    _install_fake_sdk(monkeypatch, lambda modules: None)
    tracing.enable()
    import os

    assert os.environ["AWS_XRAY_CONTEXT_MISSING"] == "LOG_ERROR"


def test_explicit_setting_is_respected(monkeypatch):
    monkeypatch.setenv("AWS_XRAY_CONTEXT_MISSING", "RUNTIME_ERROR")
    _install_fake_sdk(monkeypatch, lambda modules: None)
    tracing.enable()
    import os

    assert os.environ["AWS_XRAY_CONTEXT_MISSING"] == "RUNTIME_ERROR"


def test_enable_without_sdk_is_a_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "aws_xray_sdk", None)  # import falha com ImportError
    monkeypatch.setitem(sys.modules, "aws_xray_sdk.core", None)
    assert tracing.enable() is False


def test_enable_never_breaks_the_handler_when_patching_fails(monkeypatch):
    def boom(modules):
        raise RuntimeError("daemon indisponível")

    _install_fake_sdk(monkeypatch, boom)
    assert tracing.enable() is False


def test_handler_module_turns_tracing_on_at_import(monkeypatch):
    calls = []
    _install_fake_sdk(monkeypatch, lambda modules: calls.append(modules))
    sys.modules.pop("handler", None)
    import handler  # noqa: F401

    assert calls == [("botocore",)]
