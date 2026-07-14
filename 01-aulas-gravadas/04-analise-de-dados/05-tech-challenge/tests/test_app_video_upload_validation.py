"""Tests for the Vídeo tab's upload format validation in app.py.

``app.py`` runs Streamlit UI code unconditionally at module level (no
``if __name__ == "__main__"`` guard), so it cannot be imported directly in
a test process without executing the whole app. These tests instead parse
``VIDEO_ALLOWED_EXTENSIONS`` out of the source via ``ast``, which is enough
to verify the constant that drives both the uploader's accepted-type filter
and the post-upload extension check.
"""
import ast
from pathlib import Path

APP_PY_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _load_video_allowed_extensions():
    tree = ast.parse(APP_PY_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "VIDEO_ALLOWED_EXTENSIONS":
                    return ast.literal_eval(node.value)
    raise AssertionError("VIDEO_ALLOWED_EXTENSIONS not found in app.py")


def test_video_allowed_extensions_includes_mkv():
    extensions = _load_video_allowed_extensions()

    assert "mkv" in extensions


def test_video_allowed_extensions_still_includes_original_formats():
    extensions = _load_video_allowed_extensions()

    assert extensions == ("mp4", "avi", "mov", "mkv")
