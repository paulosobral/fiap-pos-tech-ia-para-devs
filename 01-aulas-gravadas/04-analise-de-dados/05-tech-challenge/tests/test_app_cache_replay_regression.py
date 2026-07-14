"""Regression test for the CacheReplayClosureError fixed in app.py.

``app.py`` cannot be imported directly in a test process: it runs
Streamlit UI code unconditionally at module level (``st.set_page_config``,
tab construction, etc.), which the rest of this test suite already avoids
for the same reason (see the README's note on the AppTest/pandas/pyarrow
segfault). Instead, this test exercises the exact caching *pattern* now
used by ``_extract_pose_frame_series``: a ``st.progress`` bar created and
mutated entirely inside a function decorated with ``@st.cache_data``.

Before the fix, ``app.py``'s ``_extract_pose_frame_series`` created its
progress bar in the *caller* and mutated it via a callback invoked from
inside the cached function. On a cache hit (processing the same video a
second time, or a different video after Streamlit replays recorded calls),
Streamlit raised ``CacheReplayClosureError`` because the replayed UI calls
referenced a layout block from a previous script run that no longer
exists. Calling a ``@st.cache_data`` function twice with the same
arguments — the second call is a cache hit — is exactly the scenario that
previously broke; this test locks in that it no longer does.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

_SCRIPT = """
import streamlit as st


@st.cache_data(show_spinner=False)
def extract_with_self_contained_progress(x):
    progress_bar = st.progress(0.0, text="working...")
    for i in range(3):
        progress_bar.progress((i + 1) / 3, text=f"working... {i + 1}/3")
    progress_bar.empty()
    return x * 2


value = st.number_input("value", value=1)
if st.button("run"):
    result = extract_with_self_contained_progress(value)
    st.write(f"result: {result}")
"""


def test_cached_function_with_self_contained_progress_bar_survives_repeated_calls(tmp_path):
    script_path = tmp_path / "repro.py"
    script_path.write_text(_SCRIPT, encoding="utf-8")

    at = AppTest.from_file(str(script_path))
    at.run()

    at.button[0].click().run()
    assert at.exception == []

    # Second click with the same input is a cache hit — this is exactly
    # the call that raised CacheReplayClosureError before the fix.
    at.button[0].click().run()
    assert at.exception == []
