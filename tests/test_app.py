import json
from pathlib import Path

import pytest

import settings
from conftest import PAGES, ROOT

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture
def tfidf_project(tmp_path, monkeypatch):
    from chunking import build_chunks
    import streamlit as st

    data = tmp_path / "data"
    data.mkdir()
    (data / "chunks.json").write_text(json.dumps(build_chunks(PAGES, 60, 10, 5)), encoding="utf-8")
    (data / "config.json").write_text(json.dumps({"embedder": "none", "method": "tfidf", "tau": 0.05}), encoding="utf-8")
    monkeypatch.setattr(settings, "DATA_DIR", data)
    monkeypatch.setattr(settings, "CHUNKS_PATH", data / "chunks.json")
    monkeypatch.setattr(settings, "CONFIG_PATH", data / "config.json")
    monkeypatch.setattr(settings, "DOCS_DIR", tmp_path / "docs")
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    st.cache_resource.clear()
    return tmp_path


def ask(at, question):
    at.text_input[0].set_value(question)
    at.button[0].click()
    return at.run()


def test_app_answers_with_sources(tfidf_project):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception
    at = ask(at, "Is an Aadhaar linked bank account mandatory for scholarship payment?")
    assert not at.exception
    assert any(s.value == "Answer" for s in at.subheader)
    assert any("scholarship.pdf" in e.label for e in at.expander)


def test_app_refuses_off_topic_question(tfidf_project):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    at = ask(at, "zzzz qqqq")
    assert not at.exception
    assert any("Not found" in w.value for w in at.warning)


def test_app_shows_setup_help_when_index_is_missing(tfidf_project, monkeypatch):
    import streamlit as st
    monkeypatch.setattr(settings, "CHUNKS_PATH", Path(tfidf_project) / "missing.json")
    st.cache_resource.clear()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert any("No index found" in e.value for e in at.error)
