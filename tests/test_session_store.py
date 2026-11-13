from types import SimpleNamespace

import pytest

from squid.generate.session_store import SessionStore


def test_session_store_reuses_only_matching_session():
    store = SessionStore(
        factory=lambda: SimpleNamespace(),
        max_sessions=2,
    )

    first = store.get("a")
    second = store.get("b")

    assert store.get("a") is first
    assert store.get("b") is second
    assert first is not second
    assert len(store) == 2


def test_session_store_uses_lru_eviction_and_clear():
    store = SessionStore(
        factory=lambda: SimpleNamespace(),
        max_sessions=2,
    )

    first = store.get("a")
    second = store.get("b")
    store.get("a")
    third = store.get("c")

    assert store.get("a") is first
    assert store.get("c") is third
    assert len(store) == 2

    store.clear("a")
    assert len(store) == 1
    assert store.get("a") is not first
    assert len(store) == 2

    with pytest.raises(ValueError, match="非空"):
        store.get("")

    with pytest.raises(ValueError, match="128"):
        store.get("x" * 129)


def test_session_store_lists_metadata_and_finds_existing_session():
    store = SessionStore(factory=lambda: SimpleNamespace(history=[]))
    empty = store.get("empty")
    store.get("used").history.append(("第一句话", "回答"))

    listed = store.list_sessions()
    assert listed == [
        {"session_id": "used", "title": "第一句话", "turns": 1},
        {"session_id": "empty", "title": "新会话", "turns": 0},
    ]
    assert store.find("used").history == [("第一句话", "回答")]
    assert store.find("missing") is None
    assert empty.history == []
