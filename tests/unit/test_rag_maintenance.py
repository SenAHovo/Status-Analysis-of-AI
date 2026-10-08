import socket

import pytest

from ai_status_report.rag.maintenance import RagStoreMaintenanceError, reset_rag_store


def test_reset_rag_store_deletes_only_project_chroma_directory(tmp_path):
    store = tmp_path / "data" / "vector_store" / "chroma"
    store.mkdir(parents=True)
    (store / "chroma.sqlite3").write_text("derived", encoding="utf-8")
    retained = tmp_path / "data" / "vector_store" / "retained.txt"
    retained.write_text("keep", encoding="utf-8")

    result = reset_rag_store(tmp_path, port=65530)

    assert result["status"] == "cleared"
    assert store.exists() is False
    assert retained.read_text(encoding="utf-8") == "keep"


def test_reset_rag_store_refuses_when_chroma_port_is_open(tmp_path, monkeypatch):
    class OpenConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: OpenConnection())

    with pytest.raises(RagStoreMaintenanceError, match="chroma_service_still_running"):
        reset_rag_store(tmp_path)
