"""Explicit, local-only maintenance for the disposable Chroma evidence store."""

from __future__ import annotations

import shutil
import socket
from pathlib import Path


class RagStoreMaintenanceError(RuntimeError):
    """Raised when a destructive local store operation is unsafe."""


def reset_rag_store(root: Path, *, host: str = "127.0.0.1", port: int = 8000) -> dict[str, object]:
    """Delete only this project's stopped Chroma persistence directory.

    Chroma client-server mode persists data under the directory passed to
    ``chroma run --path``. The evidence index is derived from durable
    EvidenceChunks, so it may be rebuilt after a damaged HNSW segment.
    Source: https://docs.trychroma.com/docs/run-chroma/client-server
    """

    project_root = root.resolve()
    store_parent = (project_root / "data" / "vector_store").resolve()
    store = (store_parent / "chroma").resolve()
    if store.parent != store_parent or store == project_root:
        raise RagStoreMaintenanceError("invalid_rag_store_path")
    try:
        with socket.create_connection((host, port), timeout=0.2):
            raise RagStoreMaintenanceError("chroma_service_still_running")
    except OSError:
        pass
    existed = store.exists()
    if existed:
        shutil.rmtree(store)
    return {"status": "cleared", "path": str(store), "existed": existed}
