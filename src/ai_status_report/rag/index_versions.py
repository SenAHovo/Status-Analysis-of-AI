"""Version and distance-space policy for persistent evidence indexes."""

from __future__ import annotations

DEFAULT_INDEX_VERSION = "v2"
MAX_INDEX_VERSION_LENGTH = 64
LEGACY_READ_ONLY_INDEX_VERSIONS = frozenset({"v1"})
_INDEX_SPACES = {"v1": "l2"}


def is_valid_index_version(value: str) -> bool:
    """Return whether a version is safe for contracts and collection names."""

    return bool(
        value
        and len(value) <= MAX_INDEX_VERSION_LENGTH
        and value.replace("_", "").replace("-", "").isalnum()
    )


def distance_space_for(index_version: str) -> str:
    """Return the persisted Chroma HNSW space for an index version."""

    return _INDEX_SPACES.get(index_version, "cosine")


def is_legacy_read_only(index_version: str) -> bool:
    """Keep historical indexes available for replay without allowing mutation."""

    return index_version in LEGACY_READ_ONLY_INDEX_VERSIONS
