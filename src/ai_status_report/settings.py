"""Local configuration. Source: https://bbc2.github.io/python-dotenv/ ."""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values, set_key


class ConfigError(ValueError):
    """Safe message only; never include source text or secret values."""


DEFAULTS = {
    "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    "DEEPSEEK_MODEL": "deepseek-v4-flash",
    "GLM_EMBEDDING_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
    "GLM_EMBEDDING_MODEL": "embedding-3",
    "GLM_EMBEDDING_DIMENSIONS": "2048",
    "API_TIMEOUT_SECONDS": "60",
    "TAVILY_MCP_URL": "https://mcp.tavily.com/mcp/",
    # A local server is required when the controller and 8002 are separate
    # processes. PersistentClient remains available only for isolated tests.
    "CHROMA_CLIENT_MODE": "http",
    "CHROMA_HOST": "127.0.0.1",
    "CHROMA_PORT": "8000",
}
KEYS = ("DEEPSEEK_API_KEY", "GLM_EMBEDDING_API_KEY")


@dataclass(frozen=True)
class Profile:
    model: str
    base_url: str
    api_key: str = field(repr=False)


@dataclass(frozen=True)
class Settings:
    generation: Profile
    embedding: Profile
    dimensions: int
    timeout: float
    tavily_api_key: str = field(default="", repr=False)
    tavily_mcp_url: str = DEFAULTS["TAVILY_MCP_URL"]
    chroma_client_mode: str = DEFAULTS["CHROMA_CLIENT_MODE"]
    chroma_host: str = DEFAULTS["CHROMA_HOST"]
    chroma_port: int = 8000


@dataclass(frozen=True)
class SearchSettings:
    deepseek: Profile | None
    timeout: float
    tavily_api_key: str = field(default="", repr=False)
    tavily_mcp_url: str = DEFAULTS["TAVILY_MCP_URL"]


def _profile_from_values(values: dict, prefix: str, *, required: bool) -> Profile | None:
    key = str(values.get(f"{prefix}_API_KEY") or "")
    if not key:
        if required:
            raise ConfigError(f"missing credential: {prefix}_API_KEY")
        return None
    url = str(values[f"{prefix}_BASE_URL"]).rstrip("/")
    allowed = {DEFAULTS[f"{prefix}_BASE_URL"]}
    if prefix == "DEEPSEEK":
        allowed.add("https://api.deepseek.com/v1")
    if url not in allowed:
        raise ConfigError(f"unapproved endpoint: {prefix}")
    model = str(values[f"{prefix}_MODEL"]).lower()
    if not re.fullmatch(r"[a-z0-9._-]{1,80}", model):
        raise ConfigError(f"invalid model identifier: {prefix}")
    if prefix == "DEEPSEEK" and model not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        raise ConfigError("unsupported DeepSeek model")
    if any(c.isspace() for c in key):
        raise ConfigError(f"invalid credential format: {prefix}")
    return Profile(model, url, key)


def _search_values(values: dict) -> tuple[float, str]:
    try:
        timeout = float(values["API_TIMEOUT_SECONDS"])
    except (ValueError, TypeError):
        raise ConfigError("invalid numeric configuration") from None
    if not 1 <= timeout <= 120:
        raise ConfigError("numeric configuration outside approved bounds")
    tavily_url = str(values["TAVILY_MCP_URL"]).rstrip("/") + "/"
    if not tavily_url.startswith(("https://", "http://")):
        raise ConfigError("invalid Tavily MCP URL")
    return timeout, tavily_url


def _chroma_values(values: dict) -> tuple[str, str, int]:
    mode = str(values["CHROMA_CLIENT_MODE"]).lower()
    host = str(values["CHROMA_HOST"]).lower()
    try:
        port = int(values["CHROMA_PORT"])
    except (TypeError, ValueError):
        raise ConfigError("invalid Chroma port") from None
    if mode not in {"http", "persistent"}:
        raise ConfigError("invalid Chroma client mode")
    if not re.fullmatch(r"[a-z0-9.-]{1,253}", host) or not 1 <= port <= 65535:
        raise ConfigError("invalid Chroma endpoint")
    return mode, host, port


def load_search_settings(root: Path, environ=None) -> SearchSettings:
    values = {**DEFAULTS, **dotenv_values(root / ".env", interpolate=False)}
    values.update(os.environ if environ is None else environ)
    timeout, tavily_url = _search_values(values)
    return SearchSettings(
        deepseek=_profile_from_values(values, "DEEPSEEK", required=False),
        timeout=timeout,
        tavily_api_key=str(values.get("TAVILY_API_KEY") or ""),
        tavily_mcp_url=tavily_url,
    )


def load_settings(root: Path, environ=None) -> Settings:
    # No interpolation: source text must never expand other environment secrets.
    values = {**DEFAULTS, **dotenv_values(root / ".env", interpolate=False)}
    values.update(os.environ if environ is None else environ)
    for key in KEYS:
        if not values.get(key) or str(values[key]).strip() in {"", "<your-key>"}:
            raise ConfigError(f"missing credential: {key}")
    profiles = [
        _profile_from_values(values, prefix, required=True)
        for prefix in ("DEEPSEEK", "GLM_EMBEDDING")
    ]
    try:
        dimensions = int(values["GLM_EMBEDDING_DIMENSIONS"])
        timeout, tavily_url = _search_values(values)
        chroma_mode, chroma_host, chroma_port = _chroma_values(values)
    except (ValueError, TypeError):
        raise ConfigError("invalid numeric configuration") from None
    if (
        dimensions not in {256, 512, 1024, 2048}
        or not 1 <= timeout <= 120
    ):
        raise ConfigError("numeric configuration outside approved bounds")
    return Settings(
        generation=profiles[0],
        embedding=profiles[1],
        dimensions=dimensions,
        timeout=timeout,
        tavily_api_key=str(values.get("TAVILY_API_KEY") or ""),
        tavily_mcp_url=tavily_url,
        chroma_client_mode=chroma_mode,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
    )


def parse_legacy(text: str) -> dict[str, str]:
    """Parse blank-line separated API/model/base_url sections, never execute text."""
    result = {}
    for section in re.split(r"\n\s*\n", text):
        fields: dict[str, list[str]] = {}
        for line in section.splitlines():
            match = re.match(r"^\s*([A-Za-z_]+)\s*[:=：]\s*(.*?)\s*$", line)
            if match:
                name, value = match.groups()
                fields.setdefault(name.lower(), []).append(value.strip("\"'"))
        models = [m.lower() for m in fields.get("model", [])]
        prefixes = []
        if any(m.startswith("deepseek-") for m in models):
            prefixes.append(("DEEPSEEK", next(m for m in models if m.startswith("deepseek-"))))
        if "embedding-3" in models:
            prefixes.append(("GLM_EMBEDDING", "embedding-3"))
        if not prefixes:
            continue
        keys = fields.get("api", []) + fields.get("api_key", [])
        urls = fields.get("base_url", [])
        if len(keys) != 1 or not keys[0] or len(urls) != 1:
            raise ConfigError("ambiguous legacy service section")
        for prefix, model in prefixes:
            for suffix, value in (("API_KEY", keys[0]), ("BASE_URL", urls[0]), ("MODEL", model)):
                name = f"{prefix}_{suffix}"
                if name in result:
                    raise ConfigError("duplicate legacy service section")
                result[name] = value
    if not all(result.get(key) for key in KEYS):
        raise ConfigError("legacy file missing required service sections")
    return result


def bootstrap(root: Path, source: Path) -> list[str]:
    imported = parse_legacy(source.read_text(encoding="utf-8-sig"))
    destination = root / ".env"
    if destination.is_symlink():
        raise ConfigError("refusing symlink configuration")
    existing = dotenv_values(destination, interpolate=False)
    # Validate before writing. Keep every nonempty value; conflicts are never overwritten.
    effective = {**DEFAULTS, **imported}
    effective.update({k: v for k, v in existing.items() if v})
    effective.update(os.environ)
    load_settings(root, effective)
    destination.touch(mode=0o600, exist_ok=True)
    added = []
    for name, value in {**DEFAULTS, **imported}.items():
        if not existing.get(name) and not os.environ.get(name):
            set_key(destination, name, value, quote_mode="always")
            added.append(name)
    if os.name == "nt":
        # dotenv atomically replaces files; restore owner-only ACL after replacement.
        identity = subprocess.run(
            ["whoami"], capture_output=True, text=True, check=True
        ).stdout.strip()
        acl = subprocess.run(
            ["icacls", str(destination), "/inheritance:r", "/grant:r", f"{identity}:(F)"],
            capture_output=True,
            check=False,
        )
        if acl.returncode:
            raise ConfigError("env imported but ACL restriction failed; restrict local access")
    else:
        destination.chmod(0o600)
    return added
