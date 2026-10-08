from pathlib import Path

import pytest
from dotenv import dotenv_values

from ai_status_report.settings import (
    KEYS,
    ConfigError,
    bootstrap,
    load_search_settings,
    load_settings,
    parse_legacy,
)

LEGACY = """生成服务
API: synthetic-deepseek
model: deepseek-v4-flash
base_url: https://api.deepseek.com

文档服务
API: synthetic-glm
model: GLM-OCR
model: Embedding-3
base_url: https://open.bigmodel.cn/api/paas/v4
"""


def test_sections_and_case_normalization():
    parsed = parse_legacy(LEGACY)
    assert parsed["GLM_OCR_MODEL"] == "glm-ocr"
    assert parsed["DEEPSEEK_API_KEY"] != parsed["GLM_OCR_API_KEY"]
    assert parsed["GLM_OCR_API_KEY"] == parsed["GLM_EMBEDDING_API_KEY"]


@pytest.mark.parametrize(
    "text",
    ["", LEGACY + "\n" + LEGACY, LEGACY.replace("API: synthetic-glm", "API: first\nAPI: second")],
)
def test_missing_and_ambiguous_sections_rejected(text):
    with pytest.raises(ConfigError):
        parse_legacy(text)


def test_import_idempotence_and_existing_value_preservation(tmp_path, monkeypatch):
    for key in (*parse_legacy(LEGACY), "API_TIMEOUT_SECONDS", "GLM_EMBEDDING_DIMENSIONS"):
        monkeypatch.delenv(key, raising=False)
    source = tmp_path / "input.txt"
    source.write_text(LEGACY, encoding="utf-8")
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY='keep-existing'\n", encoding="utf-8")
    bootstrap(tmp_path, source)
    assert dotenv_values(tmp_path / ".env")["DEEPSEEK_API_KEY"] == "keep-existing"
    before = (tmp_path / ".env").read_bytes()
    assert bootstrap(tmp_path, source) == []
    assert (tmp_path / ".env").read_bytes() == before


def test_environment_priority_and_secret_repr(tmp_path):
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=local\n", encoding="utf-8")
    env = {key: "unique-test-secret" for key in KEYS}
    settings = load_settings(tmp_path, env)
    assert settings.generation.api_key == "unique-test-secret"
    assert "unique-test-secret" not in repr(settings)
    assert settings.chroma_client_mode == "http"
    assert (settings.chroma_host, settings.chroma_port) == ("127.0.0.1", 8000)


@pytest.mark.parametrize(
    "override",
    [
        {"DEEPSEEK_BASE_URL": "https://untrusted.example"},
        {"API_TIMEOUT_SECONDS": "nan"},
        {"API_TIMEOUT_SECONDS": "0"},
        {"GLM_EMBEDDING_DIMENSIONS": "3"},
        {"CHROMA_CLIENT_MODE": "unknown"},
        {"CHROMA_PORT": "0"},
        {"DEEPSEEK_MODEL": "deepseek-flash"},
        {"DEEPSEEK_API_KEY": ""},
        {"DEEPSEEK_API_KEY": "bad\nkey"},
    ],
)
def test_invalid_settings_fail_safely(tmp_path, override):
    with pytest.raises(ConfigError) as error:
        load_settings(tmp_path, {**dict.fromkeys(KEYS, "synthetic"), **override})
    assert "untrusted.example" not in str(error.value)
    assert "bad\nkey" not in str(error.value)


def test_interpolation_is_disabled(tmp_path):
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY='${UNRELATED_SECRET}'\n", encoding="utf-8")
    settings = load_settings(tmp_path, {"GLM_OCR_API_KEY": "ocr", "GLM_EMBEDDING_API_KEY": "embed"})
    assert settings.generation.api_key == "${UNRELATED_SECRET}"


def test_invalid_source_does_not_create_env(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("invalid", encoding="utf-8")
    with pytest.raises(ConfigError):
        bootstrap(tmp_path, source)
    assert not Path(tmp_path / ".env").exists()


def test_search_settings_loads_an_approved_optional_model(tmp_path):
    settings = load_search_settings(
        tmp_path,
        {
            "DEEPSEEK_API_KEY": "search-secret",
            "DEEPSEEK_MODEL": "deepseek-v4-flash",
        },
    )
    assert settings.deepseek.model == "deepseek-v4-flash"
