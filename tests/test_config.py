"""The .env loader and the path layout (no real .env is read)."""

import os

import pytest

from significance import config


def test_read_env_file_parses_comments_export_and_quotes(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "PLAIN=abc\n"
        "export EXPORTED=def\n"
        "QUOTED='g h'\n"
        'DOUBLE="i=j"\n'
        "SPACED = k \n"
        "no_equals_line\n"
    )
    assert config.read_env_file(env) == {
        "PLAIN": "abc",
        "EXPORTED": "def",
        "QUOTED": "g h",
        "DOUBLE": "i=j",
        "SPACED": "k",
    }


def test_load_env_returns_names_only_and_existing_values_win(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DS_TEST_A=from_file\nDS_TEST_B=from_file\n")
    monkeypatch.setenv("DS_TEST_A", "from_env")
    monkeypatch.delenv("DS_TEST_B", raising=False)
    names = config.load_env(env)
    assert names == ["DS_TEST_A", "DS_TEST_B"]
    assert os.environ["DS_TEST_A"] == "from_env"
    assert os.environ["DS_TEST_B"] == "from_file"
    config.load_env(env, override=True)
    assert os.environ["DS_TEST_A"] == "from_file"


def test_load_env_missing_file_is_empty(tmp_path):
    assert config.load_env(tmp_path / "absent.env") == []


def test_require_secret_names_the_variable_but_never_a_value(monkeypatch):
    monkeypatch.delenv("DS_TEST_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="DS_TEST_SECRET"):
        config.require_secret("DS_TEST_SECRET")
    monkeypatch.setenv("DS_TEST_SECRET", "s3cr3t-value")
    assert config.require_secret("DS_TEST_SECRET") == "s3cr3t-value"


def test_paths_layout_keeps_data_in_temp(tmp_path):
    p = config.Paths(tmp_path)
    for folder in (p.data, p.raw, p.catalog, p.work, p.cache):
        assert folder.is_relative_to(tmp_path / ".temp")
    assert p.outputs == tmp_path / "outputs"
    assert p.env_file == tmp_path / ".env"


def test_project_root_prefers_ds_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DS_ROOT", str(tmp_path))
    assert config.project_root() == tmp_path.resolve()


def test_fixed_settings_are_consistent():
    assert config.MAX_IDS_PER_CALL == 100
    assert set(config.DEFAULT_VARIANTS) <= set(config.VARIANTS)
    assert "V3" not in config.DEFAULT_VARIANTS
    assert config.OUTPUT_COLUMNS[:2] == ("record_id", "news_id")
    assert set(config.MIND_FILES) == {"train", "dev"}
