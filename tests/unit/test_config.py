"""Unit tests for config loading (no game needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rainworld_rl import config as cfg


def test_defaults_and_explicit_file(tmp_path):
    c = cfg.Config()
    assert c.game_dir == cfg.DEFAULT_GAME_DIR
    assert c.source is None
    p = tmp_path / "x.toml"
    p.write_text('game_dir = "D:/Games/Rain World"\nlaunch_timeout = 30\n', encoding = "utf-8")
    c = cfg.load_config(p)
    assert c.game_dir == Path("D:/Games/Rain World")
    assert c.plugin_dll_path == c.game_dir / "BepInEx" / "plugins" / "RainWorldRL.dll"
    assert c.launch_timeout == 30.0
    assert c.source == p


def test_lookup_order(tmp_path, monkeypatch):
    """RAINWORLD_RL_CONFIG overrides the repo/user files; an explicit path beats it."""
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text('game_dir = "E:/A"\n', encoding = "utf-8")
    b.write_text('game_dir = "E:/B"\n', encoding = "utf-8")
    monkeypatch.setenv(cfg.CONFIG_ENV_VAR, str(b))
    assert cfg.load_config().game_dir == Path("E:/B")
    assert cfg.load_config(a).game_dir == Path("E:/A")


def test_defaults_when_nothing_present(monkeypatch):
    monkeypatch.delenv(cfg.CONFIG_ENV_VAR, raising = False)
    monkeypatch.setattr(cfg, "REPO_ROOT", Path("Q:/definitely/not/here"))
    monkeypatch.setattr(cfg, "USER_CONFIG_PATH", Path("Q:/definitely/not/here.toml"))
    assert cfg.load_config() == cfg.Config()


def test_errors(tmp_path, monkeypatch):
    with pytest.raises(cfg.ConfigError):
        cfg.load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("game_dir = 5\n", encoding = "utf-8")
    with pytest.raises(cfg.ConfigError):
        cfg.load_config(bad)
    unknown = tmp_path / "unknown.toml"
    unknown.write_text('slugcat = "white"\n', encoding = "utf-8")
    with pytest.raises(cfg.ConfigError, match = "Unknown config keys"):
        cfg.load_config(unknown)
    malformed = tmp_path / "malformed.toml"
    malformed.write_text("game_dir = \n", encoding = "utf-8")
    with pytest.raises(cfg.ConfigError, match = "Invalid TOML"):
        cfg.load_config(malformed)
    monkeypatch.setenv(cfg.CONFIG_ENV_VAR, str(tmp_path / "nope.toml"))
    with pytest.raises(cfg.ConfigError):
        cfg.load_config()


def test_example_toml_parses_and_matches_defaults():
    c = cfg.load_config(cfg.REPO_ROOT / "rainworld_rl.example.toml")
    assert c.launch_timeout == cfg.Config().launch_timeout


def test_renderer(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('renderer = "virtualgl"\n', encoding = "utf-8")
    assert cfg.load_config(p).renderer == "virtualgl"
    assert cfg.Config().renderer == "auto"
    p.write_text('renderer = "vulkan"\n', encoding = "utf-8")
    with pytest.raises(cfg.ConfigError):
        cfg.load_config(p)
