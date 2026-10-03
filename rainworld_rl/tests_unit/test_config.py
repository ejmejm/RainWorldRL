"""Unit tests for config loading (no game needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rainworld_rl import config as cfg


def test_defaults():
    c = cfg.Config()
    assert c.game_dir == Path("Z:/SteamLibrary/steamapps/common/Rain World")
    assert c.exe_path == c.game_dir / "RainWorld.exe"
    assert c.plugins_dir == c.game_dir / "BepInEx" / "plugins"
    assert c.plugin_dll_path == c.plugins_dir / "RainWorldRL.dll"
    assert c.bepinex_log == c.game_dir / "BepInEx" / "LogOutput.log"
    assert c.rl_save_dir == c.plugins_dir / "RainWorldRL" / "saves"
    assert c.launch_timeout == 120.0
    assert c.source is None


def test_load_explicit_file(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text('game_dir = "D:/Games/Rain World"\nlaunch_timeout = 30\n', encoding = "utf-8")
    c = cfg.load_config(p)
    assert c.game_dir == Path("D:/Games/Rain World")
    assert c.launch_timeout == 30.0
    assert c.rl_save_dir == Path("D:/Games/Rain World/BepInEx/plugins/RainWorldRL/saves")
    assert c.source == p


def test_env_var_overrides_repo_file(tmp_path, monkeypatch):
    p = tmp_path / "env.toml"
    p.write_text('game_dir = "E:/Env"\nrl_save_dir = "E:/saves"\n', encoding = "utf-8")
    monkeypatch.setenv(cfg.CONFIG_ENV_VAR, str(p))
    c = cfg.load_config()
    assert c.game_dir == Path("E:/Env")
    assert c.rl_save_dir == Path("E:/saves")


def test_explicit_arg_beats_env_var(tmp_path, monkeypatch):
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text('game_dir = "E:/A"\n', encoding = "utf-8")
    b.write_text('game_dir = "E:/B"\n', encoding = "utf-8")
    monkeypatch.setenv(cfg.CONFIG_ENV_VAR, str(b))
    assert cfg.load_config(a).game_dir == Path("E:/A")


def test_defaults_when_nothing_present(monkeypatch):
    monkeypatch.delenv(cfg.CONFIG_ENV_VAR, raising = False)
    monkeypatch.setattr(cfg, "REPO_ROOT", Path("Q:/definitely/not/here"))
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
    example = cfg.REPO_ROOT / "rainworld_rl.example.toml"
    assert example.is_file()
    c = cfg.load_config(example)
    assert c.game_dir == cfg.Config().game_dir
    assert c.launch_timeout == cfg.Config().launch_timeout
