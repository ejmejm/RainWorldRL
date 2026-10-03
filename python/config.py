"""
Configuration for the Rain World RL Python side.

Resolution order for ``load_config()``:

1. An explicit ``path`` argument.
2. The file named by the ``RAINWORLD_RL_CONFIG`` environment variable.
3. ``rainworld_rl.toml`` at the repository root (git-ignored; copy
   ``rainworld_rl.example.toml`` to create it).
4. Built-in defaults.

Only ``game_dir``, ``rl_save_dir`` and ``launch_timeout`` are read from the
file; everything else is derived from ``game_dir``. ``rl_save_dir`` is
informational: the mod decides where the RL save lives, this value only tells
Python where to look (e.g. for debugging).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


PathLike = Union[str, os.PathLike]

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILENAME = "rainworld_rl.toml"
CONFIG_ENV_VAR = "RAINWORLD_RL_CONFIG"

DEFAULT_GAME_DIR = Path("Z:/SteamLibrary/steamapps/common/Rain World")
DEFAULT_LAUNCH_TIMEOUT = 120.0

GAME_EXE_NAME = "RainWorld.exe"
PLUGIN_DLL_NAME = "RainWorldRL.dll"


class ConfigError(ValueError):
    """The config file could not be read or contains invalid values."""


@dataclass(frozen = True)
class Config:
    """
    Resolved configuration.

    Attributes:
        game_dir: Rain World installation directory (contains ``RainWorld.exe``).
        launch_timeout: Seconds ``launch()`` waits for the mod's heartbeat after
            starting the game.
        rl_save_dir: Where the mod keeps the RL save (informational).
        source: The config file the values came from, or None for defaults.
    """

    game_dir: Path = DEFAULT_GAME_DIR
    launch_timeout: float = DEFAULT_LAUNCH_TIMEOUT
    rl_save_dir: Optional[Path] = None
    source: Optional[Path] = field(default = None, compare = False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "game_dir", Path(self.game_dir))
        object.__setattr__(self, "launch_timeout", float(self.launch_timeout))
        if self.launch_timeout <= 0:
            raise ConfigError(f"launch_timeout must be positive, got {self.launch_timeout}")
        if self.rl_save_dir is None:
            object.__setattr__(self, "rl_save_dir", self.plugins_dir / "RainWorldRL" / "saves")
        else:
            object.__setattr__(self, "rl_save_dir", Path(self.rl_save_dir))
        if self.source is not None:
            object.__setattr__(self, "source", Path(self.source))

    # -- derived paths -----------------------------------------------------

    @property
    def exe_path(self) -> Path:
        return self.game_dir / GAME_EXE_NAME

    @property
    def bepinex_dir(self) -> Path:
        return self.game_dir / "BepInEx"

    @property
    def plugins_dir(self) -> Path:
        return self.bepinex_dir / "plugins"

    @property
    def plugin_dll_path(self) -> Path:
        """Where the built mod DLL is deployed to."""
        return self.plugins_dir / PLUGIN_DLL_NAME

    @property
    def bepinex_log(self) -> Path:
        return self.bepinex_dir / "LogOutput.log"

    def validate_game_dir(self) -> None:
        """Raise ``ConfigError`` if ``game_dir`` does not look like a Rain World install."""
        if not self.exe_path.is_file():
            raise ConfigError(
                f"{self.exe_path} not found. Set game_dir in {CONFIG_FILENAME} "
                f"(see rainworld_rl.example.toml) or via {CONFIG_ENV_VAR}."
            )


def find_config_path(path: Optional[PathLike] = None) -> Optional[Path]:
    """Return the config file that ``load_config`` would use, or None for defaults."""
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
        return p

    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        p = Path(env_path)
        if not p.is_file():
            raise ConfigError(f"{CONFIG_ENV_VAR} points to a missing file: {p}")
        return p

    repo_cfg = REPO_ROOT / CONFIG_FILENAME
    if repo_cfg.is_file():
        return repo_cfg
    return None


def _config_from_dict(data: Dict[str, Any], source: Optional[Path]) -> Config:
    known = {"game_dir", "launch_timeout", "rl_save_dir"}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown config keys in {source}: {sorted(unknown)}")

    kwargs: Dict[str, Any] = {}
    if "game_dir" in data:
        if not isinstance(data["game_dir"], str):
            raise ConfigError("game_dir must be a string")
        kwargs["game_dir"] = Path(data["game_dir"])
    if "launch_timeout" in data:
        if isinstance(data["launch_timeout"], bool) or not isinstance(data["launch_timeout"], (int, float)):
            raise ConfigError("launch_timeout must be a number")
        kwargs["launch_timeout"] = float(data["launch_timeout"])
    if "rl_save_dir" in data:
        if not isinstance(data["rl_save_dir"], str):
            raise ConfigError("rl_save_dir must be a string")
        kwargs["rl_save_dir"] = Path(data["rl_save_dir"])
    return Config(source = source, **kwargs)


def load_config(path: Optional[PathLike] = None) -> Config:
    """
    Load the configuration (see module docstring for the resolution order).

    Args:
        path: Explicit config file. Overrides the environment variable and the
            repo-root file.
    """
    cfg_path = find_config_path(path)
    if cfg_path is None:
        return Config()
    try:
        with open(cfg_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {cfg_path}: {e}") from e
    return _config_from_dict(data, cfg_path)
