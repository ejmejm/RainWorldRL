"""
Cross-process lock around the single Rain World instance.

Several worktrees / agents may want the game; only one can drive it at a time.
The lock is a directory created with ``mkdir`` (atomic: fails if it exists).
Hold it for the whole launch + run + kill, release it in ``finally``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Iterator, Optional, Union

logger = logging.getLogger(__name__)


class GameLockTimeout(TimeoutError):
    """The lock directory stayed occupied past the timeout."""


@contextlib.contextmanager
def game_lock(
    path: Optional[Union[str, os.PathLike]],
    *,
    poll_seconds: float = 30.0,
    timeout_seconds: float = 20 * 60,
) -> Iterator[None]:
    """
    Acquire the mkdir lock at ``path`` (no-op when ``path`` is None), yield,
    release. Waits ``poll_seconds`` between attempts up to ``timeout_seconds``.
    """
    if path is None:
        yield
        return

    lock_dir = Path(path)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            lock_dir.mkdir(parents = False, exist_ok = False)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise GameLockTimeout(f"game lock {lock_dir} still held after {timeout_seconds:.0f}s")
            logger.info("game lock %s is held by someone else; retrying in %.0fs", lock_dir, poll_seconds)
            time.sleep(poll_seconds)
    logger.info("acquired game lock %s", lock_dir)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
            logger.info("released game lock %s", lock_dir)
        except OSError as e:  # pragma: no cover - best effort
            logger.warning("could not release game lock %s: %s", lock_dir, e)
