from __future__ import annotations

from rainworld_rl import launcher


def test_game_cpus_blocks_per_instance(monkeypatch):
    monkeypatch.setattr(launcher.os, "sched_getaffinity", lambda pid: {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26})
    assert launcher._game_cpus(0) == [0, 2, 4, 6, 8, 10]
    assert launcher._game_cpus(1) == [12, 14, 16, 18, 20, 22]
    assert launcher._game_cpus(2) == [24, 26, 0, 2, 4, 6]  # wraps around when there are too few


def test_game_cpus_small_allocation_not_pinned(monkeypatch):
    monkeypatch.setattr(launcher.os, "sched_getaffinity", lambda pid: set(range(launcher.GAME_CPUS)))
    assert launcher._game_cpus(0) is None
