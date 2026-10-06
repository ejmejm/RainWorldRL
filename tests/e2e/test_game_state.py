"""Game-state info fields: food meter, cycle_progress, rain, in_shelter, malnourished, cycle_survived, region."""

from __future__ import annotations

from tests.e2e.helpers import assert_info_contract, infos, step_n


def test_fresh_save_game_state(fresh_env):
    """
    Right after a fresh reset: a consistent food meter, early in the cycle, no rain, not malnourished, and
    region "SU" (a fresh Survivor save starts in Outskirts). The next 50 steps stay early and dry, and
    (region, room_index) is the same on every READY step while the slugcat stays in the start shelter.
    """
    results = step_n(fresh_env, 51)
    first = results[0][1]
    assert_info_contract(first)
    assert first["ready"], "fresh_env should be READY after reset"

    assert 0 < first["food_to_hibernate"] <= first["food_max"] <= 255, first
    assert 0 <= first["food"] <= first["food_max"], first
    assert first["malnourished"] is False
    assert first["cycle_survived"] is False
    assert first["rain"] is False
    assert 0.0 <= first["cycle_progress"] < 0.5, f"fresh cycle should be early, got {first['cycle_progress']}"
    assert first["region"] == "SU" and first["room_index"] >= 0, first

    assert not any(infos(results, "rain"))
    assert all(p < 0.5 for p in infos(results, "cycle_progress"))
    assert not any(infos(results, "cycle_survived"))
    assert not any(infos(results, "malnourished"))
    ready = [info for _obs, info in results if info["ready"]]
    assert len(ready) >= len(results) - 5, f"too few READY steps: {len(ready)}"
    keys = {(info["region"], info["room_index"]) for info in ready}
    assert keys == {("SU", first["room_index"])}, f"(region, room_index) changed while standing still: {keys}"
    print(f"[game_state] in_shelter at reset = {first['in_shelter']}, room {first['room_index']}, "
          f"food meter {first['food']}/{first['food_to_hibernate']}/{first['food_max']}")


# On a fresh save the yellow overseer's tutorial pins RainCycle.timer to 2000 (OverseerTutorialBehavior.cs:1769
# pauseRain = true, :2015 timer = 2000) until the player passes x > 600 in SU_A43 or enters SU_A22, so in the
# start rooms cycle_progress sits at exactly 2000 / cycleLength (about 0.06..0.12) and only then starts moving.
TUTORIAL_PINNED_TIMER = 2000


def test_cycle_progress_is_monotone_and_either_advances_or_is_tutorial_pinned(env):
    """
    Over 300 no-op steps cycle_progress never decreases (while READY and in the same cycle). It either
    advances, or - in the first cycle of a fresh save, before the tutorial releases the rain - stays at
    exactly 2000 / cycleLength (cycleLength is 400..800 s * 40 ticks = 16000..32000).
    """
    results = step_n(env, 300)
    progress = [
        (i, float(p)) for i, (p, ready, cyc) in enumerate(zip(
            infos(results, "cycle_progress"), infos(results, "ready"), infos(results, "cycle_number")))
        if ready and cyc == results[0][1]["cycle_number"]
    ]
    assert len(progress) >= 250, f"too few READY steps in the same cycle: {len(progress)}"
    decreases = [(i, a, b) for (i, a), (_j, b) in zip(progress, progress[1:]) if b < a]
    assert not decreases, f"cycle_progress decreased at {decreases[:5]}"
    assert all(0.0 < p <= 1.5 for _i, p in progress)
    first, last = progress[0][1], progress[-1][1]
    if last == first:
        implied_cycle_length = TUTORIAL_PINNED_TIMER / first
        assert 16000 * 0.95 <= implied_cycle_length <= 32000 * 1.05, (
            f"cycle_progress constant at {first} but not the tutorial-pinned 2000/cycleLength "
            f"(implied cycleLength {implied_cycle_length:.0f})")
        print(f"[game_state] cycle_progress pinned by the first-cycle tutorial at {first:.4f} "
              f"(cycleLength ~{implied_cycle_length:.0f} ticks)")
    else:
        assert last > first

