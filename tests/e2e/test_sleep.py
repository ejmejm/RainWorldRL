"""cycle_survived edge and the sleep -> next cycle flow, through a real hibernation (debug ENTER_SHELTER command)."""

from __future__ import annotations

from tests.harness import DOWN, LEFT, NOOP, RIGHT

# Steps (1 tick each) bound from ENTER_SHELTER to READY in the next cycle: the shelter loads and the slugcat
# crosses the entrance pipe (~20 ticks), walks in (WALK_STEPS), stands still 20 ticks or holds DOWN 260 ticks
# (Player.cs:5770-5786), the door closes in 320 ticks (ShelterDoor.Close, closeSpeed 1/320) and the cycle reloads.
MAX_STEPS = 1500

# Walking on the way the pipe spat the slugcat out takes it to the far wall, well over the 6 tiles from the
# entrance that the game requires before the door closes (Player.cs:5768).
WALK_STEPS = 80


def _sleep_in_shelter(renv, food, rest):
    """
    ENTER_SHELTER with ``food`` pips, wait until the slugcat pops out of the shelter's entrance pipe, walk on for
    WALK_STEPS, then hold ``rest`` until READY has dropped for the cycle reload and come back. Returns the infos.
    """
    pos = renv.step(NOOP)[4]["player_pos"]
    renv.unwrapped.debug_enter_shelter(food)
    infos, out, walk, reloading = [], None, NOOP, False
    for i in range(MAX_STEPS):
        action = NOOP if out is None or i <= out + 1 else walk if i <= out + 1 + WALK_STEPS else rest
        info = renv.step(action)[4]
        infos.append(info)
        if out is None and info["in_shelter"] and info["player_pos"] != pos:     # frozen while in the pipe
            out = i
        elif out is not None and i == out + 1:
            walk = LEFT if info["player_pos"][0] < infos[-2]["player_pos"][0] else RIGHT
        reloading |= not info["ready"]
        if reloading and info["ready"]:
            return infos
    raise AssertionError(f"no sleep and reload within {MAX_STEPS} steps (out of the pipe at {out}); last {infos[-1]}")


def test_fed_sleep_reports_cycle_survived_once(env):
    """
    With food_to_hibernate pips the game hibernates the slugcat (ShelterDoor.DoorClosed -> RainWorldGame.Win with
    malnourished = false): exactly one step reports cycle_survived, the Sleep term pays on it, and the next cycle
    starts in the shelter with cycle_number + 1 and karma + 1 (up to the cap).
    """
    from rainworld_rl.rewards import DriveReward

    renv = DriveReward(env)
    _obs, start = renv.reset()                     # fresh save: cycle 0, no food, outside any shelter
    infos = _sleep_in_shelter(renv, start["food_to_hibernate"], NOOP)
    infos += [renv.step(NOOP)[4] for _ in range(30)]

    edges = [i for i, info in enumerate(infos) if info["cycle_survived"]]
    assert len(edges) == 1, f"cycle_survived on steps {edges}"
    assert infos[edges[0]]["reward_terms"]["Sleep"] > 0, infos[edges[0]]
    after = infos[-1]
    assert after["ready"] and after["in_shelter"], after
    assert after["cycle_number"] == start["cycle_number"] + 1, after
    assert after["karma"] == min(start["karma"] + 1, start["karma_cap"]), after
    assert after["malnourished"] is False, after
    print(f"[sleep] fed: edge at step {edges[0]}, Sleep {infos[edges[0]]['reward_terms']['Sleep']:.2e}, "
          f"cycle {start['cycle_number']}->{after['cycle_number']}, karma {start['karma']}->{after['karma']}, "
          f"food {after['food']}")


def test_starving_sleep_is_not_cycle_survived(env):
    """
    With 1 pip (enough not to starve to death, ShelterDoor.cs:1583) holding DOWN forces a starving sleep
    (RainWorldGame.Win with malnourished = true): no cycle_survived and no Sleep reward, but the next cycle starts in
    the shelter with cycle_number + 1, malnourished and food_to_hibernate == food_max.
    """
    from rainworld_rl.rewards import DriveReward

    renv = DriveReward(env)
    _obs, start = renv.reset()
    infos = _sleep_in_shelter(renv, 1, DOWN)

    assert not any(info["cycle_survived"] for info in infos)
    assert all(info["reward_terms"]["Sleep"] == 0 for info in infos)
    after = infos[-1]
    assert after["ready"] and after["in_shelter"], after
    assert after["cycle_number"] == start["cycle_number"] + 1, after
    assert after["malnourished"] is True, after
    assert after["food_to_hibernate"] == after["food_max"], after
    print(f"[sleep] starving: {len(infos)} steps, cycle {start['cycle_number']}->{after['cycle_number']}, "
          f"karma {start['karma']}->{after['karma']}, food {after['food']}")
