"""Room and region transitions through the game's own code (debug HOP_ROOM / SWITCH_REGION commands)."""

from __future__ import annotations

from tests.harness import LEFT, NOOP, RIGHT

# Steps (1 tick each) bound for one leg of a hop: the next room loads (usually already realized as a neighbour),
# then the slugcat crosses the pipe. Both take a few to a few dozen ticks.
MAX_HOP_STEPS = 400

# Steps bound for the next region's world to load after SWITCH_REGION (~200 ticks at the mod's step speed).
MAX_REGION_STEPS = 3000


def _step_until(env, pred, max_steps, what):
    for _ in range(max_steps):
        info = env.step(NOOP)[4]
        if pred(info):
            return info
    raise AssertionError(f"{what} not within {max_steps} steps; last {info}")


def _out_of_pipe(env, info):
    """Steps until the slugcat comes out of the pipe: its position stays frozen while it is inside."""
    pos = info["player_pos"]
    return _step_until(env, lambda i: i["player_pos"] != pos, MAX_HOP_STEPS, "out of the pipe")


def _hop(env, exit):
    before = env.step(NOOP)[4]
    env.unwrapped.debug_hop_room(exit)
    moved = _step_until(env, lambda i: i["room_index"] != before["room_index"], MAX_HOP_STEPS,
                        f"room change from {before['region']}/{before['room_index']}")
    after = _out_of_pipe(env, moved)
    assert after["ready"] and not after["player_dead"], after
    assert after["region"] == before["region"], after
    return after


def _assert_playable(env):
    """
    Right after coming out of a pipe: alive, READY and taking input. Walks on the way the pipe spat the slugcat out,
    so it does not walk back into that pipe (the next command would find it between rooms).
    """
    a, b = env.step(NOOP)[4], env.step(NOOP)[4]
    walk = LEFT if b["player_pos"][0] < a["player_pos"][0] else RIGHT
    infos = [env.step(walk)[4] for _ in range(20)]
    assert all(i["ready"] and not i["player_dead"] for i in infos), infos[-1]
    assert len({i["player_pos"] for i in infos}) > 1, infos[-1]


def test_room_hops(env):
    """Five hops in a row each land in a different room of the same region, READY all along."""
    rooms = [env.step(NOOP)[4]["room_index"]]
    for exit in range(5):
        info = _hop(env, exit)
        assert info["room_index"] != rooms[-1], info
        rooms.append(info["room_index"])
    _assert_playable(env)
    print(f"[transitions] rooms {rooms} in region {info['region']}")


def test_switch_region_then_hops(env):
    """
    SWITCH_REGION: the region changes once the next world has loaded, then the slugcat leaves the gate room into the
    new region and takes input again; two hops there stay in the new region.
    """
    start = env.step(NOOP)[4]
    env.unwrapped.debug_switch_region()
    switched = _step_until(env, lambda i: i["region"] != start["region"], MAX_REGION_STEPS, "region change")
    left_gate = _step_until(env, lambda i: i["room_index"] != switched["room_index"], MAX_HOP_STEPS, "leaving the gate room")
    arrived = _out_of_pipe(env, left_gate)
    assert arrived["ready"] and arrived["region"] == switched["region"], arrived
    rooms = [arrived["room_index"]] + [_hop(env, exit)["room_index"] for exit in range(2)]
    _assert_playable(env)
    print(f"[transitions] {start['region']} -> {switched['region']} via gate room {switched['room_index']}, rooms {rooms}")
