"""player_dead edge flag and the death -> respawn flow (driven by the debug KILL_PLAYER command)."""

from __future__ import annotations

import math

from tests.e2e.helpers import infos, step_n

# Upper bound on the steps we are willing to spend from the kill to the slugcat being back in
# play. Budget: 1 step with the edge, ~40 ticks of the game's "game over" prompt (the mod
# presses the restart key after exactly the prompt's own minimum of 40 ticks), a 0.05 s fade,
# then the reload of the cycle at 1x real time while READY is clear (a few seconds -> a few
# dozen steps, each step being one FixedUpdate at the mod's speed multiplier).
MAX_STEPS_AFTER_KILL = 400

# Steps to keep going after READY returns, so the respawned slugcat has settled and we can be
# sure the death edge does not fire again for the new player instance.
SETTLE_STEPS = 30

# The slugcat spawns at the shelter's den position both after RESET (New game) and after a
# death reload (Load of the start-of-cycle save), but it may have fallen / shuffled a bit
# between the first frame and the moment we sample, so compare positions loosely.
SPAWN_POS_TOLERANCE_PX = 150.0


def test_debug_kill_death_edge_and_respawn(env):
    """
    Kill the slugcat with the debug KILL_PLAYER command and watch the whole death -> respawn
    sequence through ordinary no-op steps.

    Expectations (see docs/PROTOCOL.md "Commands" / "Status bits" and GameFlowController.cs):

    * KILL_PLAYER is applied synchronously, so the very first step after the ack carries the
      PLAYER_DEAD edge, and no other step in the sequence does (edge, not level; the respawned
      player is a new instance and must not re-trigger it).
    * The game then sits in its "game over" prompt for 40 ticks with the dead slugcat still in
      the game (READY stays set), after which the mod calls GoToDeathScreen; the DeathScreen
      request is rewritten into a reload of the cycle. During the fade + reload READY is clear
      for a while, then it comes back.
    * Every step in the sequence is serviced (no StepTimeoutError, step_counter +1 each time),
      including the ones while READY is clear.
    * After the respawn the slugcat is back in the start-of-cycle shelter: same room_index as
      right after reset, roughly the same spawn position.
    * Karma follows the game's own death rule (DeathPersistentSaveData.SaveToString writes
      karma-1 unless reinforced, clamped to >= 0 on load) so it can only go down or stay.
      cycle_number does NOT change: a death never re-saves the world state (SaveState.cycleNumber
      only advances in RainCycleTick, which runs for survived cycles only), and on a never-slept
      fresh save there is not even a SAVE STATE line yet, so the reload is a fresh cycle 0.
    """
    # Reset here (not via fresh_env) so we hold the info of the very first post-reset frame.
    _obs, at_reset = env.reset()
    assert at_reset["ready"] and at_reset["in_game"], f"not in game after reset: {at_reset}"
    room_at_reset = int(at_reset["room_index"])
    pos_at_reset = tuple(float(c) for c in at_reset["player_pos"])
    assert room_at_reset >= 0

    # A short alive baseline right before the kill.
    baseline = step_n(env, 5)
    assert not any(infos(baseline, "player_dead")), "died during the alive baseline"
    before = baseline[-1][1]
    assert before["ready"] and before["in_game"]
    karma_before = int(before["karma"])
    cycle_before = int(before["cycle_number"])
    counter_before = int(before["step_counter"])

    env.debug_kill()

    # Step through death -> game over prompt -> (skipped) death screen -> reload -> READY.
    results = []
    saw_not_ready = False
    respawn_index = None
    for i in range(MAX_STEPS_AFTER_KILL):
        obs, _reward, terminated, truncated, info = env.step(0)  # a timeout raises -> fails the test
        assert terminated is False and truncated is False, "continual env: death must not end an episode"
        assert obs.shape == tuple(env.observation_space.shape), f"bad frame {obs.shape} at step {i}"
        results.append((obs, info))
        if not info["ready"]:
            saw_not_ready = True
        elif saw_not_ready and int(info["room_index"]) >= 0:
            respawn_index = i
            break

    assert saw_not_ready, (
        f"READY never dropped within {MAX_STEPS_AFTER_KILL} steps after the kill - the death screen "
        f"skip / reload did not happen (player_dead seen at "
        f"{[i for i, d in enumerate(infos(results, 'player_dead')) if d]})"
    )
    assert respawn_index is not None, (
        f"READY did not come back within {MAX_STEPS_AFTER_KILL} steps after the kill "
        f"(last info: {results[-1][1]})"
    )

    # Let the new slugcat settle and make sure nothing re-triggers the edge.
    results += step_n(env, SETTLE_STEPS)

    player_dead = [bool(v) for v in infos(results, "player_dead")]
    ready = [bool(v) for v in infos(results, "ready")]
    in_game = [bool(v) for v in infos(results, "in_game")]
    counters = [int(v) for v in infos(results, "step_counter")]

    # -- death edge: exactly one step, and it is the first one after the (synchronous) kill
    dead_at = [i for i, d in enumerate(player_dead) if d]
    assert dead_at == [0], f"player_dead edge expected exactly at step 0 after the kill, got steps {dead_at}"

    # -- READY drops for a while and comes back; the dead slugcat is still "in game" for the
    #    40-tick game-over prompt first, so the drop starts after the edge, not on it.
    not_ready_at = [i for i, r in enumerate(ready) if not r]
    assert not_ready_at, "READY never dropped"
    assert not_ready_at[0] > 0, "READY dropped on the very step of the death edge (expected the game-over prompt first)"
    assert ready[respawn_index] and all(ready[respawn_index:]), (
        f"READY flapped after the respawn at step {respawn_index}: {ready[respawn_index:]}"
    )
    assert not_ready_at[-1] < respawn_index
    # in_game is the coarser signal (main process is a RainWorldGame): the Game -> Game switch
    # replaces the process inside a single ProcessManager.Update, so it may or may not be
    # observed as a dip depending on when the frame is captured; it must be set whenever READY
    # is, and must be set again once we are back.
    assert all(g for g, r in zip(in_game, ready) if r), "ready without in_game"
    assert in_game[-1], "not in_game after the respawn"

    # -- every step was serviced, also while READY was clear
    expected = list(range(counter_before + 1, counter_before + 1 + len(results)))
    assert counters == expected, (
        f"step_counter skipped/stalled across the death sequence: "
        f"first mismatch at index {next(i for i, (a, b) in enumerate(zip(counters, expected)) if a != b)}"
    )

    # -- game-state fields: READY clears as soon as the process switch is *requested*, while the
    #    old RainWorldGame (dead slugcat, room, cycle) is still current for the fade-out, and the
    #    new one exposes its abstract player before it is realized - so fields may still be valid
    #    while not READY. The protocol only promises the converse: cleared fields (-1) are never
    #    paired with READY.
    cleared_at = [i for i, (_o, info) in enumerate(results) if int(info["room_index"]) == -1]
    assert all(not ready[i] for i in cleared_at), f"READY set while fields were cleared at {cleared_at}"
    assert all(int(results[i][1]["cycle_number"]) == -1 for i in cleared_at), "room cleared but cycle not"

    # -- respawn: back in the start-of-cycle shelter
    after = results[-1][1]
    assert after["ready"] and after["in_game"]
    assert int(after["room_index"]) == room_at_reset, (
        f"respawned in room {after['room_index']}, start-of-cycle shelter is room {room_at_reset}"
    )
    pos_after = tuple(float(c) for c in after["player_pos"])
    dist = math.dist(pos_after, pos_at_reset)
    assert dist <= SPAWN_POS_TOLERANCE_PX, (
        f"respawn position {pos_after} is {dist:.0f}px from the post-reset spawn {pos_at_reset}"
    )

    # -- karma / cycle (see docstring)
    karma_after = int(after["karma"])
    cycle_after = int(after["cycle_number"])
    assert 0 <= karma_after <= karma_before, f"karma went {karma_before} -> {karma_after} across a death"
    assert cycle_after == cycle_before, (
        f"cycle_number went {cycle_before} -> {cycle_after} across a death; a death must not advance the cycle"
    )

    print(
        f"[death] edge at step 0; READY clear for steps {not_ready_at[0]}..{not_ready_at[-1]} "
        f"({len(not_ready_at)} steps); respawned at step {respawn_index} "
        f"(in_game dipped: {not all(in_game)}); karma {karma_before}->{karma_after}, "
        f"cycle {cycle_before}->{cycle_after}, room {room_at_reset}->{after['room_index']}, "
        f"spawn offset {dist:.1f}px"
    )
