"""player_dead edge flag."""

from __future__ import annotations

import pytest

from tests.harness import infos, step_n


def test_no_death_at_start(fresh_env):
    """Right after a fresh reset the slugcat is safe; 100 no-op steps never report a death."""
    results = step_n(fresh_env, 100)
    dead_at = [i for i, dead in enumerate(infos(results, "player_dead")) if dead]
    assert dead_at == [], f"player_dead was True at steps {dead_at[:10]}"
    assert all(bool(v) for v in infos(results, "in_game")), "left the game during 100 idle steps"


@pytest.mark.skip(reason="needs a scripted death; TODO once a kill command or hazard room exists")
def test_death_edge_is_exactly_one_step(fresh_env):
    """
    Structure for the death-edge test once a way to kill the slugcat exists
    (e.g. a KILL command byte in the header, or a RESET option that spawns in a
    hazard room / next to a lizard with a scripted drop):

    1. fresh_env -> reset(); confirm player_dead is False for a few steps.
    2. Trigger the death (send the kill command, or walk into the hazard with
       step_n(env, N, lambda i: ACTION_RIGHT)).
    3. Step until info["player_dead"] is True (bounded, e.g. <= 300 steps);
       record that step index k.
    4. PROTOCOL.md: PLAYER_DEAD is an EDGE - set for exactly one step. Assert
       results[k+1].info["player_dead"] is False and stays False for the next ~20 steps.
    5. in_game should toggle: the mod leaves RainWorldGame for the death screen
       (in_game False, game-state fields zeroed/-1) and then auto-returns to the
       game within a few frames (in_game True again). Assert both edges happen,
       and that steps keep being serviced throughout (no timeout, step_counter +1).
    6. Optionally assert karma dropped by one and the cycle restarted.
    """
    raise NotImplementedError
