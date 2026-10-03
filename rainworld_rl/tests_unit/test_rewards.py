"""
Unit tests for ``rainworld_rl.rewards`` using hand-written ``info`` sequences.
No game is needed: ``ScriptedEnv`` replays a list of info dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rainworld_rl.rewards import (
    Alive,
    Ate,
    CycleSurvived,
    Death,
    FunctionTerm,
    InfoReward,
    NewRoom,
    RewardTerm,
    SurviveCycleReward,
    default_terms,
    make_default_reward_env,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def info(**overrides: Any) -> Dict[str, Any]:
    """A fully populated, 'alive and in game' info dict with the given overrides."""
    base = dict(
        player_dead = False, karma = 1, karma_cap = 5, food = 0, food_max = 7, food_to_hibernate = 4,
        malnourished = False, player_pos = (0.0, 0.0), room_index = 10, cycle_number = 0,
        step_counter = 0, in_game = True, ready = True, human_override = False,
        in_shelter = False, cycle_survived = False, rain = False, dialog_open = False,
        cycle_progress = 0.1,
    )
    base.update(overrides)
    return base


def not_ready() -> Dict[str, Any]:
    """What the mod reports while not READY: zeros / -1, in_game False."""
    return info(ready = False, in_game = False, food = 0, food_max = 0, food_to_hibernate = 0,
                room_index = -1, cycle_number = -1, cycle_progress = 0.0)


def run(term: RewardTerm, seq: List[Dict[str, Any]]) -> List[float]:
    """reset on seq[0], then call the term on each consecutive pair."""
    term.reset(seq[0])
    return [term(prev, cur) for prev, cur in zip(seq, seq[1:])]


class ScriptedEnv(gym.Env):
    """Replays ``infos``: reset() returns infos[0], each step() the next one. Reward is always 0."""

    def __init__(self, infos: List[Dict[str, Any]], env_reward: float = 0.0):
        self.infos = infos
        self.env_reward = env_reward
        self.i = 0
        self.resets = 0
        self.observation_space = spaces.Box(0, 255, (2, 2, 3), np.uint8)
        self.action_space = spaces.Discrete(18)

    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)
        self.resets += 1
        self.i = 0
        return np.zeros((2, 2, 3), np.uint8), dict(self.infos[0])

    def step(self, action):
        self.i += 1
        return np.zeros((2, 2, 3), np.uint8), self.env_reward, False, False, dict(self.infos[self.i])


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------

def test_cycle_survived_fires_once_per_edge_and_scales():
    seq = [info(), info(), info(cycle_survived = True, cycle_number = 1), info(cycle_number = 1), info(cycle_survived = True, cycle_number = 2)]
    assert run(CycleSurvived(), seq) == [0.0, 1.0, 0.0, 1.0]
    assert run(CycleSurvived(weight = 5.0), seq) == [0.0, 5.0, 0.0, 5.0]


def test_cycle_survived_counts_even_when_not_ready_that_step():
    # The edge may land on the frame where the SleepScreen redirect already left the game.
    gone = not_ready(); gone["cycle_survived"] = True
    assert run(CycleSurvived(), [info(), gone, info(cycle_number = 1)]) == [1.0, 0.0]


def test_death_is_negative_by_default_and_edge_only():
    seq = [info(), info(player_dead = True), info(), info(player_dead = True), info()]
    assert run(Death(), seq) == [-1.0, 0.0, -1.0, 0.0]
    assert run(Death(weight = -3.5), seq) == [-3.5, 0.0, -3.5, 0.0]


def test_new_room_rewards_first_visits_only_and_ignores_start_room():
    seq = [info(room_index = 10), info(room_index = 10), info(room_index = 11), info(room_index = 11),
           info(room_index = 10), info(room_index = 12), info(room_index = 11)]
    term = NewRoom()
    assert run(term, seq) == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert term.visited == {10, 11, 12}


def test_new_room_ignores_not_ready_and_invalid_rooms():
    seq = [info(room_index = 10), not_ready(), info(room_index = -1), info(room_index = 10), info(room_index = 11)]
    assert run(NewRoom(weight = 2.0), seq) == [0.0, 0.0, 0.0, 2.0]


def test_new_room_visited_set_clears_on_reset():
    term = NewRoom()
    assert run(term, [info(room_index = 1), info(room_index = 2)]) == [1.0]
    assert run(term, [info(room_index = 5), info(room_index = 2)]) == [1.0]   # room 2 is new again
    assert term.visited == {5, 2}


def test_ate_counts_pips_gained_within_a_cycle_only():
    seq = [info(food = 0), info(food = 1), info(food = 3), info(food = 3), info(food = 2),
           info(food = 0, cycle_number = 1), info(food = 1, cycle_number = 1)]
    assert run(Ate(), seq) == [1.0, 2.0, 0.0, 0.0, 0.0, 1.0]
    assert run(Ate(weight = 0.5), seq) == [0.5, 1.0, 0.0, 0.0, 0.0, 0.5]


def test_ate_ignores_cycle_change_and_not_ready_jumps():
    # food reads 0 while not READY; coming back with 4 pips is not eating.
    seq = [info(food = 4), not_ready(), info(food = 4), info(food = 5)]
    assert run(Ate(), seq) == [0.0, 0.0, 1.0]
    # a new cycle that happens to start with more food is not eating either
    seq = [info(food = 1), info(food = 3, cycle_number = 1)]
    assert run(Ate(), seq) == [0.0]


def test_alive_per_step_except_death_and_not_ready():
    seq = [info(), info(), info(player_dead = True), not_ready(), info()]
    assert run(Alive(weight = 0.1), seq) == pytest.approx([0.1, 0.0, 0.0, 0.1])
    assert Alive().weight == 0.01


def test_function_term_wraps_callable_and_names_it():
    def karma_delta(prev, cur):
        return cur["karma"] - prev["karma"]

    term = FunctionTerm(karma_delta, weight = 2.0)
    assert term.name == "karma_delta"
    assert run(term, [info(karma = 1), info(karma = 2), info(karma = 0)]) == [2.0, -4.0]
    assert FunctionTerm(lambda p, c: 1.0, name = "one").name == "one"


def test_term_repr_and_default_name():
    assert repr(CycleSurvived()) == "CycleSurvived(weight = 1.0)"
    assert CycleSurvived().name == "CycleSurvived"
    assert CycleSurvived(name = "survive").name == "survive"


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

def test_info_reward_sums_terms_and_reports_breakdown():
    seq = [info(room_index = 1), info(room_index = 2), info(room_index = 2, player_dead = True),
           info(room_index = 3, cycle_survived = True)]
    env = InfoReward(ScriptedEnv(seq), [CycleSurvived(1.0), Death(-1.0), NewRoom(0.5)])
    _obs, first = env.reset()
    assert "reward_terms" not in first

    rewards, breakdowns = [], []
    for _ in range(3):
        _obs, r, term, trunc, inf = env.step(0)
        assert (term, trunc) == (False, False)
        rewards.append(r)
        breakdowns.append(inf["reward_terms"])

    assert rewards == pytest.approx([0.5, -1.0, 1.5])
    assert breakdowns[0] == {"CycleSurvived": 0.0, "Death": 0.0, "NewRoom": 0.5}
    assert breakdowns[2] == {"CycleSurvived": 1.0, "Death": 0.0, "NewRoom": 0.5}
    assert all(isinstance(r, float) for r in rewards)


def test_info_reward_resets_terms_on_env_reset():
    seq = [info(room_index = 1), info(room_index = 2), info(room_index = 1)]
    env = InfoReward(ScriptedEnv(seq), [NewRoom()])
    env.reset()
    assert env.step(0)[1] == 1.0          # room 2 new
    env.reset()                           # visited cleared, start room 1 re-recorded
    assert env.step(0)[1] == 1.0          # room 2 new again
    assert env.step(0)[1] == 0.0          # back to room 1 (start room, visited)


def test_info_reward_replaces_env_reward_unless_asked():
    seq = [info(), info(cycle_survived = True)]
    assert InfoReward(ScriptedEnv(seq, env_reward = 10.0), [CycleSurvived()]).reset() is not None
    env = InfoReward(ScriptedEnv(seq, env_reward = 10.0), [CycleSurvived()])
    env.reset()
    assert env.step(0)[1] == 1.0
    env = InfoReward(ScriptedEnv(seq, env_reward = 10.0), [CycleSurvived()], add_env_reward = True)
    env.reset()
    assert env.step(0)[1] == 11.0


def test_info_reward_breakdown_key_can_be_disabled_or_renamed():
    seq = [info(), info()]
    env = InfoReward(ScriptedEnv(seq), [Alive(1.0)], breakdown_key = None)
    env.reset()
    assert "reward_terms" not in env.step(0)[4]
    env = InfoReward(ScriptedEnv(seq), [Alive(1.0)], breakdown_key = "terms")
    env.reset()
    assert env.step(0)[4]["terms"] == {"Alive": 1.0}


def test_info_reward_step_before_reset_uses_first_info_as_baseline():
    seq = [info(food = 2), info(food = 3), info(food = 5)]
    env = InfoReward(ScriptedEnv(seq), [Ate()])
    # No reset(): the first step's info becomes the baseline, so no delta is credited for it.
    assert env.step(0)[1] == 0.0
    assert env.step(0)[1] == 2.0


def test_info_reward_rejects_empty_or_duplicate_terms():
    with pytest.raises(ValueError):
        InfoReward(ScriptedEnv([info()]), [])
    with pytest.raises(ValueError):
        InfoReward(ScriptedEnv([info()]), [Alive(), Alive()])
    InfoReward(ScriptedEnv([info()]), [Alive(name = "a"), Alive(name = "b")])  # ok


def test_prev_info_is_a_copy():
    seq = [info(food = 1), info(food = 2)]
    env = InfoReward(ScriptedEnv(seq), [Ate()])
    _obs, first = env.reset()
    first["food"] = 99                     # mutating the returned dict must not affect the baseline
    assert env.prev_info["food"] == 1
    assert env.step(0)[1] == 1.0


# ---------------------------------------------------------------------------
# Default
# ---------------------------------------------------------------------------

def test_default_reward_is_only_cycle_survived():
    terms = default_terms()
    assert [type(t) for t in terms] == [CycleSurvived] and terms[0].weight == 1.0
    assert default_terms() is not terms                       # fresh instances every time

    seq = [info(), info(player_dead = True), info(room_index = 99, food = 5), info(cycle_survived = True)]
    env = make_default_reward_env(ScriptedEnv(seq))
    assert isinstance(env, SurviveCycleReward) and isinstance(env, InfoReward)
    env.reset()
    rewards = [env.step(0)[1] for _ in range(3)]
    assert rewards == [0.0, 0.0, 1.0]      # death, exploring and eating are worth nothing; surviving is +1


def test_default_reward_ignores_starving_sleep():
    # A starving sleep increments cycle_number and sets malnourished, but the mod does not set
    # cycle_survived - so the default reward stays 0.
    seq = [info(in_shelter = True), info(cycle_number = 1, malnourished = True, food_to_hibernate = 7)]
    env = make_default_reward_env(ScriptedEnv(seq))
    env.reset()
    assert env.step(0)[1] == 0.0
