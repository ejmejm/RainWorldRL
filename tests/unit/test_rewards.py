"""
Unit tests for ``rainworld_rl.rewards`` using hand-written ``info`` sequences.
No game is needed: ``ScriptedEnv`` replays a list of info dicts.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rainworld_rl.rewards import (
    NOMINAL_CYCLE_TICKS,
    Alive,
    Ate,
    CycleSurvived,
    Death,
    DriveReward,
    Food,
    FunctionTerm,
    InfoReward,
    Malnourished,
    NewRoom,
    RewardTerm,
    Sleep,
    SurviveCycleReward,
    drive_terms,
    make_default_reward_env,
    nominal_cycle_steps,
    room_key,
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
        cycle_progress = 0.1, region = "SU",
    )
    base.update(overrides)
    return base


def not_ready() -> Dict[str, Any]:
    """What the mod reports while not READY: zeros / -1, in_game False."""
    return info(ready = False, in_game = False, food = 0, food_max = 0, food_to_hibernate = 0,
                room_index = -1, cycle_number = -1, cycle_progress = 0.0, region = "")


def run(term: RewardTerm, seq: List[Dict[str, Any]]) -> List[float]:
    """reset on seq[0], then call the term on each consecutive pair."""
    term.reset(seq[0])
    return [term(prev, cur) for prev, cur in zip(seq, seq[1:])]


def repeat(step: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    return [dict(step) for _ in range(n)]


class ScriptedEnv(gym.Env):
    """Replays ``infos``: reset() returns infos[0], each step() the next one. Reward is always 0."""

    def __init__(self, infos: List[Dict[str, Any]], env_reward: float = 0.0, ticks_per_step: int = 1):
        self.infos = infos
        self.env_reward = env_reward
        self.ticks_per_step = ticks_per_step
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
# Nominal cycle
# ---------------------------------------------------------------------------

def test_nominal_cycle_steps_derives_from_ticks_per_step():
    assert nominal_cycle_steps(1) == NOMINAL_CYCLE_TICKS
    assert nominal_cycle_steps(4) == NOMINAL_CYCLE_TICKS // 4
    assert nominal_cycle_steps(7) == round(NOMINAL_CYCLE_TICKS / 7)
    assert nominal_cycle_steps(0) == NOMINAL_CYCLE_TICKS   # 0 is treated as 1, like the mod does
    assert nominal_cycle_steps(10 ** 9) == 1


def test_room_key_uses_region_and_ignores_not_ready():
    assert room_key(info(region = "SU", room_index = 5)) == ("SU", 5)
    assert room_key(info(region = "", room_index = 5)) == ("", 5)
    assert room_key(info(room_index = -1)) is None
    assert room_key(not_ready()) is None


# ---------------------------------------------------------------------------
# NewRoom
# ---------------------------------------------------------------------------

def test_new_room_rewards_first_visits_only_and_ignores_start_room():
    seq = [info(room_index = 10), info(room_index = 10), info(room_index = 11), info(room_index = 11),
           info(room_index = 10), info(room_index = 12), info(room_index = 11)]
    term = NewRoom()
    assert run(term, seq) == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert term.visited == {("SU", 10), ("SU", 11), ("SU", 12)}
    # not-ready and invalid rooms are ignored
    seq = [info(room_index = 10), not_ready(), info(room_index = -1), info(room_index = 10), info(room_index = 11)]
    assert run(NewRoom(weight = 2.0), seq) == [0.0, 0.0, 0.0, 2.0]


def test_new_room_is_keyed_by_region_so_same_index_in_two_regions_pays_twice():
    seq = [info(region = "SU", room_index = 3), info(region = "SU", room_index = 7),
           info(region = "HI", room_index = 7), info(region = "HI", room_index = 3),
           info(region = "SU", room_index = 7), info(region = "SU", room_index = 3)]
    term = NewRoom()
    assert run(term, seq) == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert term.visited == {("SU", 3), ("SU", 7), ("HI", 7), ("HI", 3)}


def test_new_room_visited_set_persists_across_death_and_respawn():
    seq = [info(room_index = 1), info(room_index = 2), info(room_index = 2, player_dead = True),
           not_ready(), not_ready(), info(room_index = 1), info(room_index = 2), info(room_index = 3)]
    assert run(NewRoom(), seq) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_new_room_visited_set_clears_on_reset():
    term = NewRoom()
    assert run(term, [info(room_index = 1), info(room_index = 2)]) == [1.0]
    assert run(term, [info(room_index = 5), info(room_index = 2)]) == [1.0]   # room 2 is new again
    assert term.visited == {("SU", 5), ("SU", 2)}


# ---------------------------------------------------------------------------
# Food
# ---------------------------------------------------------------------------

def test_food_pips_at_or_below_threshold_pay_below_and_above_pay_above():
    term = Food(below = 0.3, above = 0.1, satiety_decay = 1.0, recovery_steps = 100)  # no satiety here
    seq = [info(food = 0), info(food = 1), info(food = 4), info(food = 5), info(food = 7)]
    assert run(term, seq) == pytest.approx([0.3, 0.9, 0.1, 0.2])
    # the threshold comes from info: while malnourished the game needs food_max pips, so all pay 'below'
    seq = [info(food = 0, malnourished = True, food_to_hibernate = 7), info(food = 7, malnourished = True, food_to_hibernate = 7)]
    assert run(Food(satiety_decay = 1.0, recovery_steps = 100), seq) == pytest.approx([7 * 0.3])


def test_food_satiety_decays_per_pip_in_the_same_room():
    term = Food(below = 0.3, above = 0.1, satiety_decay = 0.5, recovery_steps = 10 ** 9)  # effectively no recovery
    seq = [info(food = 0), info(food = 1), info(food = 2), info(food = 4)]
    rewards = run(term, seq)
    assert rewards == pytest.approx([0.3, 0.15, 0.3 * 0.25 + 0.3 * 0.125])
    assert term.satiety(("SU", 10)) == pytest.approx(0.5 ** 4)
    assert term.satiety(("SU", 11)) == 1.0                   # another room is untouched


def test_food_satiety_recovers_linearly_in_real_steps_not_cycles():
    term = Food(below = 1.0, above = 1.0, satiety_decay = 0.5, recovery_steps = 100)
    term.reset(info(food = 0))
    assert term(info(food = 0), info(food = 1)) == pytest.approx(1.0)     # factor now 0.5
    assert term.satiety(("SU", 10)) == pytest.approx(0.5)
    # 25 idle steps -> +0.25
    for _ in range(25):
        assert term(info(food = 1), info(food = 1)) == 0.0
    assert term.satiety(("SU", 10)) == pytest.approx(0.75)
    # sleeping (cycle change, food drop) does NOT refresh: the step across the cycle is just one step
    assert term(info(food = 1), info(food = 0, cycle_number = 1)) == 0.0
    assert term.satiety(("SU", 10)) == pytest.approx(0.76)
    # non-ready steps still count as time passing
    for _ in range(24):
        assert term(not_ready(), not_ready()) == 0.0
    assert term.satiety(("SU", 10)) == pytest.approx(1.0)
    for _ in range(10):
        term(info(cycle_number = 1), info(cycle_number = 1))
    assert term.satiety(("SU", 10)) == 1.0                                # clamped at fresh
    assert term(info(food = 0, cycle_number = 1), info(food = 1, cycle_number = 1)) == pytest.approx(1.0)
    # recovery is applied before the next pip is paid
    term = Food(below = 1.0, above = 1.0, satiety_decay = 0.5, recovery_steps = 10)
    term.reset(info())
    term(info(food = 0), info(food = 1))                      # factor 0.5 at step 1
    for _ in range(5):
        term(info(food = 1), info(food = 1))                  # +0.5 -> 1.0 at step 6
    assert term(info(food = 1), info(food = 2)) == pytest.approx(1.0)


def test_food_ignores_decreases_cycle_changes_and_not_ready_jumps():
    term = Food(satiety_decay = 1.0, recovery_steps = 100)
    seq = [info(food = 4), info(food = 2), info(food = 0, cycle_number = 1), not_ready(), info(food = 4, cycle_number = 1),
           info(food = 0, cycle_number = 1, player_dead = True), info(food = 1, cycle_number = 1)]
    assert run(term, seq) == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.3])


def test_food_reset_clears_satiety_and_weight_scales():
    term = Food(below = 1.0, satiety_decay = 0.5, recovery_steps = 10 ** 9, weight = 2.0)
    assert run(term, [info(food = 0), info(food = 1)]) == pytest.approx([2.0])
    assert run(term, [info(food = 0), info(food = 1)]) == pytest.approx([2.0])   # fresh after reset
    with pytest.raises(ValueError):
        Food(satiety_decay = 1.5)
    with pytest.raises(ValueError):
        Food(recovery_steps = 0)


# ---------------------------------------------------------------------------
# Sleep
# ---------------------------------------------------------------------------

def sleep_seq(awake_steps: int, food_before: int, cycle: int = 0, survived: bool = True) -> List[Dict[str, Any]]:
    """Awake for ``awake_steps`` steps with ``food_before`` pips, then the sleep edge, then the new cycle."""
    seq = repeat(info(food = food_before, cycle_number = cycle), awake_steps + 1)
    if survived:
        seq.append(info(food = food_before, cycle_number = cycle, cycle_survived = True))
    seq.append(not_ready())
    seq.append(info(food = max(0, food_before - 4) if survived else 0, cycle_number = cycle + 1,
                    malnourished = not survived, food_to_hibernate = 7 if not survived else 4))
    return seq


def test_sleep_scales_with_tiredness_after_ten_percent_of_a_cycle_pays_one_percent():
    term = Sleep(weight = 1.0, full_belly_per_pip = 0.25, nominal_cycle_steps = 1000)
    rewards = run(term, sleep_seq(awake_steps = 99, food_before = 4))     # 99 steps + the edge step = 100 awake
    assert sum(rewards) == pytest.approx(0.10 ** 2)   # convex sleep pressure: 10% awake -> 1%
    assert rewards[-3] == pytest.approx(0.10 ** 2) and rewards[-1] == 0.0 and rewards[-2] == 0.0


def test_sleep_tiredness_is_capped_at_one_full_cycle():
    term = Sleep(nominal_cycle_steps = 100)
    rewards = run(term, sleep_seq(awake_steps = 1000, food_before = 4))
    assert sum(rewards) == pytest.approx(1.0)
    assert term.steps_awake <= 2                                           # reset by the sleep


def test_sleep_full_belly_bonus_per_pip_above_threshold_uses_food_before_the_edge():
    term = Sleep(weight = 1.0, full_belly_per_pip = 0.25, nominal_cycle_steps = 100)
    assert sum(run(term, sleep_seq(100, food_before = 7))) == pytest.approx(1.0 + 0.25 * 3)
    assert sum(run(term, sleep_seq(100, food_before = 4))) == pytest.approx(1.0)
    # the edge may land on a non-ready frame: food is taken from the last ready info
    seq = repeat(info(food = 6), 101)
    edge = not_ready(); edge["cycle_survived"] = True
    seq += [edge, info(food = 2, cycle_number = 1)]
    assert sum(run(term, seq)) == pytest.approx(1.0 + 0.5)


def test_starving_sleep_earns_nothing_and_resets_tiredness():
    term = Sleep(nominal_cycle_steps = 100)
    seq = sleep_seq(awake_steps = 500, food_before = 2, survived = False)
    assert sum(run(term, seq)) == 0.0
    assert term.steps_awake <= 1
    # a cycle_number increment alone (no malnourished flag) also counts as a sleep for the tiredness clock
    term = Sleep(nominal_cycle_steps = 100)
    term.reset(info())
    for _ in range(500):
        term(info(), info())
    assert term.tiredness == 1.0
    term(info(), info(cycle_number = 1))
    assert term.steps_awake == 0 and term.tiredness == 0.0
    # malnourished turning True also resets
    term(info(cycle_number = 1), info(cycle_number = 1, malnourished = True, food_to_hibernate = 7))
    assert term.steps_awake == 0


def test_sleep_resets_steps_awake_on_env_reset_and_counts_non_ready_steps():
    term = Sleep(nominal_cycle_steps = 10)
    term.reset(info())
    for _ in range(5):
        term(info(), info())
    term.reset(info())
    assert term.steps_awake == 0
    for _ in range(5):
        term(not_ready(), not_ready())                                     # loading screens still take time
    assert term.tiredness == pytest.approx(0.5 ** 2)
    with pytest.raises(ValueError):
        Sleep(nominal_cycle_steps = 0)


def test_sleep_edge_pays_once_per_survived_cycle():
    term = Sleep(nominal_cycle_steps = 10)
    seq = sleep_seq(20, 4, cycle = 0) + sleep_seq(20, 4, cycle = 1)[1:]
    rewards = run(term, seq)
    assert [r for r in rewards if r > 0] == pytest.approx([1.0, 1.0])


# ---------------------------------------------------------------------------
# Death / Malnourished
# ---------------------------------------------------------------------------

def test_death_is_minus_three_by_default_and_once_per_edge():
    seq = [info(), info(player_dead = True), info(), not_ready(), info(player_dead = True), info()]
    assert run(Death(), seq) == [-3.0, 0.0, 0.0, -3.0, 0.0]
    assert run(Death(weight = -1.5), seq) == [-1.5, 0.0, 0.0, -1.5, 0.0]


def test_malnourished_per_step_while_ready():
    seq = [info(), info(malnourished = True), info(malnourished = True), not_ready(), info(malnourished = True), info()]
    assert run(Malnourished(), seq) == pytest.approx([-0.0003, -0.0003, 0.0, -0.0003, 0.0])


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def test_cycle_survived_term_and_survive_cycle_reward():
    seq = [info(), info(), info(cycle_survived = True, cycle_number = 1), info(cycle_number = 1), info(cycle_survived = True, cycle_number = 2)]
    assert run(CycleSurvived(), seq) == [0.0, 1.0, 0.0, 1.0]
    assert run(CycleSurvived(weight = 5.0), seq) == [0.0, 5.0, 0.0, 5.0]
    # The edge may land on the frame where the SleepScreen redirect already left the game.
    gone = not_ready(); gone["cycle_survived"] = True
    assert run(CycleSurvived(), [info(), gone, info(cycle_number = 1)]) == [1.0, 0.0]
    # SurviveCycleReward: +1 on a survived cycle only; a starving sleep (cycle advances, malnourished) pays nothing
    seq = [info(), info(player_dead = True), info(room_index = 99, food = 5), info(cycle_survived = True)]
    env = SurviveCycleReward(ScriptedEnv(seq))
    env.reset()
    assert [env.step(0)[1] for _ in range(3)] == [0.0, 0.0, 1.0]
    seq = [info(in_shelter = True), info(cycle_number = 1, malnourished = True, food_to_hibernate = 7)]
    env = SurviveCycleReward(ScriptedEnv(seq))
    env.reset()
    assert env.step(0)[1] == 0.0


def test_ate_counts_pips_gained_within_a_cycle_only():
    seq = [info(food = 0), info(food = 1), info(food = 3), info(food = 3), info(food = 2),
           info(food = 0, cycle_number = 1), info(food = 1, cycle_number = 1)]
    assert run(Ate(), seq) == [1.0, 2.0, 0.0, 0.0, 0.0, 1.0]
    assert run(Ate(weight = 0.5), seq) == [0.5, 1.0, 0.0, 0.0, 0.0, 0.5]
    # not-ready jumps and a cycle change are not eating
    seq = [info(food = 4), not_ready(), info(food = 4), info(food = 5)]
    assert run(Ate(), seq) == [0.0, 0.0, 1.0]
    seq = [info(food = 1), info(food = 3, cycle_number = 1)]
    assert run(Ate(), seq) == [0.0]


def test_alive_per_step_except_death_and_not_ready():
    seq = [info(), info(), info(player_dead = True), not_ready(), info()]
    assert run(Alive(weight = 0.1), seq) == pytest.approx([0.1, 0.0, 0.0, 0.1])


def test_function_term_wraps_callable_and_names_it():
    def karma_delta(prev, cur):
        return cur["karma"] - prev["karma"]

    term = FunctionTerm(karma_delta, weight = 2.0)
    assert term.name == "karma_delta"
    assert run(term, [info(karma = 1), info(karma = 2), info(karma = 0)]) == [2.0, -4.0]
    assert FunctionTerm(lambda p, c: 1.0, name = "one").name == "one"


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
    assert isinstance(env.term("NewRoom"), NewRoom)
    with pytest.raises(KeyError):
        env.term("nope")


def test_info_reward_resets_terms_on_env_reset():
    seq = [info(room_index = 1), info(room_index = 2), info(room_index = 1)]
    env = InfoReward(ScriptedEnv(seq), [NewRoom()])
    env.reset()
    assert env.step(0)[1] == 1.0          # room 2 new
    env.reset()                           # visited cleared, start room 1 re-recorded
    assert env.step(0)[1] == 1.0          # room 2 new again
    assert env.step(0)[1] == 0.0          # back to room 1 (start room, visited)


def test_info_reward_options_add_env_reward_and_breakdown_key():
    seq = [info(), info(cycle_survived = True)]
    env = InfoReward(ScriptedEnv(seq, env_reward = 10.0), [CycleSurvived()])
    env.reset()
    assert env.step(0)[1] == 1.0                           # the env's own reward is replaced ...
    env = InfoReward(ScriptedEnv(seq, env_reward = 10.0), [CycleSurvived()], add_env_reward = True)
    env.reset()
    assert env.step(0)[1] == 11.0                          # ... unless asked
    env = InfoReward(ScriptedEnv(seq), [CycleSurvived()], breakdown_key = None)
    env.reset()
    assert "reward_terms" not in env.step(0)[4]
    env = InfoReward(ScriptedEnv(seq), [CycleSurvived()], breakdown_key = "terms")
    env.reset()
    assert env.step(0)[4]["terms"] == {"CycleSurvived": 1.0}


def test_info_reward_edge_cases():
    with pytest.raises(ValueError):
        InfoReward(ScriptedEnv([info()]), [])
    with pytest.raises(ValueError):
        InfoReward(ScriptedEnv([info()]), [Alive(), Alive()])
    InfoReward(ScriptedEnv([info()]), [Alive(name = "a"), Alive(name = "b")])  # ok
    # step() before reset() uses the first info as the baseline
    env = InfoReward(ScriptedEnv([info(food = 2), info(food = 3), info(food = 5)]), [Ate()])
    assert env.step(0)[1] == 0.0
    assert env.step(0)[1] == 2.0
    # prev_info is a copy of what reset() returned
    env = InfoReward(ScriptedEnv([info(food = 1), info(food = 2)]), [Ate()])
    _obs, first = env.reset()
    first["food"] = 99
    assert env.prev_info["food"] == 1
    assert env.step(0)[1] == 1.0


# ---------------------------------------------------------------------------
# Default: DriveReward
# ---------------------------------------------------------------------------

def test_make_default_reward_env_returns_drive_reward_and_derives_ticks_per_step():
    seq = [info(room_index = 1), info(room_index = 2, food = 1), info(room_index = 2, player_dead = True),
           info(room_index = 2, malnourished = True)]
    env = make_default_reward_env(ScriptedEnv(seq, ticks_per_step = 4))
    assert isinstance(env, DriveReward) and isinstance(env, InfoReward)
    assert env.novelty is True and env.ticks_per_step == 4
    assert [t.name for t in env.terms] == ["NewRoom", "Food", "Sleep", "Death", "Malnourished"]
    assert env.term("Food").recovery_steps == 6000 and env.term("Sleep").nominal_cycle_steps == 6000
    env.reset()
    _o, r1, _t, _tr, i1 = env.step(0)
    assert r1 == pytest.approx(1.0 + 0.3)                          # new room + first pip
    assert i1["reward_terms"] == pytest.approx({"NewRoom": 1.0, "Food": 0.3, "Sleep": 0.0, "Death": 0.0, "Malnourished": 0.0})
    assert env.step(0)[1] == pytest.approx(-3.0)
    assert env.step(0)[1] == pytest.approx(-0.0003)

    env = make_default_reward_env(ScriptedEnv(seq), novelty = False)
    assert [t.name for t in env.terms] == ["Food", "Sleep", "Death", "Malnourished"] and env.ticks_per_step == 1
    env = DriveReward(ScriptedEnv(seq, ticks_per_step = 4), ticks_per_step = 8)
    assert env.term("Food").recovery_steps == 3000                 # explicit override wins
    # weights are overridable by keyword
    env = make_default_reward_env(ScriptedEnv(seq), death = -1.0, food_below = 0.5, recovery_steps = 123)
    assert env.term("Death").weight == -1.0 and env.term("Food").below == 0.5 and env.term("Food").recovery_steps == 123


# ---------------------------------------------------------------------------
# Scenario: camper vs forager
# ---------------------------------------------------------------------------

TPS = 4
CYCLE = nominal_cycle_steps(TPS)          # 6000 env steps per nominal cycle


def camper(total_steps: int, camp_steps: int = 100) -> Iterator[Dict[str, Any]]:
    """
    Each pseudo-cycle: eat 4 pips at the hive (room 20), step into the adjacent shelter
    (room 21), stand still, hibernate at once; wake up in the shelter with 0 pips.
    ``camp_steps`` steps per pseudo-cycle, all in the same two rooms.
    """
    cycle = 0
    yield info(room_index = 21, in_shelter = True, cycle_number = cycle)
    produced = 1
    while produced < total_steps:
        steps: List[Dict[str, Any]] = []
        eat_at = [10, 20, 30, 40]
        food = 0
        for s in range(camp_steps):
            if s < 50:
                if s in eat_at:
                    food += 1
                steps.append(info(room_index = 20, food = food, cycle_number = cycle))
            elif s < camp_steps - 2:
                steps.append(info(room_index = 21, in_shelter = True, food = 4, cycle_number = cycle))
            elif s == camp_steps - 2:
                steps.append(info(room_index = 21, in_shelter = True, food = 4, cycle_number = cycle, cycle_survived = True))
            else:
                cycle += 1
                steps.append(info(room_index = 21, in_shelter = True, food = 0, cycle_number = cycle))
        for st in steps:
            if produced >= total_steps:
                return
            yield st
            produced += 1


def forager(total_steps: int, cycle_steps: int = CYCLE) -> Iterator[Dict[str, Any]]:
    """
    Each cycle: roam 7 rooms (30..36) eating one pip in each, spread over the whole
    nominal cycle, then sleep in the shelter (room 37) with a full belly (7 pips).
    """
    cycle = 0
    yield info(room_index = 37, in_shelter = True, cycle_number = cycle)
    produced = 1
    rooms = list(range(30, 37))
    while produced < total_steps:
        steps: List[Dict[str, Any]] = []
        food = 0
        segment = (cycle_steps - 3) // 7
        for k, room in enumerate(rooms):
            for s in range(segment):
                if s == segment // 2:
                    food += 1
                steps.append(info(room_index = room, food = food, cycle_number = cycle))
        while len(steps) < cycle_steps - 2:
            steps.append(info(room_index = 37, in_shelter = True, food = food, cycle_number = cycle))
        steps.append(info(room_index = 37, in_shelter = True, food = food, cycle_number = cycle, cycle_survived = True))
        cycle += 1
        steps.append(info(room_index = 37, in_shelter = True, food = food - 4, cycle_number = cycle))
        for st in steps:
            if produced >= total_steps:
                return
            yield st
            produced += 1


def total_reward(terms: List[RewardTerm], infos: Iterator[Dict[str, Any]]) -> Dict[str, float]:
    first = next(infos)
    for t in terms:
        t.reset(first)
    totals = {t.name: 0.0 for t in terms}
    prev = first
    n = 1
    for cur in infos:
        for t in terms:
            totals[t.name] += t(prev, cur)
        prev = cur
        n += 1
    totals["TOTAL"] = sum(totals.values())
    totals["steps"] = n
    return totals


@pytest.mark.parametrize("novelty", [True, False])
def test_scenario_forager_beats_camper_over_the_same_number_of_steps(novelty: bool):
    steps = 3 * CYCLE                                 # three nominal cycles of env time
    camp = total_reward(drive_terms(novelty, ticks_per_step = TPS), camper(steps))
    forage = total_reward(drive_terms(novelty, ticks_per_step = TPS), forager(steps))
    print(f"\n[scenario novelty={novelty}] {steps} steps ({steps // CYCLE} nominal cycles, ticks_per_step={TPS})")
    print(f"  camper  (eat 4 at the hive, sleep at once, {steps // 100} sleeps): {camp}")
    print(f"  forager (7 pips in 7 rooms over a full cycle, {steps // CYCLE} sleeps): {forage}")
    assert camp["steps"] == forage["steps"] == steps
    assert forage["TOTAL"] > camp["TOTAL"]
    # and the margin is not a rounding artefact
    assert forage["TOTAL"] > 1.5 * camp["TOTAL"]


def test_scenario_camper_sleep_reward_is_tiny_and_food_reward_collapses():
    camp = total_reward(drive_terms(False, ticks_per_step = TPS), camper(3 * CYCLE))
    sleeps = 3 * CYCLE // 100
    assert camp["Sleep"] == pytest.approx(sleeps * (100 / CYCLE) ** 2, rel = 0.05)   # (1.7 %)^2 per sleep
    # first pseudo-cycle at the hive: 0.3 * (1 + .5 + .25 + .125); afterwards the room is nearly exhausted
    assert camp["Food"] < 0.3 * 1.875 + 0.05 * (sleeps - 1)


def glutton_camper(total_steps: int, camp_steps: int = 150) -> Iterator[Dict[str, Any]]:
    """Like ``camper`` but eats all 7 pips (two hive rooms) before each immediate sleep."""
    cycle = 0
    yield info(room_index = 21, in_shelter = True, cycle_number = cycle)
    produced = 1
    while produced < total_steps:
        steps: List[Dict[str, Any]] = []
        food = 0
        for s in range(camp_steps):
            if s < 100:
                room = 20 if s < 50 else 22
                if s % 15 == 10 and food < 7:
                    food += 1
                steps.append(info(room_index = room, food = food, cycle_number = cycle))
            elif s < camp_steps - 2:
                steps.append(info(room_index = 21, in_shelter = True, food = food, cycle_number = cycle))
            elif s == camp_steps - 2:
                steps.append(info(room_index = 21, in_shelter = True, food = food, cycle_number = cycle, cycle_survived = True))
            else:
                cycle += 1
                steps.append(info(room_index = 21, in_shelter = True, food = food - 4, cycle_number = cycle))
        for st in steps:
            if produced >= total_steps:
                return
            yield st
            produced += 1


def test_scenario_glutton_camper_report():
    """A camper eating 7 pips per quick cycle must not farm the full-belly bonus:
    the whole sleep reward (surplus included) scales with tiredness."""
    steps = 3 * CYCLE
    glutton = total_reward(drive_terms(False, ticks_per_step = TPS), glutton_camper(steps))
    forage = total_reward(drive_terms(False, ticks_per_step = TPS), forager(steps))
    print(f"\n[scenario glutton, novelty=False] {steps} steps")
    print(f"  glutton camper (7 pips, sleep at once, {steps // 150} sleeps): {glutton}")
    print(f"  forager: {forage}")
    assert forage["TOTAL"] > 1.5 * glutton["TOTAL"]
