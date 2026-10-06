# Rewards

The environment returns `reward = 0.0`. Observation is the frame; **everything
else lives in `info`**, and rewards are Python wrappers over `info`
(`rainworld_rl/rewards.py`). This page explains the default (`DriveReward`),
the camping problem it is built around, the two mechanisms that solve it, the
weights, how to compose alternatives, and what the `info` fields mean when you
design your own term.

## Philosophy

The environment exists to test exploration / intrinsic-motivation algorithms.
A reward that tells the agent *how* to play (distance-to-goal, "move right")
would measure the shaping, not the algorithm. So the default is not a task; it
mimics the **drives of an animal** - hunger, tiredness, satisfaction, pain -
and leaves everything about *how* to satisfy them to the agent. The game
itself supplies the structure: food depletes locally, shelters have to be
found before the rain, predators are in the way.

## The camping problem

The obvious sparse reward ("+1 per survived cycle") has a degenerate optimum
that involves no exploration at all. Three facts from the game code
(Rain World v1.11.8 decompile):

1. **Early sleep is legal.** A slugcat with at least `food_to_hibernate` pips
   standing still in a shelter hibernates after ~20 ticks of no input (40 with Remix), no
   matter how early in the cycle it is: `Player.cs` ~5734-5776 sets
   `readyForWin` when `FoodInRoom(...) >= foodToHibernate` and calls
   `room.shelterDoor.Close()` once `touchedNoInputCounter > 40`. (The *only*
   other way the door closes early is the starving sleep, `forceSleepCounter`,
   which does not count.)
2. **Batflies respawn every cycle.** `FliesWorldAI.cs` :60-80 refills
   `fliesToSpawn` for every swarm room each time a `World` is created (every
   cycle); in SU the tutorial hive even gets `foodToHibernate + 1` flies.
   Placed consumables (`AbstractConsumable.cs` :122-123,
   `PlacedObject.ConsumableObjectData`, `PlacedObject.cs` :553-580) regrow
   after `minRegen..maxRegen` **cycles** (2-3 by default), and `RegionState`
   ticks them forward by the *cycle counter* (`RegionState.cs` :1109), so a
   player who sleeps often sees them regrow in minutes of game time.
3. **The stomach is capped.** `Player.AddFood` clamps to `MaxFoodInStomach`
   (`Player.cs` :4398, 2078-2086 = `SlugcatStats.maxFood`, 7 for Survivor
   with 4 to hibernate, `SlugcatStats.cs` :117-119). Food reward per cycle is
   therefore bounded no matter how far the agent ranges.

So "camp next to a batfly hive, eat 4, step into the adjacent shelter, sleep
immediately" survives a cycle in about a minute of game time, forever, and
anything that pays per survived cycle - or per pip, or per in-shelter step -
rewards it maximally.

## The default: `DriveReward`

```python
from rainworld_rl import RainWorldEnv
from rainworld_rl.rewards import make_default_reward_env

env = make_default_reward_env(RainWorldEnv(ticks_per_step = 4))   # DriveReward, novelty on
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
info["reward_terms"]   # {"NewRoom": 0.0, "Food": 0.3, "Sleep": 0.0, "Death": 0.0, "Malnourished": 0.0}
```

`DriveReward(env, novelty = True, **weights)` = `InfoReward(env, [NewRoom,
Food, Sleep, Death, Malnourished])` (`NewRoom` only when `novelty`). Two of
the terms carry the anti-camping design:

### Mechanism 1: per-room satiety, recovering in real time (`Food`)

Every room has a **satiety factor** starting at 1.0. Each pip eaten in a room
pays `below` (0.3) if the pip index - the food count after that pip - is
`<= food_to_hibernate`, else `above` (0.1), **times the room's factor**, and
then multiplies the factor by `satiety_decay` (0.5). The factor recovers
**linearly** toward 1.0 at `1 / recovery_steps` per **real env step** (every
step, ready or not), never per cycle. `recovery_steps` defaults to one nominal
cycle: `NOMINAL_CYCLE_TICKS = 24000` ticks (10 min; the game draws 400-800 s =
16000-32000 ticks per cycle, `RainCycle.cs` :245) divided by the env's
`ticks_per_step` (`nominal_cycle_steps(ticks_per_step)`, 6000 at 4 ticks).

Because recovery runs on the env clock and not on the cycle counter, sleeping
early does **not** refresh a patch. The camper's hive is worth
`0.3 * (1 + .5 + .25 + .125) = 0.56` on the first visit and then collapses:
with 100 steps between sleeps the factor only recovers by `100 / 6000` per
pseudo-cycle and settles near 0.018, so the 4 pips pay about 0.01 per sleep.
A forager eating one pip in each of 7 rooms gets `4 * 0.3 + 3 * 0.1 = 1.5` per
cycle, and those rooms are fresh again when it comes back a cycle later.

Food decreases (hibernation subtracts `food_to_hibernate`, death resets, the
mod reads 0 while not READY) never give negative reward, and deltas across
`cycle_number` changes or non-ready steps are ignored.

### Mechanism 2: tiredness-scaled sleep (`Sleep`)

On the `cycle_survived` edge the reward is

```
tiredness * (weight + full_belly_per_pip * max(0, food_before_sleep - food_to_hibernate))
tiredness = min(1, steps_awake / nominal_cycle_steps) ** tiredness_power    # default power 2
tiredness = min(1, steps_awake / nominal_cycle_steps)
```

`steps_awake` counts real env steps since the last sleep (or since `reset()`),
so a sleep after 10 % of a nominal cycle is worth 1 % of `weight` (the ramp is
squared, see below), and a full
sleep (1.0) is only collectable by staying out a whole cycle - at which point
the food you need is not where you slept. `food_before_sleep` is
`prev_info["food"]` (the save has already subtracted the hibernation cost by
the time the edge is reported); the surplus bonus is bounded by `food_max -
food_to_hibernate` (3 pips = 0.75 for Survivor). A starving sleep
(`cycle_number` increments without `cycle_survived`, or `malnourished` turns
True) earns nothing but still resets `steps_awake`.

**Why the surplus bonus is inside the tiredness factor:** if it were added
unscaled, a "glutton camper" that eats *7* pips per quick pseudo-cycle (hive +
fruit that regrows on the cycle counter) would collect 0.75 per sleep and beat
the forager by a wide margin (synthetic run: 67 vs 10). Scaling the whole sleep
reward by tiredness closes that: a sleep after 2.5% of a cycle pays 2.5% of
everything. `test_scenario_glutton_camper_report` asserts the forager wins.

**Why tiredness is squared:** with a linear ramp the lifetime sleep reward is
rate-neutral - it accrues per step awake, so 120 quick sleeps earn as much as
3 full ones and the forager only won through food (9.7 vs 8.5). A convex ramp
(`tiredness_power = 2`) makes a sleep after 2.5% of a cycle worth 0.06% of a
full one, which is what sleep pressure feels like and what makes quick cycles
strictly lose.

### The other terms

* `NewRoom(weight = 1.0)` - first entry into each `(region, room_index)`. Room
  indices are only unique within a region, so the mod's `region` field is part
  of the key. The visited set persists across deaths (the environment is
  continual) and clears only on `reset()`; the reset room counts as visited.
  **Novelty is a toggle** (`novelty = False`) because an intrinsic-motivation
  algorithm brings its own curiosity signal; with the toggle off the
  environment's reward is pure drives and the exploration pressure comes only
  from satiety.
* `Death(weight = -3.0)` - on the `player_dead` edge (one step).
* `Malnourished(per_step = -0.0003)` - every ready step while `malnourished`
  is True (the previous sleep was a starving one). Per env step, so at 1 tick
  per step a malnourished cycle costs about -7, at 4 ticks about -1.8.

### Weights

| Term | Parameter | Default | Fires |
|------|-----------|--------:|-------|
| `NewRoom` | `weight` | `1.0` | first entry into a `(region, room_index)` |
| `Food` | `below` | `0.3` | per pip with index `<= food_to_hibernate`, x satiety |
| `Food` | `above` | `0.1` | per pip with index `> food_to_hibernate`, x satiety |
| `Food` | `satiety_decay` | `0.5` | factor multiplier per pip eaten in the room |
| `Food` | `recovery_steps` | `nominal_cycle_steps(ticks_per_step)` | steps for a factor to recover from 0 to 1 |
| `Sleep` | `weight` | `1.0` | x tiredness on `cycle_survived` |
| `Sleep` | `full_belly_per_pip` | `0.25` | per pip above `food_to_hibernate` at the sleep |
| `Sleep` | `nominal_cycle_steps` | `nominal_cycle_steps(ticks_per_step)` | steps awake for tiredness 1.0 |
| `Death` | `weight` | `-3.0` | `player_dead` edge |
| `Malnourished` | `per_step` | `-0.0003` | every ready step while malnourished |

All are keyword arguments of `drive_terms` / `DriveReward` /
`make_default_reward_env` (`new_room`, `food_below`, `food_above`,
`satiety_decay`, `recovery_steps`, `sleep`, `full_belly_per_pip`, `death`,
`malnourished`; `ticks_per_step` overrides what is read from the env).

### Camper vs forager (unit-test scenario)

`rainworld_rl/tests_unit/test_rewards.py::test_scenario_forager_beats_camper_over_the_same_number_of_steps`
replays synthetic infos for 18000 env steps at 4 ticks/step (3 nominal
cycles): a camper that eats 4 pips at one hive and sleeps at once every 100
steps (180 sleeps) against a forager that eats 7 pips in 7 rooms over a full
cycle and then sleeps (3 sleeps).

| | NewRoom | Food | Sleep | Total |
|--|--:|--:|--:|--:|
| camper, novelty on | 1.0 | 2.39 | 2.97 | **6.36** |
| forager, novelty on | 7.0 | 4.50 | 5.25 | **16.75** |
| camper, novelty off | - | 2.39 | 2.97 | **5.36** |
| forager, novelty off | - | 4.50 | 5.25 | **9.75** |

## Composing alternatives

```python
from rainworld_rl.rewards import (InfoReward, DriveReward, SurviveCycleReward,
                                  NewRoom, Food, Sleep, Death, Malnourished,
                                  CycleSurvived, Ate, Alive, FunctionTerm, nominal_cycle_steps)

env = DriveReward(RainWorldEnv(ticks_per_step = 4), novelty = False, death = -1.0)   # tweak weights
env = SurviveCycleReward(RainWorldEnv())                                             # sparse baseline
env = InfoReward(RainWorldEnv(ticks_per_step = 4), [                                 # hand-rolled
    Food(recovery_steps = nominal_cycle_steps(4)),
    Sleep(nominal_cycle_steps = nominal_cycle_steps(4), full_belly_per_pip = 0.0),
    FunctionTerm(lambda prev, cur: cur["karma"] - prev["karma"], name = "karma"),
])
```

`InfoReward(env, terms, breakdown_key = "reward_terms", add_env_reward = False)`
is a `gym.Wrapper`. On every step it calls each term as
`term(prev_info, info) -> float`, sums the results into the reward and writes
the per-term breakdown to `info["reward_terms"]`. `reset()` calls
`term.reset(info)` on each term with the first `info`. Each term applies its
own signed `weight`; `name` (default: class name) must be unique within one
wrapper; `env.term("Food")` looks one up.

Building blocks that are **not** in the default (each has the camping optimum
on its own):

| Term | Default weight | Fires |
|------|---------------:|-------|
| `CycleSurvived` | `+1.0` | on the step `cycle_survived` is True (`SurviveCycleReward` is exactly this) |
| `Ate` | `+1.0` per pip | `food` increased since the previous step, same cycle, both steps `ready`; no satiety |
| `Alive` | `+0.01` | every step with real game state and no death |
| `FunctionTerm(fn, name, weight)` | `1.0` | `weight * fn(prev_info, info)` for any callable |

Write your own by subclassing `RewardTerm` (implement `__call__`, optionally
`reset`). Terms see two consecutive `info` dicts, so edges and deltas are easy;
anything needing longer memory keeps its own state, clears it in `reset`, and
must tolerate non-ready steps (fields zero / -1).

## `info` fields relevant to reward design

All come straight from the mod's header written just before the frame
(`docs/PROTOCOL.md`). While the mod is not `ready` (menus, death screen,
loading) the numeric fields read `0` / `-1`, strings are empty and the flags
are `False`; guard deltas with `info["ready"]` as the built-in terms do.

| Field | Meaning for rewards |
|-------|---------------------|
| `cycle_survived` | **Edge.** Slept with enough food. `Sleep` pays on it. |
| `player_dead` | **Edge.** The player died this step. Not an episode end; the game respawns at the last shelter. |
| `food`, `food_max`, `food_to_hibernate` | Pips in the stomach, the slugcat's maximum (Survivor 7) and how many are needed to sleep (Survivor 4). `food_to_hibernate == food_max` while `malnourished`. Food is reset by hibernation (`food - food_to_hibernate` carries over) and by death. |
| `malnourished` | **Level.** The previous sleep was a starving one: this cycle the slugcat is weaker and needs `food_max` pips to sleep. Starving twice in a row is lethal. |
| `cycle_progress` | `RainCycle.timer / cycleLength`: 0 at cycle start, 1.0 when the rain arrives, keeps growing while it falls. Monotone within a cycle; resets with each new cycle / death. **First cycle of a fresh save:** the yellow overseer's tutorial pins the timer at 2000 ticks (`cycle_progress` about 0.06-0.12) until the slugcat gets past the start rooms (x > 600 in `SU_A43`, or `SU_A22`). |
| `rain` | **Level.** `cycle_progress >= 1`: the lethal rain is falling. Being outside a shelter now is death within seconds. |
| `in_shelter` | **Level.** The player is in a shelter room (`AbstractRoom.shelter`). |
| `karma`, `karma_cap` | Karma level (0-9) and cap. Up with every survived cycle, down with every death; gates require a minimum karma. A natural secondary long-horizon signal. |
| `region`, `room_index`, `player_pos` | Region acronym (`"SU"`, `"HI"`, ...; `""` while unavailable), abstract room id (unique only within a region) and body position: raw material for novelty / coverage bonuses (`NewRoom`, `Food` satiety). Room ids are stable within a save. |
| `cycle_number` | Save-state cycle counter; increments on every sleep (fed or starving), not on death. |
| `dialog_open` | **Level.** An in-game text/dialog overlay awaits input; game time is effectively paused. |

Things to keep in mind:

- Deltas across `ready == False` gaps and across `cycle_number` changes are not
  meaningful (fields reset); `Food` and `Ate` show the guard pattern.
- `cycle_survived` can land on a step where `ready` is already `False` (the
  sleep-screen redirect may have left the game in the same step); the flag is
  still delivered, so do not gate it on `ready`. `Sleep` reads the food it
  needs from the last ready step.
- Rewarding `in_shelter` or `rain` directly re-introduces shaping (and a
  camping optimum); the default deliberately leaves them to the agent.
