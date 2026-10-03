# Rewards

The environment returns `reward = 0.0`. Observation is the frame; **everything
else lives in `info`**, and rewards are Python wrappers over `info`
(`rainworld_rl/rewards.py`). This page explains the default, why it is what it
is, the building blocks for alternatives, and what the `info` fields mean when
you design your own term.

## Philosophy

The environment exists to test exploration / intrinsic-motivation algorithms.
A reward that tells the agent *how* to play (step bonuses, distance-to-goal,
"move right") would measure the shaping, not the algorithm. So the default is a
single sparse event that the game itself already gates behind everything we
care about, and nothing else is shaped:

- **Food depletes locally.** Edible things respawn slowly and a spot that fed
  you once is empty next cycle, so the same sitting spot stops working and the
  agent has to range further every cycle.
- **Shelters have to be found** and reached before the rain.
- **The rain is on a clock** (`cycle_progress`); staying out past `rain == True`
  is death.
- **Predators** are between you and all of the above.

Surviving one cycle requires all four. We do not reward any of them separately.
If current algorithms cannot reach the signal, that is a result, not a bug.

## The default: `SurviveCycleReward`

```python
from rainworld_rl import RainWorldEnv
from rainworld_rl.rewards import make_default_reward_env

env = make_default_reward_env(RainWorldEnv(ticks_per_step = 4))
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)   # reward in {0.0, 1.0}
```

`SurviveCycleReward` = `InfoReward(env, [CycleSurvived(1.0)])`: `+1` on the
step where `info["cycle_survived"]` is `True`, `0` on every other step. The
mod sets that one-step edge when `RainWorldGame.Win` runs with
`malnourished == false`, i.e. the slugcat slept in a shelter **with at least
`food_to_hibernate` pips**. A starving sleep (not enough food: the game's
"starve" screen) does **not** set it; it is visible instead as `cycle_number`
going up with `malnourished == True`, and camping in the start shelter
therefore earns nothing. Death is not penalised (the environment is continual
and respawns; the agent simply does not get the +1 it was working towards).

## Composing alternatives

```python
from rainworld_rl.rewards import InfoReward, CycleSurvived, Death, NewRoom, Ate, Alive, FunctionTerm

env = InfoReward(RainWorldEnv(), [CycleSurvived(1.0), Death(-1.0), NewRoom(0.1)])
```

`InfoReward(env, terms, breakdown_key = "reward_terms", add_env_reward = False)`
is a `gym.Wrapper`. On every step it calls each term as
`term(prev_info, info) -> float`, sums the results into the reward and writes
the per-term breakdown to `info["reward_terms"]`. `reset()` calls
`term.reset(info)` on each term with the first `info`. Each term applies its
own signed `weight`; `name` (default: class name) must be unique within one
wrapper.

| Term | Default weight | Fires |
|------|---------------:|-------|
| `CycleSurvived` | `+1.0` | on the step `cycle_survived` is True |
| `Death` | `-1.0` | on the step `player_dead` is True (one-step edge) |
| `NewRoom` | `+1.0` | the first time each `room_index` is entered; the reset room counts as visited; the set clears on `reset()` |
| `Ate` | `+1.0` per pip | `food` increased since the previous step, same `cycle_number`, both steps `ready` |
| `Alive` | `+0.01` | every step with real game state and no death |
| `FunctionTerm(fn, name, weight)` | `1.0` | `weight * fn(prev_info, info)` for any callable |

Write your own by subclassing `RewardTerm` (implement `__call__`, optionally
`reset`). Terms see two consecutive `info` dicts, so edges and deltas are easy;
anything needing longer memory keeps its own state and clears it in `reset`.

## `info` fields relevant to reward design

All come straight from the mod's header written just before the frame
(`docs/PROTOCOL.md`). While the mod is not `ready` (menus, death screen,
loading) the numeric fields read `0` / `-1` and the flags are `False`; guard
deltas with `info["ready"]` as the built-in terms do.

| Field | Meaning for rewards |
|-------|---------------------|
| `cycle_survived` | **Edge.** Slept with enough food. The one event the default rewards. |
| `player_dead` | **Edge.** The player died this step. Not an episode end; the game respawns at the last shelter. |
| `food`, `food_max`, `food_to_hibernate` | Pips in the stomach, the slugcat's maximum (Survivor 7) and how many are needed to sleep (Survivor 4). `food_to_hibernate == food_max` while `malnourished`. Food is reset by hibernation (`food - food_to_hibernate` carries over) and by death. |
| `malnourished` | **Level.** The previous sleep was a starving one: this cycle the slugcat is weaker and needs `food_max` pips to sleep. Starving twice in a row is lethal. |
| `cycle_progress` | `RainCycle.timer / cycleLength`: 0 at cycle start, 1.0 when the rain arrives, keeps growing while it falls. Monotone within a cycle; resets with each new cycle / death. The cycle length is drawn per cycle by the game (400-800 s of game time, 16000-32000 ticks). **First cycle of a fresh save:** the yellow overseer's tutorial pins the timer at 2000 ticks (`cycle_progress` sits at about 0.06-0.12, verified live) until the slugcat gets past the start rooms (x > 600 in `SU_A43`, or `SU_A22`), after which the rain is brought in quickly. So on cycle 0 the clock does not run until the agent explores. |
| `rain` | **Level.** `cycle_progress >= 1`: the lethal rain is falling. Being outside a shelter now is death within seconds. |
| `in_shelter` | **Level.** The player is in a shelter room (`AbstractRoom.shelter`). The shelter door only closes (and the cycle only ends) once the rain is near; being in a shelter early is safe but not yet a win. |
| `karma`, `karma_cap` | Karma level (0-9) and cap. Karma goes up with every survived cycle and down with every death; gates (`karma_cap`) require a minimum karma to pass. A natural secondary long-horizon signal. |
| `room_index`, `player_pos` | Abstract room id and body position: raw material for novelty / coverage bonuses (`NewRoom`). Room ids are stable within a save. |
| `cycle_number` | Save-state cycle counter; increments on every sleep (fed or starving), not on death. |
| `dialog_open` | **Level.** An in-game text/dialog overlay awaits input; game time is effectively paused. |

Things to keep in mind:

- Deltas across `ready == False` gaps and across `cycle_number` changes are not
  meaningful (fields reset); `Ate` shows the guard pattern.
- `cycle_survived` can land on a step where `ready` is already `False` (the
  sleep-screen redirect may have left the game in the same step); the flag is
  still delivered, so do not gate it on `ready`.
- Rewarding `in_shelter` or `rain` directly re-introduces shaping; the default
  deliberately leaves them to the agent.
