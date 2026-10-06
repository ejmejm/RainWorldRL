# Python API

Python client for the Rain World RL mod. Protocol details live in
[PROTOCOL.md](PROTOCOL.md); this page covers day-to-day usage.

Requirements: Windows or Linux (see the README's Quick start), Python >= 3.11,
`numpy`, `gymnasium` (`pip install -e .`). The package ships a prebuilt mod DLL;
the .NET SDK is only needed to change the mod.

## Importing

The repository directory *is* the package (`rainworld_rl`) and the code lives
in its `python` subpackage, so the **parent** of the repo must be on
`sys.path` (e.g. `E:/projects` for `E:/projects/rainworld_rl`):

```python
import sys; sys.path.insert(0, "E:/projects")      # or set PYTHONPATH
from rainworld_rl import RainWorldEnv, GameNotRunningError
```

`gym.make("RainWorld-v0")` works once `rainworld_rl.rainworld_env` has
been imported (registration happens at import time with entry point
`rainworld_rl.rainworld_env:RainWorldEnv`).

## Quick start

```python
from rainworld_rl import RainWorldEnv

env = RainWorldEnv(frame_width = 160, frame_height = 90, ticks_per_step = 4)  # cheap, no game contact

env.launch()                 # build mod, (re)start RainWorld.exe, wait for the mod, connect
# ...or, if the game is already running with the mod:
# env.connect()

obs, info = env.reset()      # wipe the RL save, start a fresh game, return the first frame
for _ in range(10_000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if info["player_dead"]:
        print("died at step", info["step_counter"])   # not an episode end; keep stepping
env.close()                  # detach; the game keeps running
```

## Lifecycle

| Call | What it does |
|------|--------------|
| `RainWorldEnv(...)` | Builds spaces and a client object. Never launches or touches the game. |
| `env.launch(build=None, restart=True, wait_ready=True)` | **Heavy.** Build the mod if possible (source checkout with `dotnet`), else deploy the prebuilt DLL; kill a running game; start it (Windows: `RainWorld.exe` directly; Linux: under Wine, see `launcher`); wait for the mod heartbeat, then `connect()`. Raises `LaunchError` with the tail of the game log on failure. |
| `env.connect(wait_ready=True)` | Attach to a running game: liveness check (heartbeat), set `CONNECTED`, optionally wait for `READY`. Raises `GameNotRunningError` if nothing is running. |
| `env.reset(options={"wipe": True})` | Connects if needed, sends `RESET` (wipe RL save, fresh story game, wait for READY), then does one no-op step and returns `(frame, info)`. `options={"wipe": False}` skips the command and just returns the current frame of the game in progress (requires READY). |
| `env.step(action)` | One step of `ticks_per_step` physics ticks. Returns `(frame, 0.0, False, False, info)`. |
| `env.disconnect()` / `env.close()` | Clear `CONNECTED`; the mod hands the game back to normal play. Does not quit the game. |
| `env.debug_kill(timeout=10)` | **Debug/testing only.** Sends `KILL_PLAYER`: the mod kills the slugcat immediately and acks; the respawn is observed through later `step()` calls (see below). Not part of the RL interface. |

Constructor keyword knobs: `ready_timeout` (60 s), `frame_timeout` (10 s),
`reset_timeout` (90 s), `render_mode="rgb_array"`, `debug_timing`, `config`,
`instance` (Linux: which of several games running side by side to drive).

### Reset semantics

This is a continual environment. **Death is not a reset**: `terminated` and
`truncated` are always `False`, and the game continues (the mod handles the
death screen / cycle restart). Call `reset()` only when you really want a
fresh game; it wipes the mod's RL save slot.

`info["player_dead"]` is a one-step edge straight from the mod's status bit:
`True` exactly for the step during which the player died, `False` otherwise.

What a death looks like from Python (natural or via `debug_kill()`): the step
in which the slugcat dies has `player_dead == True`; the game then shows its
"game over" prompt for ~40 ticks, after which the mod presses the restart key
for the agent, skips the death screen and reloads the cycle. During the reload
`ready` is `False` (and the game-state fields are zero / -1) for a number of
steps - steps keep being serviced with the current frame, they never time out -
and then `ready` comes back with the slugcat in the start-of-cycle shelter.
Karma follows the game's own rules (-1 unless reinforced, floor 0); the cycle
number does not change on death (only surviving a cycle advances it).

### Debugging aid: `debug_kill()`

```python
env.debug_kill()                       # mod kills the slugcat, acks immediately
_, _, _, _, info = env.step(0)         # info["player_dead"] is True on this step
while not env.step(0)[4]["ready"]:     # a few dozen steps of death-screen skip + reload
    pass
```

`debug_kill()` exists so tests and tooling can exercise the death -> respawn
flow deterministically. It raises `CommandError` if the mod rejects it (not in
a game, no live player, already dead) and `GameNotRunningError` if the env is
not connected.

### Observation and `info`

Observation: `uint8` array of shape `(frame_height, frame_width, 3)`, RGB,
top row first. It is a fresh copy each step.

`info` fields (all from the shared header written by the mod just before the
frame flag):

| Key | Type | Meaning |
|-----|------|---------|
| `player_dead` | bool | Edge: player died during this step. |
| `karma`, `karma_cap` | int | Current karma level and cap. |
| `food` | int | Food pips in the stomach. |
| `food_max` | int | The slugcat's maximum food pips (Survivor: 7). |
| `food_to_hibernate` | int | Pips needed to sleep this cycle (Survivor: 4; equals `food_max` while `malnourished`). |
| `malnourished` | bool | Level: the previous sleep was a starving one; this cycle the slugcat is weaker and needs `food_max` pips to sleep. |
| `player_pos` | (float, float) | Body chunk 0 position in room coordinates. |
| `room_index` | int | Abstract room index, -1 if unavailable. |
| `cycle_number` | int | Save-state cycle number, -1 if unavailable. Increments on every sleep (fed or starving). |
| `cycle_progress` | float | `RainCycle.timer / cycleLength`: 0 at cycle start, 1.0 when the rain arrives, keeps growing while it falls. |
| `rain` | bool | Level: the cycle timer has expired and the lethal rain is falling (`cycle_progress >= 1`). |
| `in_shelter` | bool | Level: the player is in a shelter room. |
| `cycle_survived` | bool | Edge: the player hibernated with enough food during this step (`+1` cycle, karma up). A starving sleep does not set it. |
| `dialog_open` | bool | Level: an in-game text/dialog overlay awaits input. |
| `region` | str | Region acronym of the active world (`"SU"`, `"HI"`, ...); `""` while unavailable. Room indices are unique only within a region, so key rooms by `(region, room_index)`. |
| `step_counter` | int | Mod-side count of completed steps. |
| `in_game` | bool | Main process is `RainWorldGame`. |
| `ready` | bool | Mod is in RL mode with a live player; steps are fully serviced. |
| `human_override` | bool | F10 override active (see below). |

When `ready` is `False` (menus, loading, death screen) the mod still answers
steps with the current rendered frame so the agent never hangs; game-state
fields are zero / -1 (strings empty) then.

### Rewards

`env.step()` always returns `reward = 0.0`; rewards are wrappers over `info`
(`rainworld_rl.rewards`, see [REWARDS.md](REWARDS.md) for the design and the
field semantics). The default, `DriveReward`, mimics animal drives - hunger
with per-room satiety, tiredness-scaled sleep, death, malnourishment and an
optional novelty bonus - and is built so that camping next to a hive and
sleeping early is strictly suboptimal:

```python
from rainworld_rl import RainWorldEnv
from rainworld_rl.rewards import make_default_reward_env, DriveReward

env = make_default_reward_env(RainWorldEnv(ticks_per_step = 4))      # DriveReward, novelty on
env = DriveReward(RainWorldEnv(ticks_per_step = 4), novelty = False) # the agent brings its own curiosity
obs, reward, terminated, truncated, info = env.step(action)
info["reward_terms"]   # {"NewRoom": ..., "Food": ..., "Sleep": ..., "Death": ..., "Malnourished": ...}
```

`DriveReward(env, novelty = True, **weights)` = `InfoReward(env, drive_terms(...))`.
The time-based terms (`Food` recovery, `Sleep` tiredness) measure a nominal
cycle of `NOMINAL_CYCLE_TICKS = 24000` ticks in env steps, derived from the
env's `ticks_per_step` (`nominal_cycle_steps(tps)`; pass `ticks_per_step = ...`
to override).

| Term | Defaults | Fires |
|------|----------|-------|
| `NewRoom` | `weight = 1.0` | first entry into each `(region, room_index)`; persists across deaths, clears on `reset()` |
| `Food` | `below = 0.3, above = 0.1, satiety_decay = 0.5, recovery_steps = nominal cycle` | per pip gained, x the room's satiety factor (x0.5 per pip eaten there, recovers linearly over `recovery_steps` real steps) |
| `Sleep` | `weight = 1.0, full_belly_per_pip = 0.25` | on `cycle_survived`: `weight * min(1, steps_awake / nominal_cycle_steps) + 0.25 * max(0, food - food_to_hibernate)`; a starving sleep earns nothing |
| `Death` | `weight = -3.0` | the step `player_dead` is True |
| `Malnourished` | `per_step = -0.0003` | every ready step while `malnourished` |

Weights are keyword arguments: `DriveReward(env, death = -1.0, food_below = 0.5,
full_belly_per_pip = 0.0, ...)` (`new_room`, `food_below`, `food_above`,
`satiety_decay`, `recovery_steps`, `sleep`, `full_belly_per_pip`, `death`,
`malnourished`).

Alternatives are composed from `RewardTerm`s, each called as
`term(prev_info, info) -> float` once per step and `term.reset(info)` on
`reset()`; the per-term breakdown is written to `info["reward_terms"]`:

```python
from rainworld_rl.rewards import InfoReward, SurviveCycleReward, CycleSurvived, Death, NewRoom, Ate, Alive, FunctionTerm

env = SurviveCycleReward(RainWorldEnv())                                   # sparse baseline: +1 per survived cycle
env = InfoReward(RainWorldEnv(), [CycleSurvived(1.0), Death(-1.0), NewRoom(0.1)])
env = InfoReward(RainWorldEnv(), [FunctionTerm(lambda prev, cur: cur["karma"] - prev["karma"], name = "karma")])
```

Building blocks (not in the default): `CycleSurvived(+1.0)` on the
`cycle_survived` step, `Ate(+1.0/pip)` flat per pip gained within a cycle,
`Alive(+0.01)` per ready non-death step, `FunctionTerm(fn, name, weight)`.
Subclass `RewardTerm` (implement `__call__`, optionally `reset`) for anything
else; every term must handle non-ready steps.

### Actions

The native action space is **raw key presses**: `MultiBinary(9)`, one entry
per key the player can hold, any combination at once. The bit/index order is
`rainworld_rl.KEY_NAMES`:

| index | key | what it does |
|------:|-----|--------------|
| 0 | `left` | move left (left + right together cancel out) |
| 1 | `right` | move right |
| 2 | `up` | up / climb / look up |
| 3 | `down` | down / crouch; with a side key: crawl / roll direction |
| 4 | `jump` | jump; also "continue" in dialogs and prompts |
| 5 | `grab` | pick up / eat / interact |
| 6 | `throw` | throw; also "cancel" in dialogs |
| 7 | `map` | hold to show the map; restarts from the game-over prompt |
| 8 | `special` | slugcat ability (Watcher warp/camo, Saint ascension, Artificer pyro-jump). No-op for Survivor/Monk/Hunter |

`env.step()` accepts either a length-9 0/1 vector (e.g. `env.action_space.sample()`)
**or** a plain `int` bitmask built from `KEY_*` constants:

```python
from rainworld_rl import RainWorldEnv, KEY_RIGHT, KEY_JUMP, encode_keys, pressed_key_names

env.step([0, 1, 0, 0, 1, 0, 0, 0, 0])      # right + jump as a vector
env.step(KEY_RIGHT | KEY_JUMP)             # same, as a bitmask
env.step(encode_keys("right", "jump"))     # same, by name
pressed_key_names(KEY_RIGHT | KEY_JUMP)    # ('right', 'jump')
```

Pause/Escape and menu-exit are intentionally not in the action space; the mod
blocks the pause button while the agent is in control and dismisses a pause
menu it finds open. In-game prompts that wait for a key (a dialog box, the
game-over "press SPACE to restart") receive the agent's keys; `info["dialog_open"]`
is `True` while one is waiting.

#### Optional: `DiscreteActions` wrapper (Discrete(18))

The classic 18-action set is available as an **optional** `gymnasium.ActionWrapper`
in `rainworld_rl.wrappers`; the mod itself only ever sees key bits:

```python
from rainworld_rl import RainWorldEnv, DiscreteActions, ACTION_NAMES

env = DiscreteActions(RainWorldEnv(160, 90, ticks_per_step = 4))
env.action_space        # Discrete(18)
env.step(9)             # ACTION_NAMES[9] == "Right+Jump"
```

```
0 No-op   1 Left   2 Right   3 Up   4 Down   5 Jump   6 Grab   7 Throw
8 Left+Jump   9 Right+Jump   10 Up+Jump   11 Down+Jump
12 Left+Grab  13 Right+Grab  14 Up+Grab   15 Down+Grab
16 Crawl Left (Left+Down)   17 Crawl Right (Right+Down)
```

`wrappers.DISCRETE_ACTIONS[i]` lists the keys held by action `i`;
`wrappers.discrete_to_bits(i)` / `discrete_to_keys(i)` convert. Note the discrete
set has no `map`/`special`, so a discrete agent cannot dismiss the game-over
prompt or use slugcat abilities.

### F10 human override

Press F10 in the game to take control: the mod runs at real time and ignores
agent actions. While the override is set, `env.step()` **blocks** (polling at
~50 ms) instead of timing out, and logs one warning through the `logging`
module (`rainworld_rl.shared_memory`). Press F10 again to hand control
back; the pending action is then serviced and `step()` returns normally.
`info["human_override"]` reports the state.

### Errors

All protocol errors derive from `SharedMemoryError`:

- `GameNotRunningError` - no live mod (on connect), or the heartbeat stopped
  (game exited) during a wait. The message from `connect()`/`reset()` points at
  `launch()`.
- `ReadyTimeoutError` - `READY` never rose within `ready_timeout`.
- `StepTimeoutError` - mod alive but no frame within `frame_timeout`.
- `CommandError` - `RESET` / `KILL_PLAYER` was rejected or not acknowledged.
- `NotConnectedError` - a client operation was used before `connect()`.
- `ProtocolVersionError` - on connect: the deployed mod DLL is from another version of the
  library (stale DLL). `launch()` redeploys the packaged mod; `rainworld-rl setup` installs it.

`LaunchError` (from `launcher`) covers build/start failures.

## Configuration file

Copy `rainworld_rl.example.toml` to `rainworld_rl.toml` at the repo root (it
is git-ignored) and edit:

```toml
game_dir = "Z:/SteamLibrary/steamapps/common/Rain World"
launch_timeout = 120.0
# rl_save_dir = "..."   # informational; the mod decides where the save lives
```

Lookup order: explicit `load_config(path)` argument > `$RAINWORLD_RL_CONFIG` >
`./rainworld_rl.toml` > `~/.config/rainworld_rl/rainworld_rl.toml` (written by
`rainworld-rl setup`) > defaults. Linux-only keys: `container` (the Apptainer
image), `wine_prefix_dir` and `renderer` (`auto`, `cpu`, `wsl`, `virtualgl`; see
`rainworld_rl.example.toml`). Derived from `game_dir`: `exe_path`,
`plugins_dir`, `plugin_dll_path`, `bepinex_log`. `build.ps1` reads `game_dir`
from the same file.

```python
from rainworld_rl import load_config
cfg = load_config()
env = RainWorldEnv(config = cfg)
```

## Launcher CLI

```
python -m rainworld_rl.launcher              # build, deploy, restart game, wait for mod
python -m rainworld_rl.launcher --no-build   # just (re)start
python -m rainworld_rl.launcher --no-restart # attach if running, else start
python -m rainworld_rl.launcher --build-only # dotnet build + copy DLL
python -m rainworld_rl.launcher --kill       # stop a running game
```

The Steam client does not need to be running. On Windows the launcher starts
`RainWorld.exe` directly and copies the DLL only after the old game process
exits, because BepInEx keeps plugin assemblies locked. On Linux each game runs
under Wine on its own Xvfb display; `--instance N` runs several side by side.

## Smoke test

```
python -m rainworld_rl.test_env            # connect to a running game, RESET, 1000 random key combos
python -m rainworld_rl.test_env --launch   # build + restart the game first
python -m rainworld_rl.test_env --no-wipe  # attach without resetting
python -m rainworld_rl.test_env --discrete # sample from the Discrete(18) wrapper instead
```

Prints the `info` fields every 100 steps, a line whenever `player_dead`
fires, and at the end how often each key was held (plus the most common
combinations, and the discrete action distribution with `--discrete`).

## Low-level client

`SharedMemoryClient(frame_width, frame_height, debug_timing=False,
mapping_factory=None)` wraps the mapping directly:

- `connect(wait_ready=True, ready_timeout=60, liveness_timeout=2, alive_grace=None)` / `disconnect()` - while `MOD_ALIVE` is set a stalled heartbeat is tolerated for `alive_grace` seconds (default `alive_stall_grace` = 15 s: the game's initial load runs synchronously right after launch)
- `wait_for_alive(timeout)`, `wait_for_ready(timeout)`
- `send_action(action, ticks)` (bitmask or key vector), `send_action_bits(bits, ticks)`, `send_keys("right", "jump", ticks_per_step=1)`, `wait_for_frame(timeout) -> (frame, ModState)`, `step(action, ticks, timeout)`
- `send_command(command, timeout)`, `reset_game(timeout)`, `kill_player(timeout)` (debug)
- `read_state() -> ModState` (whole 64-byte header via one `struct.Struct` read), `set_connected(bool)` (read-modify-write of Python's bit only)

`mapping_factory` lets tests inject a `bytearray`-backed fake; see
`rainworld_rl/tests_unit/`. Run the game-free tests with
`python -m pytest rainworld_rl/tests_unit -q` (or everything game-free with `python -m pytest -q`).
