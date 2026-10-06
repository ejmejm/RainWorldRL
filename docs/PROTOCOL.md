# Shared Memory Protocol (v4)

The version is `PROTOCOL_VERSION` in `rainworld_rl/shared_memory.py` and `SharedMemoryBridge.cs`, sent in
the header at offset 15. Bump it whenever the header layout or the meaning of a field/command changes.

Named memory-mapped file `RainWorldRL`, created by whichever side comes first
(`MemoryMappedFile.CreateOrOpen` in C#, `mmap(-1, size, tagname=...)` in Python).
On Linux, where the game runs under Wine, the mapping is instead the file named by
the `RAINWORLD_RL_SHM` environment variable (a `/dev/shm` file, which Wine sees as
`Z:\dev\shm\...`), mapped by both sides; the launcher sets it per instance. If
`RAINWORLD_RL_SAVE_DIR` is set, the mod keeps the RL save there.
Total size = `HEADER_SIZE + MAX_FRAME_SIZE` = 64 + 1920*1080*3.

All multi-byte integers are little-endian. Floats are IEEE-754 float32.

## Header layout (64 bytes)

| Offset | Size | Dir     | Field              | Notes |
|-------:|-----:|---------|--------------------|-------|
| 0      | 1    | both    | `sync_flag`        | 0 IDLE, 1 ACTION_READY, 2 FRAME_READY, 3 PROCESSING |
| 1      | 1    | -       | reserved           | Was the v2 `action` byte. **No longer read by the mod**; v3 clients write `action_bits` at offset 44. Leave 0. |
| 2      | 1    | py→mod  | `ticks_per_step`   | 1..255, 0 treated as 1 |
| 3      | 1    | both    | `status`           | see Status bits |
| 4      | 4    | py→mod  | `frame_width`      | uint32, clamped by mod to 1..1920 (default 160) |
| 8      | 4    | py→mod  | `frame_height`     | uint32, clamped by mod to 1..1080 (default 90) |
| 12     | 1    | py→mod  | `command`          | 0 NONE, 1 RESET (wipe RL save, start fresh game), 2 KILL_PLAYER (debug: kill player 0 so the death edge and respawn flow can be tested), 3 ENTER_SHELTER (debug: send player 0 into its den shelter with `command_arg` food pips so a real hibernation can be tested). Mod sets back to 0 when done. |
| 13     | 1    | mod→py  | `command_result`   | 0 none/in-progress, 1 OK, 2 ERROR. Mod writes after finishing a command; Python clears to 0 before issuing the next. |
| 14     | 1    | mod→py  | `game_flags`       | b0 IN_SHELTER (level), b1 CYCLE_SURVIVED (edge: hibernation succeeded during this step), b2 RAIN (level: cycle end has begun), b3 DIALOG_OPEN (level: an in-game prompt awaits a key press - a `Menu.Dialog` side process, the game-over "press X to restart" prompt, or an open pause menu; see **Action bits**) |
| 15     | 1    | mod→py  | `protocol_version` | uint8, `PROTOCOL_VERSION` (4), written as soon as the mapping exists, so it is present whenever `MOD_ALIVE` is. A mod build from before versioning leaves 0. |
| 16     | 4    | mod→py  | `heartbeat`        | uint32, incremented every Unity Update while the mod is alive (even in menus). Python uses it to detect a live game. |
| 20     | 4    | mod→py  | `step_counter`     | uint32, incremented once per completed step (frame signalled) |
| 24     | 1    | mod→py  | `karma`            | uint8, current karma level (0 if not in game) |
| 25     | 1    | mod→py  | `karma_cap`        | uint8 |
| 26     | 1    | mod→py  | `food`             | uint8, food pips |
| 27     | 1    | mod→py  | `food_max`         | uint8, the slugcat's maximum food pips (`SlugcatStats.maxFood`; Survivor 7) |
| 28     | 4    | mod→py  | `player_x`         | float32, player body chunk 0 position in room coords (0 if unavailable) |
| 32     | 4    | mod→py  | `player_y`         | float32 |
| 36     | 4    | mod→py  | `room_index`       | int32, abstract room index (-1 if unavailable) |
| 40     | 4    | mod→py  | `cycle_number`     | int32, save-state cycle number (-1 if unavailable) |
| 44     | 4    | py→mod  | `action_bits`      | uint32, raw-key action bitfield: one bit per held key, any combination. Bit assignments in **Action bits** below. Bits above the defined keys are ignored. |
| 48     | 4    | mod→py  | `cycle_progress`   | float32, `RainCycle.timer / cycleLength` (0..1; >1 once the rain is falling, the timer keeps counting; 0 if unavailable). Note: in the first cycle of a fresh save the overseer tutorial pins `timer` to 2000 (`OverseerTutorialBehavior.pauseRain`) until the player leaves the start rooms (x > 600 in `SU_A43`, or `SU_A22`), then fast-forwards it. |
| 52     | 1    | mod→py  | `food_to_hibernate`| uint8, food pips needed to hibernate this cycle (`SlugcatStats.foodToHibernate`; Survivor 4; equals `food_max` while malnourished; 0 if unavailable) |
| 53     | 1    | mod→py  | `malnourished`     | uint8 0/1 (level), `SaveState.malnourished`: the previous sleep was a starving one, so this cycle needs `food_max` pips to sleep |
| 54     | 4    | mod→py  | `region`           | ASCII region acronym of the active world (`World.region.name`, e.g. `SU`, `HI`), NUL-padded to 4 bytes; all NUL (empty string) when no world is loaded. Combine with `room_index` for a save-wide room key: room indices are only unique within a region. |
| 58     | 1    | py→mod  | `command_arg`      | uint8 argument of `command`, written before it (ENTER_SHELTER: food pips; ignored by the others) |
| 59     | 5    | -       | reserved           | |
| 64     | N    | mod→py  | `frame`            | RGB24, row-major, top row first, width*height*3 bytes |

## Action bits (offset 44)

One bit per physical key a player can hold. Any combination may be set in the same step;
the mod holds exactly these keys for every physics tick of the step. The authoritative list
lives here and in `rainworld_rl/shared_memory.py` (`KEY_*`, `KEY_NAMES` in bit order) and
`SharedMemoryBridge.cs` (`KEY_*`); the three must match. Python's gym action space is
`MultiBinary(9)` in this bit order.

| Bit | Key | Mask | Game binding (Rain World v1.11.8) | Effect |
|----:|-----|-----:|-----------------------------------|--------|
| 0 | `left`    | 0x001 | Rewired `MoveHorizontal` (1) negative | `InputPackage.x = -1`. Left + right held together = 0. |
| 1 | `right`   | 0x002 | `MoveHorizontal` (1) positive | `InputPackage.x = +1`. |
| 2 | `up`      | 0x004 | `MoveVertical` (2) positive | `InputPackage.y = +1`. Up + down together = 0. |
| 3 | `down`    | 0x008 | `MoveVertical` (2) negative | `InputPackage.y = -1`; with a side key also sets `downDiagonal` (crawl/roll). |
| 4 | `jump`    | 0x010 | `Jump` (0); UI `UISubmit` (8) | `jmp`. In dialogs / menus: press the selected button ("continue"). |
| 5 | `grab`    | 0x020 | `Take` (3) | `pckp`: pick up, eat, interact, swallow. |
| 6 | `throw`   | 0x040 | `Throw` (4); UI `UICancel` (9) | `thrw`. In dialogs: cancel / back. |
| 7 | `map`     | 0x080 | `Map` (11); UI `UICheatHoldRight` (13) | `mp`: hold to show the map. Also the "press SPACE to restart" key of the game-over prompt (`HUD.TextPrompt`). |
| 8 | `special` | 0x100 | `Special` (34) | `spec`: slugcat ability - Watcher warp / camouflage, Saint ascension, Artificer pyro-jump hold. No effect for Survivor/Monk/Hunter. |

`analogueDir` is the normalised (x, y) of the held direction keys; `gamePad = false`,
`controllerType = KeyboardSinglePlayer`, `crouchToggle = false` (the game never sets it from
any input in this version).

**Deliberately not keys**: Pause (Rewired action 5 / Escape), menu-exit and quit. While the agent
is in control (RL mode on, no human override) the mod blocks `RWInput.CheckPauseButton` so neither
a stray Escape nor a controller start button can open the pause menu; if a pause menu is open when
the agent regains control the mod dismisses it (CONTINUE).

**Prompts.** While the agent is in control, the mod also feeds `action_bits` to
`RWInput.PlayerUIInput` (menu/dialog input) and to `Options.ControlSetup.GetButton` for player 0,
but only while a `RainWorldGame` is current or a `Menu.Dialog` is running - so in-game prompts
(dialog boxes, the game-over prompt, arena overlays) can be answered with `jump`/`throw`/`map`,
while the menus the mod navigates itself between games are left alone. Conversation text
(`HUD.DialogBox`: iterators, Watcher dialogue, tutorial text) advances on a timer and needs no key.
`game_flags.DIALOG_OPEN` tells the client when such a prompt is waiting.

## Game flags (offset 14)

Mod-owned byte; every writer uses a read-modify-write of its own bit(s)
(`SharedMemoryBridge.SetGameFlag`). Bits 0-2 are written per step together with the
other game-state fields and cleared (with them) when no game state is available.

| Bit | Name              | Kind  | Meaning |
|----:|-------------------|-------|---------|
| 0   | `IN_SHELTER`      | level | Player 0 is in a shelter room (`AbstractRoom.shelter`). |
| 1   | `CYCLE_SURVIVED`  | edge  | Set for exactly one step: the player hibernated **with enough food** since the previous step (`RainWorldGame.Win` ran with `malnourished == false`, the SleepScreen path). A starving sleep (`Win(malnourished: true)`, StarveScreen) does **not** set it; it shows up as `cycle_number + 1` with `malnourished = 1`. The bit is delivered even if the game process has already switched for the sleep-screen redirect on that step (other fields may read as unavailable then). |
| 2   | `RAIN`            | level | The cycle timer has expired and the lethal rain is falling: `RainCycle.TimeUntilRain <= 0` (= `RainCycle.RainGameOver`), the same instant `cycle_progress` crosses 1.0. The visual darkening before that (`RainDarkPalette`) is *not* included. |
| 3   | `DIALOG_OPEN`     | level | A text/dialog overlay awaits player input. |

## Status bits (offset 3)

| Bit | Name              | Writer | Meaning |
|----:|-------------------|--------|---------|
| 0   | `PLAYER_DEAD`     | mod    | **Edge**: set for exactly the one step during which the player transitioned alive→dead. Cleared on the next step. |
| 1   | `CONNECTED`       | py     | Python client is attached and wants control. Mod auto-enables RL mode when this rises, disables when it falls. |
| 2   | `READY`           | mod    | Mod is in RL mode, the game process is `RainWorldGame`, and a player exists. Steps will be serviced. |
| 3   | `HUMAN_OVERRIDE`  | mod    | F10 toggle. Human has control; mod runs at real time and ignores agent actions. Python's step() must wait (not time out) while set. |
| 4   | `IN_GAME`         | mod    | Current main process is `RainWorldGame` (regardless of RL mode). |
| 5   | `MOD_ALIVE`       | mod    | Set once the mod's `Awake` has run; cleared in `OnDestroy`. Combine with `heartbeat` to detect a dead game. |

Bits 0, 2, 3, 4, 5 are owned by the mod; bit 1 is owned by Python. Each side must
read-modify-write only its own bits (read status, mask, write back).

## Step handshake

1. Python writes `action_bits`, `ticks_per_step`, then `sync_flag = ACTION_READY`.
2. Mod sees ACTION_READY → reads action_bits/ticks → writes `sync_flag = PROCESSING`. The keys stay held until the next action is consumed.
3. Mod runs exactly `ticks_per_step` game ticks (40 per second of game time), then renders the game camera once. While `READY` (the agent playing) this all happens inside one Unity frame, time stays frozen between steps, and the mod waits a few ms in that frame for the next action, so consecutive steps can share a frame. While not `READY` it unpauses at high timescale and counts FixedUpdates instead.
4. Mod (in the camera's post-render) captures the frame from the camera's render texture, writes all mod→py header fields (status edge bits, karma, food, pos, ...), increments `step_counter`, then writes `sync_flag = FRAME_READY` **last**.
5. Python sees FRAME_READY → reads frame + header fields → writes `sync_flag = IDLE`.

While `HUMAN_OVERRIDE` is set the mod does not service steps; Python keeps waiting
and polls at a lower rate. When override is cleared, a pending ACTION_READY is serviced normally.

If `READY` is clear (menu, death screen, loading), the mod still services steps:
it returns the current rendered frame so the agent never hangs, but game-state fields
are zeroed/-1 and `IN_GAME` is clear. (The mod is expected to auto-return to the game within a few frames.)

## Commands (offset 12)

Python clears `command_result` to 0, writes `command_arg` and a command (`RESET`, `KILL_PLAYER` or `ENTER_SHELTER`), then polls `command_result`
until nonzero (timeout ~60s). The mod performs the command on the main thread, writes
`command_result`, then writes `command = NONE`. RESET = delete the RL save directory
contents, start a fresh story game as the configured slugcat, wait until `READY`, then ack.
KILL_PLAYER = kill player 0 immediately (the following step reports the `PLAYER_DEAD` edge and the
normal respawn flow runs); ack once the kill has been applied (`ERROR` if RL mode is not fully on,
no realized player 0 exists, or it is already dead). The ack never waits for the respawn.
ENTER_SHELTER = set player 0's food to `command_arg` pips and send it into its den shelter (the save's
den, or the slugcat's default shelter while the den is not one, e.g. on a fresh save) through the shelter's
entrance pipe, as if it had walked in; ack at once (`ERROR` as for KILL_PLAYER, or if that shelter is not in
the current region). The game's own shelter logic takes over: with `food_to_hibernate` pips the slugcat
hibernates once it stands still more than 6 tiles from the entrance; with fewer (at least 1) holding `down`
there for 260 ticks makes it sleep starving. Either way the door closes in 320 ticks and the cycle reloads.

Death -> respawn (any death, not just KILL_PLAYER): the game only leaves its "game over" prompt on a
key press that injected RL input cannot produce, so while RL mode is on the mod presses it itself
40 ticks after the prompt appears (the game's own minimum) and the death screen is skipped straight
into a reload of the cycle. From Python: one step with `PLAYER_DEAD`, then `READY` drops for a
number of steps (steps are still serviced, fields zero/-1), then `READY` returns in the
start-of-cycle shelter.

## Connect handshake (Python side)

1. Open the mapping. Read `heartbeat` twice ~100 ms apart; if it did not change and `MOD_ALIVE` is clear → no game running → raise.
   Then check `protocol_version == PROTOCOL_VERSION`; on a mismatch close the mapping and raise `ProtocolVersionError` (without setting `CONNECTED`).
2. Write `frame_width/height`, set `CONNECTED`.
3. Wait for `READY` (timeout configurable, default 60 s; the mod may still be entering the game).
4. Done. `disconnect()` clears `CONNECTED` and the mod returns the game to normal play.
