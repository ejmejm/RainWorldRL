# Shared Memory Protocol (v3)

Named memory-mapped file `RainWorldRL`, created by whichever side comes first
(`MemoryMappedFile.CreateOrOpen` in C#, `mmap(-1, size, tagname=...)` in Python).
Total size = `HEADER_SIZE + MAX_FRAME_SIZE` = 64 + 1920*1080*3.

All multi-byte integers are little-endian. Floats are IEEE-754 float32.

## Header layout (64 bytes)

| Offset | Size | Dir     | Field              | Notes |
|-------:|-----:|---------|--------------------|-------|
| 0      | 1    | both    | `sync_flag`        | 0 IDLE, 1 ACTION_READY, 2 FRAME_READY, 3 PROCESSING |
| 1      | 1    | py→mod  | `action` (legacy)  | v2 bitfield: b0 jump, b1 grab, b2 throw, b3-4 horiz (0 none,1 left,2 right), b5-6 vert (0 none,1 down,2 up). **v3 clients write `action_bits` at 44 instead; the mod prefers `action_bits` when nonzero.** |
| 2      | 1    | py→mod  | `ticks_per_step`   | 1..255, 0 treated as 1 |
| 3      | 1    | both    | `status`           | see Status bits |
| 4      | 4    | py→mod  | `frame_width`      | uint32, clamped by mod to 1..1920 (default 160) |
| 8      | 4    | py→mod  | `frame_height`     | uint32, clamped by mod to 1..1080 (default 90) |
| 12     | 1    | py→mod  | `command`          | 0 NONE, 1 RESET (wipe RL save, start fresh game), 2 KILL_PLAYER (debug: kill player 0 so the death edge and respawn flow can be tested). Mod sets back to 0 when done. |
| 13     | 1    | mod→py  | `command_result`   | 0 none/in-progress, 1 OK, 2 ERROR. Mod writes after finishing a command; Python clears to 0 before issuing the next. |
| 14     | 1    | mod→py  | `game_flags`       | b0 IN_SHELTER (level), b1 CYCLE_SURVIVED (edge), b2 RAIN (level), b3 DIALOG_OPEN (level: a text/dialog overlay awaits player input). See **Game flags** below. |
| 15     | 1    | -       | reserved           | |
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
| 44     | 4    | py→mod  | `action_bits`      | uint32, protocol v3 raw-key action bitfield. Bit assignments in **Action bits** below. |
| 48     | 4    | mod→py  | `cycle_progress`   | float32, `RainCycle.timer / cycleLength` (0..1; >1 once the rain is falling, the timer keeps counting; 0 if unavailable). Note: in the first cycle of a fresh save the overseer tutorial pins `timer` to 2000 (`OverseerTutorialBehavior.pauseRain`) until the player leaves the start rooms (x > 600 in `SU_A43`, or `SU_A22`), then fast-forwards it. |
| 52     | 1    | mod→py  | `food_to_hibernate`| uint8, food pips needed to hibernate this cycle (`SlugcatStats.foodToHibernate`; Survivor 4; equals `food_max` while malnourished; 0 if unavailable) |
| 53     | 1    | mod→py  | `malnourished`     | uint8 0/1 (level), `SaveState.malnourished`: the previous sleep was a starving one, so this cycle needs `food_max` pips to sleep |
| 54     | 10   | -       | reserved           | |
| 64     | N    | mod→py  | `frame`            | RGB24, row-major, top row first, width*height*3 bytes |

## Action bits (offset 44)

One bit per physical key a player can hold. Any combination may be set in the same step.
The authoritative list lives here and in `rainworld_rl/shared_memory.py` (`KEY_*`) and
`SharedMemoryBridge.cs` (`KEY_*`); the three must match.

| Bit | Key | Notes |
|----:|-----|-------|
| (to be filled in by the action-space work) | | |

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

1. Python writes `action`, `ticks_per_step`, then `sync_flag = ACTION_READY`.
2. Mod (in Update) sees ACTION_READY → reads action/ticks → writes `sync_flag = PROCESSING` → unpauses at high timescale.
3. Mod counts FixedUpdates; after `ticks_per_step` ticks it pauses (timescale 0).
4. Mod (in OnPostRender) captures the frame, writes all mod→py header fields (status edge bits, karma, food, pos, ...), increments `step_counter`, then writes `sync_flag = FRAME_READY` **last**.
5. Python sees FRAME_READY → reads frame + header fields → writes `sync_flag = IDLE`.

While `HUMAN_OVERRIDE` is set the mod does not service steps; Python keeps waiting
and polls at a lower rate. When override is cleared, a pending ACTION_READY is serviced normally.

If `READY` is clear (menu, death screen, loading), the mod still services steps:
it returns the current rendered frame so the agent never hangs, but game-state fields
are zeroed/-1 and `IN_GAME` is clear. (The mod is expected to auto-return to the game within a few frames.)

## Commands (offset 12)

Python clears `command_result` to 0, writes a command (`RESET` or `KILL_PLAYER`), then polls `command_result`
until nonzero (timeout ~60s). The mod performs the command on the main thread, writes
`command_result`, then writes `command = NONE`. RESET = delete the RL save directory
contents, start a fresh story game as the configured slugcat, wait until `READY`, then ack.
KILL_PLAYER = kill player 0 immediately (the following step reports the `PLAYER_DEAD` edge and the
normal respawn flow runs); ack once the kill has been applied.

## Connect handshake (Python side)

1. Open the mapping. Read `heartbeat` twice ~100 ms apart; if it did not change and `MOD_ALIVE` is clear → no game running → raise.
2. Write `frame_width/height`, set `CONNECTED`.
3. Wait for `READY` (timeout configurable, default 60 s; the mod may still be entering the game).
4. Done. `disconnect()` clears `CONNECTED` and the mod returns the game to normal play.
