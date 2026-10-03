# Shared Memory Protocol (v2)

Named memory-mapped file `RainWorldRL`, created by whichever side comes first
(`MemoryMappedFile.CreateOrOpen` in C#, `mmap(-1, size, tagname=...)` in Python).
Total size = `HEADER_SIZE + MAX_FRAME_SIZE` = 64 + 1920*1080*3.

All multi-byte integers are little-endian. Floats are IEEE-754 float32.

## Header layout (64 bytes)

| Offset | Size | Dir     | Field              | Notes |
|-------:|-----:|---------|--------------------|-------|
| 0      | 1    | both    | `sync_flag`        | 0 IDLE, 1 ACTION_READY, 2 FRAME_READY, 3 PROCESSING |
| 1      | 1    | py→mod  | `action`           | bitfield: b0 jump, b1 grab, b2 throw, b3-4 horiz (0 none,1 left,2 right), b5-6 vert (0 none,1 down,2 up) |
| 2      | 1    | py→mod  | `ticks_per_step`   | 1..255, 0 treated as 1 |
| 3      | 1    | both    | `status`           | see Status bits |
| 4      | 4    | py→mod  | `frame_width`      | uint32, clamped by mod to 1..1920 (default 160) |
| 8      | 4    | py→mod  | `frame_height`     | uint32, clamped by mod to 1..1080 (default 90) |
| 12     | 1    | py→mod  | `command`          | 0 NONE, 1 RESET (wipe RL save, start fresh game). Mod sets back to 0 when done. |
| 13     | 1    | mod→py  | `command_result`   | 0 none/in-progress, 1 OK, 2 ERROR. Mod writes after finishing a command; Python clears to 0 before issuing the next. |
| 14     | 2    | -       | reserved           | |
| 16     | 4    | mod→py  | `heartbeat`        | uint32, incremented every Unity Update while the mod is alive (even in menus). Python uses it to detect a live game. |
| 20     | 4    | mod→py  | `step_counter`     | uint32, incremented once per completed step (frame signalled) |
| 24     | 1    | mod→py  | `karma`            | uint8, current karma level (0 if not in game) |
| 25     | 1    | mod→py  | `karma_cap`        | uint8 |
| 26     | 1    | mod→py  | `food`             | uint8, food pips |
| 27     | 1    | -       | reserved           | |
| 28     | 4    | mod→py  | `player_x`         | float32, player body chunk 0 position in room coords (0 if unavailable) |
| 32     | 4    | mod→py  | `player_y`         | float32 |
| 36     | 4    | mod→py  | `room_index`       | int32, abstract room index (-1 if unavailable) |
| 40     | 4    | mod→py  | `cycle_number`     | int32, save-state cycle number (-1 if unavailable) |
| 44     | 20   | -       | reserved           | |
| 64     | N    | mod→py  | `frame`            | RGB24, row-major, top row first, width*height*3 bytes |

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

Python clears `command_result` to 0, writes `command = RESET`, then polls `command_result`
until nonzero (timeout ~60s). The mod performs the command on the main thread, writes
`command_result`, then writes `command = NONE`. RESET = delete the RL save directory
contents, start a fresh story game as the configured slugcat, wait until `READY`, then ack.

## Connect handshake (Python side)

1. Open the mapping. Read `heartbeat` twice ~100 ms apart; if it did not change and `MOD_ALIVE` is clear → no game running → raise.
2. Write `frame_width/height`, set `CONNECTED`.
3. Wait for `READY` (timeout configurable, default 60 s; the mod may still be entering the game).
4. Done. `disconnect()` clears `CONNECTED` and the mod returns the game to normal play.
