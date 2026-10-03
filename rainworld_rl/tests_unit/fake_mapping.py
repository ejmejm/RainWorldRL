"""
In-memory stand-in for the named shared-memory mapping, playing the mod's role.

``FakeMapping`` is a ``bytearray`` (so it satisfies the slice/buffer interface
``SharedMemoryClient`` uses) that can optionally emulate the mod on every
header read:

* ``alive``         - bump ``heartbeat`` and keep ``MOD_ALIVE`` set.
* ``service_steps`` - when ``sync_flag == ACTION_READY``, fill the frame,
  write the mod->py fields, bump ``step_counter`` and set ``FRAME_READY``
  (header fields first, flag last, as the protocol demands).
* ``service_commands`` - when ``command == RESET``, "wipe" and ack with
  ``command_result = OK``, ``command = NONE``, and raise ``READY``.

Everything happens synchronously inside ``__getitem__`` so tests are
deterministic; a few tests use a thread for the time-based behaviour.
"""

from __future__ import annotations

import struct
from typing import List

from rainworld_rl import shared_memory as sm


class FakeMapping(bytearray):
    def __init__(self, size: int = sm.TOTAL_SIZE, alive: bool = True, auto_ready: bool = True):
        super().__init__(size)
        self.alive = alive
        # Like the real mod: once Python sets CONNECTED, enter RL mode and
        # raise IN_GAME + READY on the next tick.
        self.auto_ready = auto_ready
        self.service_steps = True
        self.service_commands = True
        self.fail_commands = False
        self.closed = False
        self.steps_serviced: List[tuple] = []   # (action_bits, ticks)
        self.commands_received: List[int] = []
        self.fill_value = 7                      # byte written into the frame
        self.mod_fields = dict(
            karma = 3, karma_cap = 5, food = 2, player_x = 123.5, player_y = -4.25,
            room_index = 42, cycle_number = 1,
        )
        self.next_step_dead = False
        if alive:
            self.set_mod_bit(sm.STATUS_MOD_ALIVE, True)

    # -- mod-side helpers --------------------------------------------------

    def close(self) -> None:
        self.closed = True

    def raw(self, offset: int, size: int) -> bytes:
        return bytes(bytearray.__getitem__(self, slice(offset, offset + size)))

    def put(self, offset: int, data: bytes) -> None:
        bytearray.__setitem__(self, slice(offset, offset + len(data)), data)

    def header(self) -> sm.ModState:
        return sm.ModState.unpack(self.raw(0, sm.HEADER_SIZE))

    def set_mod_bit(self, bit: int, value: bool) -> None:
        status = self.raw(sm.OFFSET_STATUS, 1)[0]
        status = (status | bit) if value else (status & ~bit & 0xFF)
        self.put(sm.OFFSET_STATUS, bytes((status,)))

    def set_status_byte(self, value: int) -> None:
        self.put(sm.OFFSET_STATUS, bytes((value & 0xFF,)))

    def bump_heartbeat(self) -> None:
        hb = struct.unpack_from("<I", self, sm.OFFSET_HEARTBEAT)[0]
        self.put(sm.OFFSET_HEARTBEAT, struct.pack("<I", (hb + 1) & 0xFFFFFFFF))

    def write_mod_fields(self, dead: bool = False) -> None:
        f = self.mod_fields
        self.put(sm.OFFSET_KARMA, bytes((f["karma"], f["karma_cap"], f["food"])))
        self.put(sm.OFFSET_PLAYER_X, struct.pack("<ffii", f["player_x"], f["player_y"], f["room_index"], f["cycle_number"]))
        self.set_mod_bit(sm.STATUS_PLAYER_DEAD, dead)

    def service_step(self) -> None:
        h = self.header()
        # Like the mod: read the raw-key bitfield at offset 44 (the legacy byte at offset 1 is ignored).
        self.steps_serviced.append((h.action_bits, h.ticks_per_step))
        self.put(sm.OFFSET_SYNC_FLAG, bytes((sm.SYNC_PROCESSING,)))
        w, hgt = h.frame_width, h.frame_height
        size = w * hgt * 3
        self.put(sm.OFFSET_FRAME_DATA, bytes([self.fill_value]) * size)
        self.write_mod_fields(dead = self.next_step_dead)
        self.next_step_dead = False
        self.put(sm.OFFSET_STEP_COUNTER, struct.pack("<I", h.step_counter + 1))
        self.put(sm.OFFSET_SYNC_FLAG, bytes((sm.SYNC_FRAME_READY,)))  # flag LAST

    def service_command(self) -> None:
        h = self.header()
        self.commands_received.append(h.command)
        if h.command == sm.COMMAND_RESET and not self.fail_commands:
            self.mod_fields["cycle_number"] = 0
            self.set_mod_bit(sm.STATUS_READY, True)
            self.set_mod_bit(sm.STATUS_IN_GAME, True)
            result = sm.COMMAND_RESULT_OK
        else:
            result = sm.COMMAND_RESULT_ERROR
        self.put(sm.OFFSET_COMMAND_RESULT, bytes((result,)))
        self.put(sm.OFFSET_COMMAND, bytes((sm.COMMAND_NONE,)))

    def tick(self) -> None:
        """One emulated Unity Update."""
        if not self.alive:
            return
        self.bump_heartbeat()
        h = self.header()
        if self.auto_ready and h.connected and not h.ready:
            self.set_mod_bit(sm.STATUS_IN_GAME, True)
            self.set_mod_bit(sm.STATUS_READY, True)
            h = self.header()
        if self.service_commands and h.command != sm.COMMAND_NONE:
            self.service_command()
        if self.service_steps and h.sync_flag == sm.SYNC_ACTION_READY and not h.human_override:
            self.service_step()

    # -- mapping interface used by the client ------------------------------

    def __getitem__(self, key):
        if isinstance(key, slice) and (key.start or 0) < sm.HEADER_SIZE:
            self.tick()
        return bytearray.__getitem__(self, key)
