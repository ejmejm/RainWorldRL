"""
Shared memory client for communicating with the Rain World RL mod.

Implements protocol v2 as described in ``docs/PROTOCOL.md``. The mapping is a
named memory-mapped file (``RainWorldRL``) consisting of a 64-byte header
followed by an RGB24 frame of up to 1920x1080.

Header layout (all little-endian)::

    Offset Size Dir     Field
    0      1    both    sync_flag       0 IDLE, 1 ACTION_READY, 2 FRAME_READY, 3 PROCESSING
    1      1    py->mod action          bitfield (b0 jump, b1 grab, b2 throw, b3-4 horiz, b5-6 vert)
    2      1    py->mod ticks_per_step  1..255 (0 treated as 1)
    3      1    both    status          see STATUS_* bits
    4      4    py->mod frame_width     uint32
    8      4    py->mod frame_height    uint32
    12     1    py->mod command         0 NONE, 1 RESET, 2 KILL_PLAYER (debug)
    13     1    mod->py command_result  0 none/in-progress, 1 OK, 2 ERROR
    14     2    -       reserved
    16     4    mod->py heartbeat       uint32, bumped every Unity Update
    20     4    mod->py step_counter    uint32, bumped once per completed step
    24     1    mod->py karma           uint8
    25     1    mod->py karma_cap       uint8
    26     1    mod->py food            uint8
    27     1    -       reserved
    28     4    mod->py player_x        float32
    32     4    mod->py player_y        float32
    36     4    mod->py room_index      int32 (-1 if unavailable)
    40     4    mod->py cycle_number    int32 (-1 if unavailable)
    44     20   -       reserved
    64     N    mod->py frame           RGB24, row-major, top row first

Status byte: bit 1 (``CONNECTED``) is owned by Python, every other bit is
owned by the mod. Each side only ever read-modify-writes its own bits.

The client talks to the mapping through a small "mapping" interface so that
unit tests can inject an in-memory stand-in: the object must support slice
reads (``m[a:b] -> bytes``), slice writes (``m[a:b] = data``), the buffer
protocol (for zero-extra-copy frame reads) and ``close()``. ``mmap.mmap``
and ``bytearray`` both satisfy this.
"""

from __future__ import annotations

import logging
import mmap
import struct
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARED_MEMORY_NAME = "RainWorldRL"
HEADER_SIZE = 64
MAX_FRAME_WIDTH = 1920
MAX_FRAME_HEIGHT = 1080
MAX_FRAME_SIZE = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * 3
TOTAL_SIZE = HEADER_SIZE + MAX_FRAME_SIZE

# Sync flag values
SYNC_IDLE = 0
SYNC_ACTION_READY = 1
SYNC_FRAME_READY = 2
SYNC_PROCESSING = 3

# Header offsets
OFFSET_SYNC_FLAG = 0
OFFSET_ACTION = 1
OFFSET_TICKS_PER_STEP = 2
OFFSET_STATUS = 3
OFFSET_WIDTH = 4
OFFSET_HEIGHT = 8
OFFSET_COMMAND = 12
OFFSET_COMMAND_RESULT = 13
OFFSET_GAME_FLAGS = 14
OFFSET_HEARTBEAT = 16
OFFSET_STEP_COUNTER = 20
OFFSET_KARMA = 24
OFFSET_KARMA_CAP = 25
OFFSET_FOOD = 26
OFFSET_FOOD_MAX = 27
OFFSET_PLAYER_X = 28
OFFSET_PLAYER_Y = 32
OFFSET_ROOM_INDEX = 36
OFFSET_CYCLE_NUMBER = 40
OFFSET_ACTION_BITS = 44       # py->mod uint32, protocol v3 action bitfield (see PROTOCOL.md)
OFFSET_CYCLE_PROGRESS = 48    # mod->py float32
OFFSET_FRAME_DATA = HEADER_SIZE

# Action bitfield
ACTION_JUMP = 0x01
ACTION_GRAB = 0x02
ACTION_THROW = 0x04
ACTION_HORIZONTAL_SHIFT = 3  # bits 3-4: 0 none, 1 left, 2 right
ACTION_VERTICAL_SHIFT = 5    # bits 5-6: 0 none, 1 down, 2 up

# Status bits
STATUS_PLAYER_DEAD = 0x01     # mod, edge: set for the one step the player died
STATUS_CONNECTED = 0x02       # python
STATUS_READY = 0x04           # mod: RL mode, in RainWorldGame, player exists
STATUS_HUMAN_OVERRIDE = 0x08  # mod: F10 toggle, human has control
STATUS_IN_GAME = 0x10         # mod: main process is RainWorldGame
STATUS_MOD_ALIVE = 0x20       # mod: Awake ran, OnDestroy has not

PY_OWNED_STATUS_MASK = STATUS_CONNECTED
MOD_OWNED_STATUS_MASK = 0xFF & ~PY_OWNED_STATUS_MASK

# Commands
COMMAND_NONE = 0
COMMAND_RESET = 1
COMMAND_KILL_PLAYER = 2       # debug: kill player 0 (death edge + respawn flow)

# Command results
COMMAND_RESULT_PENDING = 0
COMMAND_RESULT_OK = 1
COMMAND_RESULT_ERROR = 2

# game_flags (offset 14, mod-owned, protocol v3)
GAME_FLAG_IN_SHELTER = 0x01      # level: player is inside a shelter room
GAME_FLAG_CYCLE_SURVIVED = 0x02  # edge: the cycle was survived (hibernation) during this step
GAME_FLAG_RAIN = 0x04            # level: the rain/cycle-end has started
GAME_FLAG_DIALOG_OPEN = 0x08     # level: an in-game text/dialog overlay awaits player input

NUM_DISCRETE_ACTIONS = 18

# Whole-header struct. Field order matches the layout table above.
HEADER_STRUCT = struct.Struct(
    "<"
    "B"    # sync_flag
    "B"    # action
    "B"    # ticks_per_step
    "B"    # status
    "I"    # frame_width
    "I"    # frame_height
    "B"    # command
    "B"    # command_result
    "B"    # game_flags
    "x"    # reserved
    "I"    # heartbeat
    "I"    # step_counter
    "B"    # karma
    "B"    # karma_cap
    "B"    # food
    "B"    # food_max
    "f"    # player_x
    "f"    # player_y
    "i"    # room_index
    "i"    # cycle_number
    "I"    # action_bits
    "f"    # cycle_progress
    "12x"  # reserved
)
assert HEADER_STRUCT.size == HEADER_SIZE, HEADER_STRUCT.size

_UINT32 = struct.Struct("<I")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SharedMemoryError(RuntimeError):
    """Base class for shared-memory protocol errors."""


class GameNotRunningError(SharedMemoryError):
    """No live Rain World + RainWorldRL mod was detected behind the mapping."""


class NotConnectedError(SharedMemoryError):
    """An operation that requires ``connect()`` was called while disconnected."""


class StepTimeoutError(SharedMemoryError):
    """The mod did not deliver a frame within the allotted time."""


class ReadyTimeoutError(SharedMemoryError):
    """The mod did not raise READY within the allotted time."""


class CommandError(SharedMemoryError):
    """The mod reported an error, or timed out, while executing a command."""


# ---------------------------------------------------------------------------
# Header dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModState:
    """
    Snapshot of the 64-byte shared header.

    Holds every field (both directions) so a single ``read_state()`` call
    gives a consistent view. Decoded status bits are exposed as properties.
    """

    sync_flag: int = SYNC_IDLE
    action: int = 0
    ticks_per_step: int = 1
    status: int = 0
    frame_width: int = 0
    frame_height: int = 0
    command: int = COMMAND_NONE
    command_result: int = COMMAND_RESULT_PENDING
    game_flags: int = 0
    heartbeat: int = 0
    step_counter: int = 0
    karma: int = 0
    karma_cap: int = 0
    food: int = 0
    food_max: int = 0
    player_x: float = 0.0
    player_y: float = 0.0
    room_index: int = -1
    cycle_number: int = -1
    action_bits: int = 0
    cycle_progress: float = 0.0

    # -- status bits -------------------------------------------------------

    @property
    def player_dead(self) -> bool:
        return bool(self.status & STATUS_PLAYER_DEAD)

    @property
    def connected(self) -> bool:
        return bool(self.status & STATUS_CONNECTED)

    @property
    def ready(self) -> bool:
        return bool(self.status & STATUS_READY)

    @property
    def human_override(self) -> bool:
        return bool(self.status & STATUS_HUMAN_OVERRIDE)

    @property
    def in_game(self) -> bool:
        return bool(self.status & STATUS_IN_GAME)

    @property
    def mod_alive(self) -> bool:
        return bool(self.status & STATUS_MOD_ALIVE)

    @property
    def in_shelter(self) -> bool:
        return bool(self.game_flags & GAME_FLAG_IN_SHELTER)

    @property
    def cycle_survived(self) -> bool:
        return bool(self.game_flags & GAME_FLAG_CYCLE_SURVIVED)

    @property
    def rain(self) -> bool:
        return bool(self.game_flags & GAME_FLAG_RAIN)

    @property
    def dialog_open(self) -> bool:
        return bool(self.game_flags & GAME_FLAG_DIALOG_OPEN)

    @property
    def player_pos(self) -> Tuple[float, float]:
        return (self.player_x, self.player_y)

    # -- (de)serialisation -------------------------------------------------

    def pack(self) -> bytes:
        """Serialise to the 64-byte header layout."""
        return HEADER_STRUCT.pack(
            self.sync_flag & 0xFF,
            self.action & 0xFF,
            self.ticks_per_step & 0xFF,
            self.status & 0xFF,
            self.frame_width & 0xFFFFFFFF,
            self.frame_height & 0xFFFFFFFF,
            self.command & 0xFF,
            self.command_result & 0xFF,
            self.game_flags & 0xFF,
            self.heartbeat & 0xFFFFFFFF,
            self.step_counter & 0xFFFFFFFF,
            self.karma & 0xFF,
            self.karma_cap & 0xFF,
            self.food & 0xFF,
            self.food_max & 0xFF,
            float(self.player_x),
            float(self.player_y),
            int(self.room_index),
            int(self.cycle_number),
            self.action_bits & 0xFFFFFFFF,
            float(self.cycle_progress),
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ModState":
        """Parse a 64-byte header buffer."""
        fields = HEADER_STRUCT.unpack(data[:HEADER_SIZE])
        return cls(*fields)

    def to_info(self) -> Dict[str, Any]:
        """The mod->py fields in the shape used for the gym ``info`` dict."""
        return {
            "player_dead": self.player_dead,
            "karma": self.karma,
            "karma_cap": self.karma_cap,
            "food": self.food,
            "food_max": self.food_max,
            "player_pos": (self.player_x, self.player_y),
            "room_index": self.room_index,
            "cycle_number": self.cycle_number,
            "step_counter": self.step_counter,
            "in_game": self.in_game,
            "ready": self.ready,
            "human_override": self.human_override,
            "in_shelter": self.in_shelter,
            "cycle_survived": self.cycle_survived,
            "rain": self.rain,
            "dialog_open": self.dialog_open,
            "cycle_progress": self.cycle_progress,
        }


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def encode_action(
    jump: bool = False,
    grab: bool = False,
    throw: bool = False,
    horizontal: int = 0,
    vertical: int = 0,
) -> int:
    """Build the action bitfield byte from individual inputs."""
    action = 0
    if jump:
        action |= ACTION_JUMP
    if grab:
        action |= ACTION_GRAB
    if throw:
        action |= ACTION_THROW
    h_val = 0 if horizontal == 0 else (1 if horizontal < 0 else 2)
    v_val = 0 if vertical == 0 else (1 if vertical < 0 else 2)
    action |= h_val << ACTION_HORIZONTAL_SHIFT
    action |= v_val << ACTION_VERTICAL_SHIFT
    return action


# (jump, grab, throw, horizontal, vertical) per discrete action index.
DISCRETE_ACTIONS: Tuple[Tuple[bool, bool, bool, int, int], ...] = (
    (False, False, False, 0, 0),    # 0  No-op
    (False, False, False, -1, 0),   # 1  Left
    (False, False, False, 1, 0),    # 2  Right
    (False, False, False, 0, 1),    # 3  Up
    (False, False, False, 0, -1),   # 4  Down
    (True, False, False, 0, 0),     # 5  Jump
    (False, True, False, 0, 0),     # 6  Grab
    (False, False, True, 0, 0),     # 7  Throw
    (True, False, False, -1, 0),    # 8  Left + Jump
    (True, False, False, 1, 0),     # 9  Right + Jump
    (True, False, False, 0, 1),     # 10 Up + Jump
    (True, False, False, 0, -1),    # 11 Down + Jump
    (False, True, False, -1, 0),    # 12 Left + Grab
    (False, True, False, 1, 0),     # 13 Right + Grab
    (False, True, False, 0, 1),     # 14 Up + Grab
    (False, True, False, 0, -1),    # 15 Down + Grab
    (False, False, False, -1, -1),  # 16 Left + Down (crawl left)
    (False, False, False, 1, -1),   # 17 Right + Down (crawl right)
)
assert len(DISCRETE_ACTIONS) == NUM_DISCRETE_ACTIONS

DISCRETE_ACTION_NAMES: Tuple[str, ...] = (
    "No-op", "Left", "Right", "Up", "Down",
    "Jump", "Grab", "Throw",
    "Left+Jump", "Right+Jump", "Up+Jump", "Down+Jump",
    "Left+Grab", "Right+Grab", "Up+Grab", "Down+Grab",
    "Crawl Left", "Crawl Right",
)


def encode_discrete_action(action: int) -> int:
    """Map a discrete action index (0-17) to the action bitfield byte."""
    if not 0 <= action < NUM_DISCRETE_ACTIONS:
        raise ValueError(f"Discrete action must be in [0, {NUM_DISCRETE_ACTIONS}), got {action}")
    return encode_action(*DISCRETE_ACTIONS[action])


# ---------------------------------------------------------------------------
# Mapping factory
# ---------------------------------------------------------------------------

def open_mapping(name: str = SHARED_MEMORY_NAME, size: int = TOTAL_SIZE) -> mmap.mmap:
    """
    Open (or create) the named Windows file mapping.

    Note that ``mmap(-1, size, tagname=...)`` creates the mapping when nothing
    else holds it, so successfully opening it says nothing about whether the
    game is running; use the heartbeat check for that.
    """
    return mmap.mmap(-1, size, tagname = name)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class SharedMemoryClient:
    """
    Client for shared memory communication with the Rain World RL mod.

    Typical use::

        client = SharedMemoryClient(160, 90)
        client.connect()                     # raises GameNotRunningError if no game
        client.reset_game()                  # optional: wipe save, fresh game
        frame, state = client.step(action = 5, ticks_per_step = 4)
        client.disconnect()
    """

    # How long the heartbeat may stay static (while the mod claims to be
    # alive) before we treat the game as dead/hung during long waits.
    heartbeat_stall_timeout: float = 10.0

    # Poll intervals.
    fast_poll_interval: float = 0.0001   # waiting for a frame
    slow_poll_interval: float = 0.05     # waiting under HUMAN_OVERRIDE / for READY / commands

    def __init__(
        self,
        frame_width: int = 160,
        frame_height: int = 90,
        debug_timing: bool = False,
        mapping_factory: Optional[Callable[[], Any]] = None,
    ):
        """
        Args:
            frame_width: Desired frame width in pixels (1..1920).
            frame_height: Desired frame height in pixels (1..1080).
            debug_timing: If True, print per-phase timing info every ~2 s.
            mapping_factory: Callable returning a mapping object (see module
                docstring). Defaults to opening the real named mapping. Used
                by unit tests to inject an in-memory fake.
        """
        self._validate_dims(frame_width, frame_height)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_size = frame_width * frame_height * 3
        self.debug_timing = debug_timing
        self._mapping_factory = mapping_factory or open_mapping

        self._shm: Optional[Any] = None
        self._connected = False
        self._override_logged = False

        # Timing stats
        self._timing_samples = 0
        self._send_action_time = 0.0
        self._wait_poll_time = 0.0
        self._read_frame_time = 0.0
        self._reset_flag_time = 0.0
        self._last_timing_log = time.time()

    # -- lifecycle ---------------------------------------------------------

    @staticmethod
    def _validate_dims(width: int, height: int) -> None:
        if not 1 <= width <= MAX_FRAME_WIDTH or not 1 <= height <= MAX_FRAME_HEIGHT:
            raise ValueError(
                f"Frame size {width}x{height} out of range "
                f"(1..{MAX_FRAME_WIDTH} x 1..{MAX_FRAME_HEIGHT})"
            )

    def _open(self) -> None:
        if self._shm is None:
            self._shm = self._mapping_factory()

    def _close(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._shm = None

    def is_connected(self) -> bool:
        """True once ``connect()`` has succeeded and ``disconnect()`` has not been called."""
        return self._connected and self._shm is not None

    def _require_connected(self) -> None:
        if not self.is_connected():
            raise NotConnectedError("Not connected to shared memory; call connect() first")

    def wait_for_alive(self, timeout: float = 2.0, poll_interval: float = 0.01) -> ModState:
        """
        Open the mapping (without setting CONNECTED) and wait until the mod's
        heartbeat advances.

        Returns the header snapshot that showed a live heartbeat.

        Raises:
            GameNotRunningError: if the heartbeat stayed static for ``timeout``.
        """
        self._open()
        first = self.read_state()
        deadline = time.monotonic() + timeout
        while True:
            time.sleep(poll_interval)
            state = self.read_state()
            if state.heartbeat != first.heartbeat:
                return state
            if time.monotonic() >= deadline:
                break

        if not first.mod_alive:
            raise GameNotRunningError(
                "No running Rain World instance with the RainWorldRL mod was detected "
                "(heartbeat static, MOD_ALIVE clear)."
            )
        raise GameNotRunningError(
            f"The RainWorldRL mod reports MOD_ALIVE but its heartbeat did not advance in "
            f"{timeout:.1f}s; the game is hung or the mapping is stale."
        )

    def connect(
        self,
        wait_ready: bool = True,
        ready_timeout: float = 60.0,
        liveness_timeout: float = 2.0,
    ) -> ModState:
        """
        Attach to a running game.

        1. Open the mapping and confirm the mod is alive (heartbeat advances).
        2. Write the requested frame dimensions and raise ``CONNECTED``.
        3. Optionally wait for ``READY``.

        Returns the last header snapshot read.

        Raises:
            GameNotRunningError: no live mod behind the mapping. The mapping is
                closed again before raising.
            ReadyTimeoutError: ``wait_ready`` was set and READY never rose.
        """
        try:
            state = self.wait_for_alive(timeout = liveness_timeout)
        except GameNotRunningError:
            self._close()
            raise

        self._write_frame_dimensions()
        self.set_connected(True)
        self._connected = True
        self._override_logged = False
        logger.info("Connected to RainWorldRL mod (heartbeat %d)", state.heartbeat)

        if wait_ready:
            state = self.wait_for_ready(timeout = ready_timeout)
        return state

    def disconnect(self) -> None:
        """Clear ``CONNECTED`` (returning the game to normal play) and close the mapping."""
        if self._shm is not None:
            try:
                self.set_connected(False)
            except Exception:  # pragma: no cover - best effort
                pass
        self._close()
        self._connected = False

    def __enter__(self) -> "SharedMemoryClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.disconnect()
        return False

    # -- raw access --------------------------------------------------------

    def _write(self, offset: int, data: bytes) -> None:
        self._shm[offset:offset + len(data)] = data

    def _write_byte(self, offset: int, value: int) -> None:
        self._shm[offset:offset + 1] = bytes((value & 0xFF,))

    def _read_byte(self, offset: int) -> int:
        return self._shm[offset:offset + 1][0]

    def _write_uint32(self, offset: int, value: int) -> None:
        self._write(offset, _UINT32.pack(value & 0xFFFFFFFF))

    def read_state(self) -> ModState:
        """Read the whole 64-byte header in one go."""
        if self._shm is None:
            raise NotConnectedError("Mapping is not open")
        return ModState.unpack(self._shm[0:HEADER_SIZE])

    def read_status(self) -> int:
        """Read the raw status byte."""
        if self._shm is None:
            raise NotConnectedError("Mapping is not open")
        return self._read_byte(OFFSET_STATUS)

    def set_connected(self, value: bool) -> None:
        """Read-modify-write Python's ``CONNECTED`` bit, leaving the mod's bits intact."""
        status = self._read_byte(OFFSET_STATUS)
        if value:
            status |= STATUS_CONNECTED
        else:
            status &= ~STATUS_CONNECTED & 0xFF
        self._write_byte(OFFSET_STATUS, status)

    def _write_frame_dimensions(self) -> None:
        self._write_uint32(OFFSET_WIDTH, self.frame_width)
        self._write_uint32(OFFSET_HEIGHT, self.frame_height)

    def set_frame_dimensions(self, width: int, height: int) -> None:
        """Change the frame size. Takes effect on the next step the mod services."""
        self._validate_dims(width, height)
        self.frame_width = width
        self.frame_height = height
        self.frame_size = width * height * 3
        if self._shm is not None:
            self._write_frame_dimensions()

    # -- readiness ---------------------------------------------------------

    def wait_for_ready(self, timeout: float = 60.0) -> ModState:
        """
        Block until the mod raises ``READY``.

        Raises:
            ReadyTimeoutError: READY did not rise in time.
            GameNotRunningError: the heartbeat stalled for ``heartbeat_stall_timeout``.
        """
        self._require_connected()
        deadline = time.monotonic() + timeout
        state = self.read_state()
        last_hb = state.heartbeat
        last_hb_change = time.monotonic()
        while True:
            if state.ready:
                return state
            now = time.monotonic()
            if state.heartbeat != last_hb:
                last_hb = state.heartbeat
                last_hb_change = now
            elif now - last_hb_change > self.heartbeat_stall_timeout:
                raise GameNotRunningError(
                    f"Mod heartbeat stalled for {self.heartbeat_stall_timeout:.0f}s while waiting for READY"
                )
            if now >= deadline:
                raise ReadyTimeoutError(
                    f"Mod did not become READY within {timeout:.0f}s "
                    f"(in_game={state.in_game}, human_override={state.human_override})"
                )
            time.sleep(self.slow_poll_interval)
            state = self.read_state()

    # -- stepping ----------------------------------------------------------

    def send_action(
        self,
        jump: bool = False,
        grab: bool = False,
        throw: bool = False,
        horizontal: int = 0,
        vertical: int = 0,
        ticks_per_step: int = 1,
    ) -> None:
        """
        Write an action and signal ``ACTION_READY``.

        Args:
            jump / grab / throw: Button states.
            horizontal: -1 left, 0 none, 1 right.
            vertical: -1 down, 0 none, 1 up.
            ticks_per_step: Physics ticks to run before the frame is returned (1..255).
        """
        self.send_action_raw(encode_action(jump, grab, throw, horizontal, vertical), ticks_per_step)

    def send_action_raw(self, action_byte: int, ticks_per_step: int = 1) -> None:
        """Write a pre-encoded action byte and signal ``ACTION_READY``."""
        self._require_connected()
        send_start = time.perf_counter()
        self._write_byte(OFFSET_ACTION, action_byte)
        self._write_byte(OFFSET_TICKS_PER_STEP, max(1, min(255, int(ticks_per_step))))
        self._write_byte(OFFSET_SYNC_FLAG, SYNC_ACTION_READY)
        self._send_action_time += time.perf_counter() - send_start

    def send_action_discrete(self, action: int, ticks_per_step: int = 1) -> None:
        """
        Send one of the 18 discrete actions.

        Mapping (see ``DISCRETE_ACTION_NAMES``):
            0 No-op, 1 Left, 2 Right, 3 Up, 4 Down, 5 Jump, 6 Grab, 7 Throw,
            8 Left+Jump, 9 Right+Jump, 10 Up+Jump, 11 Down+Jump,
            12 Left+Grab, 13 Right+Grab, 14 Up+Grab, 15 Down+Grab,
            16 Left+Down (crawl left), 17 Right+Down (crawl right)
        """
        self.send_action_raw(encode_discrete_action(int(action)), ticks_per_step)

    def wait_for_frame(self, timeout: float = 10.0) -> Tuple[np.ndarray, ModState]:
        """
        Wait for ``FRAME_READY``, read the frame and header, then write ``IDLE``.

        While ``HUMAN_OVERRIDE`` is set this does not time out: it polls slowly
        and logs once that a human has taken over. The timeout clock restarts
        when the override is released.

        Returns:
            (frame, state) where frame is a fresh uint8 array of shape
            (height, width, 3) and state is the header snapshot taken at the
            moment FRAME_READY was observed (the mod writes it before the flag).

        Raises:
            StepTimeoutError: no frame within ``timeout`` seconds.
            GameNotRunningError: no frame and the heartbeat did not advance either.
        """
        self._require_connected()
        poll_start = time.perf_counter()
        deadline = time.monotonic() + timeout
        state = self.read_state()
        start_heartbeat = state.heartbeat

        while True:
            if state.sync_flag == SYNC_FRAME_READY:
                self._wait_poll_time += time.perf_counter() - poll_start
                frame = self._read_frame()

                reset_start = time.perf_counter()
                self._write_byte(OFFSET_SYNC_FLAG, SYNC_IDLE)
                self._reset_flag_time += time.perf_counter() - reset_start

                self._timing_samples += 1
                if self.debug_timing and time.time() - self._last_timing_log >= 2.0:
                    self._log_timing()
                return frame, state

            if state.human_override:
                if not self._override_logged:
                    logger.warning(
                        "Human override active (F10): the game is running at real time and "
                        "ignoring agent actions. step() is waiting until it is released."
                    )
                    self._override_logged = True
                time.sleep(self.slow_poll_interval)
                # Do not let the override count against the step timeout.
                deadline = time.monotonic() + timeout
                state = self.read_state()
                start_heartbeat = state.heartbeat
                continue

            if self._override_logged:
                logger.info("Human override released; resuming agent control.")
                self._override_logged = False

            if time.monotonic() >= deadline:
                if state.heartbeat == start_heartbeat:
                    raise GameNotRunningError(
                        f"No frame within {timeout:.1f}s and the mod heartbeat did not advance; "
                        "the game appears to have exited or hung."
                    )
                raise StepTimeoutError(
                    f"No frame within {timeout:.1f}s (sync_flag={state.sync_flag}, "
                    f"ready={state.ready}, in_game={state.in_game})"
                )

            # ACTION_READY (mod has not picked it up yet) or PROCESSING - keep waiting.
            time.sleep(self.fast_poll_interval)
            state = self.read_state()

    def _read_frame(self) -> np.ndarray:
        read_start = time.perf_counter()
        view = np.frombuffer(
            self._shm, dtype = np.uint8, count = self.frame_size, offset = OFFSET_FRAME_DATA
        )
        frame = view.reshape((self.frame_height, self.frame_width, 3)).copy()
        del view  # release the buffer export on the mapping
        self._read_frame_time += time.perf_counter() - read_start
        return frame

    def step(self, action: int, ticks_per_step: int = 1, timeout: float = 10.0) -> Tuple[np.ndarray, ModState]:
        """Convenience: ``send_action_discrete`` followed by ``wait_for_frame``."""
        self.send_action_discrete(action, ticks_per_step)
        return self.wait_for_frame(timeout = timeout)

    # -- commands ----------------------------------------------------------

    def send_command(self, command: int, timeout: float = 60.0) -> int:
        """
        Issue a command and block until the mod acknowledges it.

        Clears ``command_result``, writes ``command``, then polls
        ``command_result`` until it is non-zero.

        Returns:
            The result code (``COMMAND_RESULT_OK``).

        Raises:
            CommandError: the mod reported ``COMMAND_RESULT_ERROR`` or did not
                answer within ``timeout``.
        """
        self._require_connected()
        self._write_byte(OFFSET_COMMAND_RESULT, COMMAND_RESULT_PENDING)
        self._write_byte(OFFSET_COMMAND, command)

        deadline = time.monotonic() + timeout
        while True:
            state = self.read_state()
            if state.command_result != COMMAND_RESULT_PENDING:
                break
            if time.monotonic() >= deadline:
                raise CommandError(f"Command {command} was not acknowledged within {timeout:.0f}s")
            time.sleep(self.slow_poll_interval)

        if state.command_result == COMMAND_RESULT_ERROR:
            raise CommandError(f"Mod reported an error executing command {command}; see BepInEx/LogOutput.log")
        return state.command_result

    def reset_game(self, timeout: float = 60.0) -> ModState:
        """
        Send ``RESET``: wipe the RL save, start a fresh story game, wait for READY.

        A stale unread ``FRAME_READY`` is discarded first so the next step
        starts from ``IDLE``. Returns the header snapshot after the ack.
        """
        self._require_connected()
        if self._read_byte(OFFSET_SYNC_FLAG) == SYNC_FRAME_READY:
            self._write_byte(OFFSET_SYNC_FLAG, SYNC_IDLE)
        self.send_command(COMMAND_RESET, timeout = timeout)
        return self.read_state()

    def kill_player(self, timeout: float = 10.0) -> ModState:
        """
        Send ``KILL_PLAYER`` - a **debug/testing** command that kills player 0.

        The mod applies the kill synchronously and acks as soon as the slugcat
        is dead; it does *not* wait for the respawn. The next ``step()`` reports
        the ``PLAYER_DEAD`` edge, then (after the game's ~40-tick game-over
        prompt) the mod skips the death screen and reloads the cycle, during
        which ``READY`` drops and steps return menu/loading frames.

        Raises:
            CommandError: the mod reported ERROR (RL mode not fully on, no live
                player 0, player already dead) or did not ack within ``timeout``.
        """
        self._require_connected()
        self.send_command(COMMAND_KILL_PLAYER, timeout = timeout)
        return self.read_state()

    # -- misc --------------------------------------------------------------

    def get_status(self) -> Tuple[bool, bool]:
        """Backwards-compatible helper returning ``(player_dead, connected)``."""
        if not self.is_connected():
            return False, False
        status = self.read_status()
        return bool(status & STATUS_PLAYER_DEAD), bool(status & STATUS_CONNECTED)

    def _log_timing(self) -> None:
        if self._timing_samples == 0:
            return
        n = self._timing_samples
        total = (self._send_action_time + self._wait_poll_time +
                 self._read_frame_time + self._reset_flag_time)
        print(f"[SharedMemory] TIMING (avg over {n} steps):")
        print(f"  SendAction:   {self._send_action_time * 1000 / n:.3f}ms")
        print(f"  WaitPoll:     {self._wait_poll_time * 1000 / n:.3f}ms")
        print(f"  ReadFrame:    {self._read_frame_time * 1000 / n:.3f}ms")
        print(f"  ResetFlag:    {self._reset_flag_time * 1000 / n:.3f}ms")
        if total > 0:
            print(f"  TOTAL:        {total * 1000 / n:.3f}ms (max {n / total:.1f} FPS)")

        self._timing_samples = 0
        self._send_action_time = 0.0
        self._wait_poll_time = 0.0
        self._read_frame_time = 0.0
        self._reset_flag_time = 0.0
        self._last_timing_log = time.time()
