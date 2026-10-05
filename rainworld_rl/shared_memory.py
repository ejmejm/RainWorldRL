"""
Shared memory client for communicating with the Rain World RL mod.

Implements protocol v3 as described in ``docs/PROTOCOL.md``. The mapping is a
named memory-mapped file (``RainWorldRL``) consisting of a 64-byte header
followed by an RGB24 frame of up to 1920x1080.

Header layout (all little-endian)::

    Offset Size Dir     Field
    0      1    both    sync_flag       0 IDLE, 1 ACTION_READY, 2 FRAME_READY, 3 PROCESSING
    1      1    -       reserved        (legacy v2 action byte, no longer used)
    2      1    py->mod ticks_per_step  1..255 (0 treated as 1)
    3      1    both    status          see STATUS_* bits
    4      4    py->mod frame_width     uint32
    8      4    py->mod frame_height    uint32
    12     1    py->mod command         0 NONE, 1 RESET, 2 KILL_PLAYER (debug)
    13     1    mod->py command_result  0 none/in-progress, 1 OK, 2 ERROR
    14     1    mod->py game_flags      see GAME_FLAG_* bits
    15     1    -       reserved
    16     4    mod->py heartbeat       uint32, bumped every Unity Update
    20     4    mod->py step_counter    uint32, bumped once per completed step
    24     1    mod->py karma           uint8
    25     1    mod->py karma_cap       uint8
    26     1    mod->py food            uint8
    27     1    mod->py food_max        uint8, the slugcat's maximum food pips
    28     4    mod->py player_x        float32
    32     4    mod->py player_y        float32
    36     4    mod->py room_index      int32 (-1 if unavailable)
    40     4    mod->py cycle_number    int32 (-1 if unavailable)
    44     4    py->mod action_bits     uint32, one bit per held key (KEY_* / KEY_NAMES)
    48     4    mod->py cycle_progress  float32, RainCycle timer / cycleLength (>1 once the rain falls)
    52     1    mod->py food_to_hibernate uint8, pips needed to sleep this cycle (== food_max while malnourished)
    53     1    mod->py malnourished    uint8 0/1, the previous sleep was a starving one
    54     4    mod->py region          ASCII region acronym (World.region.name, e.g. "SU"), NUL-padded; all NUL if unavailable
    58     6    -       reserved
    64     N    mod->py frame           RGB24, row-major, top row first

Actions are *raw key presses*: ``action_bits`` has one bit per key a player
can hold (``KEY_LEFT`` ... ``KEY_SPECIAL``), any combination at once. The
Discrete(18) action set lives in ``rainworld_rl.wrappers`` as an optional
``gym.ActionWrapper``.

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
# offset 1 is reserved (legacy v2 action byte; the mod no longer reads it)
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
OFFSET_FOOD_TO_HIBERNATE = 52 # mod->py uint8
OFFSET_MALNOURISHED = 53      # mod->py uint8 (0/1)
OFFSET_REGION = 54            # mod->py 4 bytes ASCII, NUL-padded (region acronym, "" if unavailable)
REGION_SIZE = 4
OFFSET_FRAME_DATA = HEADER_SIZE

# action_bits (offset 44): one bit per player key. Must match SharedMemoryBridge.cs KEY_*
# and the "Action bits" table in docs/PROTOCOL.md. Any combination may be held at once.
# Game evidence (v1.11.8): RWInput.PlayerInputLogic reads Rewired actions 0 Jump,
# 1 MoveHorizontal, 2 MoveVertical, 3 Take (pckp), 4 Throw, 11 Map, 34 Special
# (RWInput.cs:187-207). Pause (action 5 / Escape) is deliberately not a key.
KEY_LEFT = 1 << 0     # InputPackage.x = -1 (left + right held -> 0)
KEY_RIGHT = 1 << 1    # InputPackage.x = +1
KEY_UP = 1 << 2       # InputPackage.y = +1 (up + down held -> 0)
KEY_DOWN = 1 << 3     # InputPackage.y = -1; down + a side key sets downDiagonal (crawl/roll)
KEY_JUMP = 1 << 4     # jmp  (also "submit"/continue in dialogs)
KEY_GRAB = 1 << 5     # pckp (pick up / eat / interact)
KEY_THROW = 1 << 6    # thrw (also "cancel" in dialogs)
KEY_MAP = 1 << 7      # mp   (map; also restarts from the game-over prompt)
KEY_SPECIAL = 1 << 8  # spec (Watcher warp/camo, Saint ascension, Artificer pyro-jump)

# Bit order == index into a MultiBinary(NUM_KEYS) action vector.
KEY_NAMES: Tuple[str, ...] = (
    "left", "right", "up", "down", "jump", "grab", "throw", "map", "special",
)
NUM_KEYS = len(KEY_NAMES)
KEY_BITS: Dict[str, int] = {name: 1 << i for i, name in enumerate(KEY_NAMES)}
KEY_ALL_MASK = (1 << NUM_KEYS) - 1
assert KEY_BITS["special"] == KEY_SPECIAL and KEY_BITS["left"] == KEY_LEFT

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
GAME_FLAG_CYCLE_SURVIVED = 0x02  # edge: hibernated with enough food during this step (a starving sleep does not count)
GAME_FLAG_RAIN = 0x04            # level: the cycle timer has expired and the lethal rain is falling
GAME_FLAG_DIALOG_OPEN = 0x08     # level: an in-game text/dialog overlay awaits player input


# Whole-header struct. Field order matches the layout table above.
HEADER_STRUCT = struct.Struct(
    "<"
    "B"    # sync_flag
    "x"    # reserved (legacy action byte)
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
    "B"    # food_to_hibernate
    "B"    # malnourished
    "4s"   # region (ASCII, NUL-padded)
    "6x"   # reserved
)
assert HEADER_STRUCT.size == HEADER_SIZE, HEADER_STRUCT.size

_UINT32 = struct.Struct("<I")


def _encode_region(region: str) -> bytes:
    """ASCII, truncated to ``REGION_SIZE`` and NUL-padded (the mod's on-the-wire form)."""
    raw = (region or "").encode("ascii", errors = "replace")[:REGION_SIZE]
    return raw.ljust(REGION_SIZE, b"\0")


def _decode_region(raw: bytes) -> str:
    """Inverse of ``_encode_region``: strip NUL padding, decode ASCII."""
    return bytes(raw).split(b"\0", 1)[0].decode("ascii", errors = "replace")


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
    food_to_hibernate: int = 0
    malnourished: int = 0  # 0/1
    region: str = ""       # region acronym ("SU", "HI", ...), "" if unavailable

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
            self.food_to_hibernate & 0xFF,
            int(bool(self.malnourished)),
            _encode_region(self.region),
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ModState":
        """Parse a 64-byte header buffer."""
        fields = list(HEADER_STRUCT.unpack(data[:HEADER_SIZE]))
        fields[-1] = _decode_region(fields[-1])
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
            "food_to_hibernate": self.food_to_hibernate,
            "malnourished": bool(self.malnourished),
            "region": self.region,
        }


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def encode_keys(*names: str, **flags: bool) -> int:
    """
    Build an ``action_bits`` mask from key names.

    ``encode_keys("left", "jump")`` and ``encode_keys(left = True, jump = True)``
    both give ``KEY_LEFT | KEY_JUMP``. Unknown names raise ``KeyError``.
    """
    bits = 0
    for name in names:
        bits |= KEY_BITS[name]
    for name, pressed in flags.items():
        if name not in KEY_BITS:
            raise KeyError(f"unknown key {name!r}; known keys: {KEY_NAMES}")
        if pressed:
            bits |= KEY_BITS[name]
    return bits


def keys_to_bits(keys) -> int:
    """
    Convert a MultiBinary-style vector (length ``NUM_KEYS``, entries 0/1 or
    bool, index == ``KEY_NAMES`` order) to an ``action_bits`` mask.
    """
    arr = np.asarray(keys).reshape(-1)
    if arr.shape[0] != NUM_KEYS:
        raise ValueError(f"key vector must have length {NUM_KEYS} ({KEY_NAMES}), got shape {np.shape(keys)}")
    bits = 0
    for i in range(NUM_KEYS):
        if arr[i]:
            bits |= 1 << i
    return bits


def bits_to_keys(bits: int) -> np.ndarray:
    """Inverse of ``keys_to_bits``: an ``int8`` vector of length ``NUM_KEYS``."""
    bits = int(bits)
    return np.array([(bits >> i) & 1 for i in range(NUM_KEYS)], dtype = np.int8)


def action_to_bits(action) -> int:
    """
    Normalise an action the env accepts into an ``action_bits`` mask.

    * an integer (``int``, ``numpy`` integer, ``bool``) is taken as a bitmask;
    * anything array-like of length ``NUM_KEYS`` is taken as a key vector.

    Raises ``ValueError`` for out-of-range masks or wrong-length vectors.
    """
    if isinstance(action, (bool, np.bool_)):
        action = int(action)
    if isinstance(action, (int, np.integer)):
        bits = int(action)
        if not 0 <= bits <= KEY_ALL_MASK:
            raise ValueError(f"action bitmask {bits} out of range [0, {KEY_ALL_MASK}]")
        return bits
    if np.ndim(action) == 0:
        # 0-d array or other scalar
        return action_to_bits(int(np.asarray(action).item()))
    return keys_to_bits(action)


def pressed_key_names(bits: int) -> Tuple[str, ...]:
    """Names of the keys set in ``bits``, in ``KEY_NAMES`` order."""
    bits = int(bits)
    return tuple(name for i, name in enumerate(KEY_NAMES) if bits & (1 << i))


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
        frame, state = client.step(KEY_RIGHT | KEY_JUMP, ticks_per_step = 4)
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

    # While MOD_ALIVE is set the heartbeat may legitimately stall for several seconds:
    # right after launch the game's initial load runs synchronously on the main thread
    # (no Unity Update, so no heartbeat) although the mod's Awake has already run.
    alive_stall_grace: float = 15.0

    def wait_for_alive(
        self,
        timeout: float = 2.0,
        poll_interval: float = 0.01,
        alive_grace: Optional[float] = None,
    ) -> ModState:
        """
        Open the mapping (without setting CONNECTED) and wait until the mod's
        heartbeat advances.

        ``timeout`` is how long a static heartbeat is tolerated when ``MOD_ALIVE``
        is clear (nothing is running). When ``MOD_ALIVE`` is set the mod has
        started but may be inside the game's synchronous initial load, so the
        wait is extended to ``alive_grace`` seconds (default
        ``alive_stall_grace``, 15 s) before giving up.

        Returns the header snapshot that showed a live heartbeat.

        Raises:
            GameNotRunningError: if the heartbeat stayed static for the whole wait.
        """
        if alive_grace is None:
            alive_grace = self.alive_stall_grace
        self._open()
        first = self.read_state()
        start = time.monotonic()
        deadline = start + timeout
        extended = False
        while True:
            time.sleep(poll_interval)
            state = self.read_state()
            if state.heartbeat != first.heartbeat:
                return state
            now = time.monotonic()
            if now >= deadline:
                if not extended and state.mod_alive and alive_grace > timeout:
                    deadline = start + alive_grace
                    extended = True
                    logger.info(
                        "RainWorldRL mod is alive but its heartbeat is stalled (initial load?); "
                        "waiting up to %.0fs", alive_grace,
                    )
                    continue
                break
        first = state  # judge on the latest snapshot: MOD_ALIVE may have risen during the wait

        if not first.mod_alive:
            raise GameNotRunningError(
                "No running Rain World instance with the RainWorldRL mod was detected "
                "(heartbeat static, MOD_ALIVE clear)."
            )
        raise GameNotRunningError(
            f"The RainWorldRL mod reports MOD_ALIVE but its heartbeat did not advance in "
            f"{max(timeout, alive_grace if extended else 0.0):.1f}s; the game is hung or the mapping is stale."
        )

    def connect(
        self,
        wait_ready: bool = True,
        ready_timeout: float = 60.0,
        liveness_timeout: float = 2.0,
        alive_grace: Optional[float] = None,
    ) -> ModState:
        """
        Attach to a running game.

        1. Open the mapping and confirm the mod is alive (heartbeat advances;
           see ``wait_for_alive`` for the ``MOD_ALIVE`` stall grace).
        2. Write the requested frame dimensions and raise ``CONNECTED``.
        3. Optionally wait for ``READY``.

        Returns the last header snapshot read.

        Raises:
            GameNotRunningError: no live mod behind the mapping. The mapping is
                closed again before raising.
            ReadyTimeoutError: ``wait_ready`` was set and READY never rose.
        """
        try:
            state = self.wait_for_alive(timeout = liveness_timeout, alive_grace = alive_grace)
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

    def send_action(self, action, ticks_per_step: int = 1) -> None:
        """
        Write ``action_bits`` and signal ``ACTION_READY``.

        Args:
            action: Keys to hold for this step - an ``int`` bitmask of ``KEY_*``
                values, or a length-``NUM_KEYS`` 0/1 vector (``KEY_NAMES`` order,
                e.g. a ``MultiBinary`` sample). See ``action_to_bits``.
            ticks_per_step: Physics ticks to run before the frame is returned (1..255).
        """
        self.send_action_bits(action_to_bits(action), ticks_per_step)

    def send_action_bits(self, action_bits: int, ticks_per_step: int = 1) -> None:
        """Write a pre-encoded ``action_bits`` mask (offset 44) and signal ``ACTION_READY``."""
        self._require_connected()
        send_start = time.perf_counter()
        self._write_uint32(OFFSET_ACTION_BITS, int(action_bits) & KEY_ALL_MASK)
        self._write_byte(OFFSET_TICKS_PER_STEP, max(1, min(255, int(ticks_per_step))))
        self._write_byte(OFFSET_SYNC_FLAG, SYNC_ACTION_READY)
        self._send_action_time += time.perf_counter() - send_start

    def send_keys(self, *names: str, ticks_per_step: int = 1) -> None:
        """Convenience: ``send_keys("right", "jump")``."""
        self.send_action_bits(encode_keys(*names), ticks_per_step)

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

    def step(self, action = 0, ticks_per_step: int = 1, timeout: float = 10.0) -> Tuple[np.ndarray, ModState]:
        """Convenience: ``send_action`` (bitmask or key vector) followed by ``wait_for_frame``."""
        self.send_action(action, ticks_per_step)
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
