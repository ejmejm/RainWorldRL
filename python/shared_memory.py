"""
Shared memory client for communicating with the Rain World RL mod.

Memory Layout:
    Offset 0:  1 byte  - Sync flag (0=idle, 1=action_ready, 2=frame_ready, 3=processing)
                         Python: idle -> action_ready. Mod: action_ready -> processing -> frame_ready.
                         Python: frame_ready -> idle after reading the frame.
    Offset 1:  1 byte  - Action bitfield
    Offset 2:  1 byte  - Ticks per step
    Offset 3:  1 byte  - Status flags
    Offset 4:  4 bytes - Frame width (uint32)
    Offset 8:  4 bytes - Frame height (uint32)
    Offset 12: N bytes - RGB frame data (width * height * 3)
"""

import mmap
import struct
import time
from typing import Optional, Tuple

import numpy as np


# Shared memory constants
SHARED_MEMORY_NAME = "RainWorldRL"
HEADER_SIZE = 12
MAX_FRAME_SIZE = 1920 * 1080 * 3  # Support up to 1080p
TOTAL_SIZE = HEADER_SIZE + MAX_FRAME_SIZE

# Sync flag values
SYNC_IDLE = 0
SYNC_ACTION_READY = 1
SYNC_FRAME_READY = 2
SYNC_PROCESSING = 3

# Memory offsets
OFFSET_SYNC_FLAG = 0
OFFSET_ACTION = 1
OFFSET_TICKS_PER_STEP = 2
OFFSET_STATUS = 3
OFFSET_WIDTH = 4
OFFSET_HEIGHT = 8
OFFSET_FRAME_DATA = 12

# Action bitfield masks
ACTION_JUMP = 0x01
ACTION_GRAB = 0x02
ACTION_THROW = 0x04
ACTION_HORIZONTAL_SHIFT = 3  # bits 3-4
ACTION_VERTICAL_SHIFT = 5    # bits 5-6

# Status flags
STATUS_PLAYER_DEAD = 0x01
STATUS_CONNECTED = 0x02


class SharedMemoryClient:
    """Client for shared memory communication with Rain World RL mod."""

    def __init__(self, frame_width: int = 160, frame_height: int = 90, debug_timing: bool = False):
        """
        Initialize shared memory client.

        Args:
            frame_width: Desired frame width in pixels.
            frame_height: Desired frame height in pixels.
            debug_timing: If True, print timing debug info periodically.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_size = frame_width * frame_height * 3
        self.debug_timing = debug_timing

        self._shm: Optional[mmap.mmap] = None
        self._connected = False

        # Timing stats
        self._timing_samples = 0
        self._send_action_time = 0.0
        self._wait_poll_time = 0.0
        self._read_frame_time = 0.0
        self._reset_flag_time = 0.0
        self._last_timing_log = time.time()

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to shared memory created by the mod.

        Args:
            timeout: Maximum time to wait for connection in seconds.

        Returns:
            True if connected successfully, False otherwise.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # On Windows, use the tagname parameter
                self._shm = mmap.mmap(-1, TOTAL_SIZE, tagname = SHARED_MEMORY_NAME)
                self._connected = True

                # Write frame dimensions
                self._write_frame_dimensions()

                # Set connected status
                self._write_byte(OFFSET_STATUS, STATUS_CONNECTED)

                return True
            except Exception:
                time.sleep(0.1)

        return False

    def disconnect(self):
        """Disconnect from shared memory."""
        if self._shm is not None:
            try:
                # Clear connected status
                self._write_byte(OFFSET_STATUS, 0)
                self._shm.close()
            except Exception:
                pass
            self._shm = None
            self._connected = False

    def is_connected(self) -> bool:
        """Check if connected to shared memory."""
        return self._connected and self._shm is not None

    def _write_byte(self, offset: int, value: int):
        """Write a single byte to shared memory."""
        self._shm.seek(offset)
        self._shm.write(bytes([value & 0xFF]))

    def _read_byte(self, offset: int) -> int:
        """Read a single byte from shared memory."""
        self._shm.seek(offset)
        return self._shm.read(1)[0]

    def _write_uint32(self, offset: int, value: int):
        """Write a uint32 to shared memory (little-endian)."""
        self._shm.seek(offset)
        self._shm.write(struct.pack("<I", value))

    def _read_uint32(self, offset: int) -> int:
        """Read a uint32 from shared memory (little-endian)."""
        self._shm.seek(offset)
        data = self._shm.read(4)
        return struct.unpack("<I", data)[0]

    def _write_frame_dimensions(self):
        """Write frame dimensions to shared memory header."""
        self._write_uint32(OFFSET_WIDTH, self.frame_width)
        self._write_uint32(OFFSET_HEIGHT, self.frame_height)

    def set_frame_dimensions(self, width: int, height: int):
        """
        Update frame dimensions.

        Args:
            width: New frame width.
            height: New frame height.
        """
        self.frame_width = width
        self.frame_height = height
        self.frame_size = width * height * 3
        if self._connected:
            self._write_frame_dimensions()

    def send_action(
        self,
        jump: bool = False,
        grab: bool = False,
        throw: bool = False,
        horizontal: int = 0,
        vertical: int = 0,
        ticks_per_step: int = 1,
    ):
        """
        Send an action to the mod.

        Args:
            jump: Whether to press jump.
            grab: Whether to press grab/pickup.
            throw: Whether to press throw.
            horizontal: Horizontal direction (-1=left, 0=none, 1=right).
            vertical: Vertical direction (-1=down, 0=none, 1=up).
            ticks_per_step: Number of physics ticks to run before returning frame.
        """
        if not self._connected:
            raise RuntimeError("Not connected to shared memory")

        send_start = time.perf_counter()

        # Build action byte
        action = 0
        if jump:
            action |= ACTION_JUMP
        if grab:
            action |= ACTION_GRAB
        if throw:
            action |= ACTION_THROW

        # Horizontal: -1 -> 1, 0 -> 0, 1 -> 2
        h_val = 0 if horizontal == 0 else (1 if horizontal < 0 else 2)
        action |= (h_val << ACTION_HORIZONTAL_SHIFT)

        # Vertical: -1 -> 1, 0 -> 0, 1 -> 2
        v_val = 0 if vertical == 0 else (1 if vertical < 0 else 2)
        action |= (v_val << ACTION_VERTICAL_SHIFT)

        # Write action and ticks
        self._write_byte(OFFSET_ACTION, action)
        self._write_byte(OFFSET_TICKS_PER_STEP, max(1, min(255, ticks_per_step)))

        # Signal action ready
        self._write_byte(OFFSET_SYNC_FLAG, SYNC_ACTION_READY)

        send_end = time.perf_counter()
        self._send_action_time += send_end - send_start

    def send_action_discrete(self, action: int, ticks_per_step: int = 1):
        """
        Send a discrete action (0-17) to the mod.

        Action mapping:
            0: No-op
            1: Left
            2: Right
            3: Up
            4: Down
            5: Jump
            6: Grab
            7: Throw
            8: Left + Jump
            9: Right + Jump
            10: Up + Jump
            11: Down + Jump
            12: Left + Grab
            13: Right + Grab
            14: Up + Grab
            15: Down + Grab
            16: Left + Down (crawl left)
            17: Right + Down (crawl right)

        Args:
            action: Discrete action index.
            ticks_per_step: Number of physics ticks to run.
        """
        # Decode discrete action
        jump, grab, throw = False, False, False
        horizontal, vertical = 0, 0

        if action == 0:
            pass  # No-op
        elif action == 1:
            horizontal = -1
        elif action == 2:
            horizontal = 1
        elif action == 3:
            vertical = 1
        elif action == 4:
            vertical = -1
        elif action == 5:
            jump = True
        elif action == 6:
            grab = True
        elif action == 7:
            throw = True
        elif action == 8:
            horizontal, jump = -1, True
        elif action == 9:
            horizontal, jump = 1, True
        elif action == 10:
            vertical, jump = 1, True
        elif action == 11:
            vertical, jump = -1, True
        elif action == 12:
            horizontal, grab = -1, True
        elif action == 13:
            horizontal, grab = 1, True
        elif action == 14:
            vertical, grab = 1, True
        elif action == 15:
            vertical, grab = -1, True
        elif action == 16:
            horizontal, vertical = -1, -1
        elif action == 17:
            horizontal, vertical = 1, -1

        self.send_action(
            jump = jump,
            grab = grab,
            throw = throw,
            horizontal = horizontal,
            vertical = vertical,
            ticks_per_step = ticks_per_step,
        )

    def wait_for_frame(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """
        Wait for a frame from the mod.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Frame as numpy array (height, width, 3) or None if timeout.
        """
        if not self._connected:
            raise RuntimeError("Not connected to shared memory")

        start_time = time.time()
        poll_start = time.perf_counter()

        while time.time() - start_time < timeout:
            sync_flag = self._read_byte(OFFSET_SYNC_FLAG)
            if sync_flag == SYNC_FRAME_READY:
                poll_end = time.perf_counter()
                self._wait_poll_time += poll_end - poll_start

                # Read frame data
                read_start = time.perf_counter()
                self._shm.seek(OFFSET_FRAME_DATA)
                frame_data = self._shm.read(self.frame_size)

                # Convert to numpy array
                frame = np.frombuffer(frame_data, dtype = np.uint8)
                frame = frame.reshape((self.frame_height, self.frame_width, 3))
                read_end = time.perf_counter()
                self._read_frame_time += read_end - read_start

                # Reset sync flag to idle
                reset_start = time.perf_counter()
                self._write_byte(OFFSET_SYNC_FLAG, SYNC_IDLE)
                reset_end = time.perf_counter()
                self._reset_flag_time += reset_end - reset_start

                self._timing_samples += 1

                # Log timing every 2 seconds
                if self.debug_timing and time.time() - self._last_timing_log >= 2.0:
                    self._log_timing()

                return frame

            # Flag is ACTION_READY (mod hasn't picked it up yet) or PROCESSING
            # (step running) - keep waiting. Small sleep to avoid busy-waiting.
            time.sleep(0.0001)

        return None

    def _log_timing(self):
        """Log timing statistics."""
        if self._timing_samples == 0:
            return

        n = self._timing_samples
        print(f"[SharedMemory] TIMING (avg over {n} steps):")
        print(f"  SendAction:   {self._send_action_time * 1000 / n:.3f}ms")
        print(f"  WaitPoll:     {self._wait_poll_time * 1000 / n:.3f}ms")
        print(f"  ReadFrame:    {self._read_frame_time * 1000 / n:.3f}ms")
        print(f"  ResetFlag:    {self._reset_flag_time * 1000 / n:.3f}ms")
        total = (self._send_action_time + self._wait_poll_time +
                 self._read_frame_time + self._reset_flag_time)
        print(f"  TOTAL:        {total * 1000 / n:.3f}ms (max {n / total:.1f} FPS)")

        # Reset stats
        self._timing_samples = 0
        self._send_action_time = 0.0
        self._wait_poll_time = 0.0
        self._read_frame_time = 0.0
        self._reset_flag_time = 0.0
        self._last_timing_log = time.time()

    def get_status(self) -> Tuple[bool, bool]:
        """
        Get current status flags.

        Returns:
            Tuple of (player_dead, connected).
        """
        if not self._connected:
            return False, False

        status = self._read_byte(OFFSET_STATUS)
        player_dead = bool(status & STATUS_PLAYER_DEAD)
        connected = bool(status & STATUS_CONNECTED)
        return player_dead, connected

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False

