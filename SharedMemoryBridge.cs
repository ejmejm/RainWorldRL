using System;
using System.IO.MemoryMappedFiles;

/// <summary>
/// Manages shared memory communication between the Rain World mod and the Python RL client.
/// Implements protocol v2 (see docs/PROTOCOL.md). All multi-byte values are little-endian.
///
/// Header layout (64 bytes):
///   0   u8   sync_flag        0 IDLE, 1 ACTION_READY, 2 FRAME_READY, 3 PROCESSING
///   1   u8   action           py->mod bitfield
///   2   u8   ticks_per_step   py->mod (0 treated as 1)
///   3   u8   status           shared bitfield; each side only writes its own bits
///   4   u32  frame_width      py->mod
///   8   u32  frame_height     py->mod
///   12  u8   command          py->mod (0 NONE, 1 RESET); mod clears when done
///   13  u8   command_result   mod->py (0 none, 1 OK, 2 ERROR)
///   14  u16  reserved
///   16  u32  heartbeat        mod->py, incremented every Unity Update
///   20  u32  step_counter     mod->py, incremented once per completed step
///   24  u8   karma
///   25  u8   karma_cap
///   26  u8   food
///   27  u8   reserved
///   28  f32  player_x
///   32  f32  player_y
///   36  i32  room_index       (-1 if unavailable)
///   40  i32  cycle_number     (-1 if unavailable)
///   44  20B  reserved
///   64  N    frame            RGB24, top row first
/// </summary>
public class SharedMemoryBridge : IDisposable
{
    public const string SHARED_MEMORY_NAME = "RainWorldRL";
    public const int HEADER_SIZE = 64;
    public const int MAX_FRAME_WIDTH = 1920;
    public const int MAX_FRAME_HEIGHT = 1080;
    public const int MAX_FRAME_SIZE = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * 3;
    public const int DEFAULT_FRAME_WIDTH = 160;
    public const int DEFAULT_FRAME_HEIGHT = 90;

    // Sync flag values
    public const byte SYNC_IDLE = 0;
    public const byte SYNC_ACTION_READY = 1;
    public const byte SYNC_FRAME_READY = 2;
    public const byte SYNC_PROCESSING = 3;

    // Commands (offset 12) and results (offset 13)
    public const byte CMD_NONE = 0;
    public const byte CMD_RESET = 1;
    public const byte RESULT_NONE = 0;
    public const byte RESULT_OK = 1;
    public const byte RESULT_ERROR = 2;

    // Header offsets
    private const int OFFSET_SYNC_FLAG = 0;
    private const int OFFSET_ACTION = 1;
    private const int OFFSET_TICKS_PER_STEP = 2;
    private const int OFFSET_STATUS = 3;
    private const int OFFSET_WIDTH = 4;
    private const int OFFSET_HEIGHT = 8;
    private const int OFFSET_COMMAND = 12;
    private const int OFFSET_COMMAND_RESULT = 13;
    private const int OFFSET_HEARTBEAT = 16;
    private const int OFFSET_STEP_COUNTER = 20;
    private const int OFFSET_KARMA = 24;
    private const int OFFSET_KARMA_CAP = 25;
    private const int OFFSET_FOOD = 26;
    private const int OFFSET_PLAYER_X = 28;
    private const int OFFSET_PLAYER_Y = 32;
    private const int OFFSET_ROOM_INDEX = 36;
    private const int OFFSET_CYCLE_NUMBER = 40;
    private const int OFFSET_FRAME_DATA = HEADER_SIZE;

    // Action bitfield masks
    public const byte ACTION_JUMP = 0x01;
    public const byte ACTION_GRAB = 0x02;
    public const byte ACTION_THROW = 0x04;
    public const byte ACTION_HORIZONTAL_MASK = 0x18; // bits 3-4
    public const byte ACTION_VERTICAL_MASK = 0x60;   // bits 5-6

    // Status bits (offset 3)
    public const byte STATUS_PLAYER_DEAD = 0x01;    // mod, edge-triggered
    public const byte STATUS_CONNECTED = 0x02;      // python
    public const byte STATUS_READY = 0x04;          // mod
    public const byte STATUS_HUMAN_OVERRIDE = 0x08; // mod
    public const byte STATUS_IN_GAME = 0x10;        // mod
    public const byte STATUS_MOD_ALIVE = 0x20;      // mod

    /// <summary>Bits the mod is allowed to write. Everything else belongs to Python.</summary>
    public const byte MOD_OWNED_STATUS_MASK =
        STATUS_PLAYER_DEAD | STATUS_READY | STATUS_HUMAN_OVERRIDE | STATUS_IN_GAME | STATUS_MOD_ALIVE;

    private MemoryMappedFile mmf;
    private MemoryMappedViewAccessor accessor;
    private bool disposed = false;

    public int FrameWidth { get; private set; } = DEFAULT_FRAME_WIDTH;
    public int FrameHeight { get; private set; } = DEFAULT_FRAME_HEIGHT;

    /// <summary>True while Python holds the CONNECTED bit.</summary>
    public bool IsConnected => (ReadStatus() & STATUS_CONNECTED) != 0;

    public SharedMemoryBridge()
    {
        int totalSize = HEADER_SIZE + MAX_FRAME_SIZE;
        mmf = MemoryMappedFile.CreateOrOpen(SHARED_MEMORY_NAME, totalSize);
        accessor = mmf.CreateViewAccessor();

        // Reset the handshake and our own status bits. Python's CONNECTED bit and
        // the py->mod fields (action, dims, command) are left untouched in case the
        // client attached first.
        WriteSyncFlag(SYNC_IDLE);
        UpdateStatusBits(MOD_OWNED_STATUS_MASK, 0);
        WriteCommandResult(RESULT_NONE);
        ClearGameState();
    }

    // ----- sync flag -----

    public byte ReadSyncFlag() => accessor.ReadByte(OFFSET_SYNC_FLAG);

    public void WriteSyncFlag(byte value) => accessor.Write(OFFSET_SYNC_FLAG, value);

    public bool IsActionReady() => ReadSyncFlag() == SYNC_ACTION_READY;

    /// <summary>Marks the pending action as consumed so it is not read twice.</summary>
    public void SignalProcessing() => WriteSyncFlag(SYNC_PROCESSING);

    /// <summary>Signals that the frame and all header fields are ready. Must be written last.</summary>
    public void SignalFrameReady() => WriteSyncFlag(SYNC_FRAME_READY);

    // ----- py -> mod inputs -----

    public byte ReadAction() => accessor.ReadByte(OFFSET_ACTION);

    public byte ReadTicksPerStep()
    {
        byte ticks = accessor.ReadByte(OFFSET_TICKS_PER_STEP);
        return ticks == 0 ? (byte)1 : ticks;
    }

    public byte ReadCommand() => accessor.ReadByte(OFFSET_COMMAND);

    public void WriteCommand(byte command) => accessor.Write(OFFSET_COMMAND, command);

    public void WriteCommandResult(byte result) => accessor.Write(OFFSET_COMMAND_RESULT, result);

    /// <summary>
    /// Reads the requested frame dimensions, clamping to the supported range and
    /// falling back to the defaults for zero/garbage values.
    /// </summary>
    public void ReadFrameDimensions()
    {
        int w = accessor.ReadInt32(OFFSET_WIDTH);
        int h = accessor.ReadInt32(OFFSET_HEIGHT);

        FrameWidth = (w <= 0 || w > MAX_FRAME_WIDTH) ? DEFAULT_FRAME_WIDTH : w;
        FrameHeight = (h <= 0 || h > MAX_FRAME_HEIGHT) ? DEFAULT_FRAME_HEIGHT : h;
    }

    // ----- status -----

    public byte ReadStatus() => accessor.ReadByte(OFFSET_STATUS);

    /// <summary>
    /// Read-modify-write of mod-owned status bits only. Bits in <paramref name="mask"/>
    /// are replaced by the corresponding bits of <paramref name="value"/>; all other bits
    /// (including Python's CONNECTED bit) are preserved. Bits outside the mod-owned set
    /// are ignored even if present in the mask.
    /// </summary>
    public void UpdateStatusBits(byte mask, byte value)
    {
        mask &= MOD_OWNED_STATUS_MASK;
        if (mask == 0)
            return;

        byte current = ReadStatus();
        byte updated = (byte)((current & ~mask) | (value & mask));
        if (updated != current)
            accessor.Write(OFFSET_STATUS, updated);
    }

    /// <summary>Sets or clears a single mod-owned status flag.</summary>
    public void SetStatusFlag(byte flag, bool value)
    {
        UpdateStatusBits(flag, value ? flag : (byte)0);
    }

    // ----- counters -----

    public uint ReadHeartbeat() => accessor.ReadUInt32(OFFSET_HEARTBEAT);

    public void IncrementHeartbeat()
    {
        accessor.Write(OFFSET_HEARTBEAT, unchecked(accessor.ReadUInt32(OFFSET_HEARTBEAT) + 1u));
    }

    public uint ReadStepCounter() => accessor.ReadUInt32(OFFSET_STEP_COUNTER);

    public void IncrementStepCounter()
    {
        accessor.Write(OFFSET_STEP_COUNTER, unchecked(accessor.ReadUInt32(OFFSET_STEP_COUNTER) + 1u));
    }

    // ----- game state -----

    /// <summary>Writes the per-step game-state fields (offsets 24..43).</summary>
    public void WriteGameState(int karma, int karmaCap, int food, float playerX, float playerY, int roomIndex, int cycleNumber)
    {
        accessor.Write(OFFSET_KARMA, ClampByte(karma));
        accessor.Write(OFFSET_KARMA_CAP, ClampByte(karmaCap));
        accessor.Write(OFFSET_FOOD, ClampByte(food));
        accessor.Write(OFFSET_PLAYER_X, playerX);
        accessor.Write(OFFSET_PLAYER_Y, playerY);
        accessor.Write(OFFSET_ROOM_INDEX, roomIndex);
        accessor.Write(OFFSET_CYCLE_NUMBER, cycleNumber);
    }

    /// <summary>Writes the "unavailable" values for all game-state fields.</summary>
    public void ClearGameState()
    {
        WriteGameState(0, 0, 0, 0f, 0f, -1, -1);
    }

    private static byte ClampByte(int v)
    {
        if (v < 0) return 0;
        if (v > 255) return 255;
        return (byte)v;
    }

    // ----- frame -----

    /// <summary>Writes RGB24 frame data; size is bounded by the current frame dimensions.</summary>
    public void WriteFrameData(byte[] frameData)
    {
        if (frameData == null || frameData.Length == 0)
            return;

        int expectedSize = FrameWidth * FrameHeight * 3;
        int writeSize = Math.Min(frameData.Length, Math.Min(expectedSize, MAX_FRAME_SIZE));

        accessor.WriteArray(OFFSET_FRAME_DATA, frameData, 0, writeSize);
    }

    public void Dispose()
    {
        if (!disposed)
        {
            accessor?.Dispose();
            mmf?.Dispose();
            disposed = true;
        }
    }
}
