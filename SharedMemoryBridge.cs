using System;
using System.IO.MemoryMappedFiles;

/// <summary>
/// Manages shared memory communication between the Rain World mod and the Python RL client.
/// Implements protocol v3 (see docs/PROTOCOL.md). All multi-byte values are little-endian.
///
/// Header layout (64 bytes):
///   0   u8   sync_flag        0 IDLE, 1 ACTION_READY, 2 FRAME_READY, 3 PROCESSING
///   1   u8   reserved         (was the v2 action byte; v3 clients write action_bits at 44)
///   2   u8   ticks_per_step   py->mod (0 treated as 1)
///   3   u8   status           shared bitfield; each side only writes its own bits
///   4   u32  frame_width      py->mod
///   8   u32  frame_height     py->mod
///   12  u8   command          py->mod (0 NONE, 1 RESET, 2 KILL_PLAYER); mod clears when done
///   13  u8   command_result   mod->py (0 none, 1 OK, 2 ERROR)
///   14  u8   game_flags       mod->py (b0 IN_SHELTER, b1 CYCLE_SURVIVED edge, b2 RAIN, b3 DIALOG_OPEN)
///   15  u8   reserved
///   16  u32  heartbeat        mod->py, incremented every Unity Update
///   20  u32  step_counter     mod->py, incremented once per completed step
///   24  u8   karma
///   25  u8   karma_cap
///   26  u8   food
///   27  u8   food_max
///   28  f32  player_x
///   32  f32  player_y
///   36  i32  room_index       (-1 if unavailable)
///   40  i32  cycle_number     (-1 if unavailable)
///   44  u32  action_bits      py->mod raw-key bitfield, one bit per key (KEY_* below, PROTOCOL.md "Action bits")
///   48  f32  cycle_progress   mod->py fraction of the cycle elapsed (0..1, >1 once rain starts)
///   52  u8   food_to_hibernate mod->py pips needed to hibernate this cycle (= food_max while malnourished)
///   53  u8   malnourished     mod->py level: 1 while the save state is malnourished (last sleep was a starving one)
///   54  4B   region           mod->py ASCII region acronym of the active world (World.region.name, e.g. "SU"), NUL-padded; all NUL if unavailable
///   58  6B   reserved
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
    public const byte CMD_KILL_PLAYER = 2;
    public const byte RESULT_NONE = 0;
    public const byte RESULT_OK = 1;
    public const byte RESULT_ERROR = 2;

    // Header offsets
    private const int OFFSET_SYNC_FLAG = 0;
    // offset 1 is reserved (legacy v2 action byte, no longer read)
    private const int OFFSET_TICKS_PER_STEP = 2;
    private const int OFFSET_STATUS = 3;
    private const int OFFSET_WIDTH = 4;
    private const int OFFSET_HEIGHT = 8;
    private const int OFFSET_COMMAND = 12;
    private const int OFFSET_COMMAND_RESULT = 13;
    private const int OFFSET_GAME_FLAGS = 14;
    private const int OFFSET_HEARTBEAT = 16;
    private const int OFFSET_STEP_COUNTER = 20;
    private const int OFFSET_KARMA = 24;
    private const int OFFSET_KARMA_CAP = 25;
    private const int OFFSET_FOOD = 26;
    private const int OFFSET_FOOD_MAX = 27;
    private const int OFFSET_PLAYER_X = 28;
    private const int OFFSET_PLAYER_Y = 32;
    private const int OFFSET_ROOM_INDEX = 36;
    private const int OFFSET_CYCLE_NUMBER = 40;
    private const int OFFSET_ACTION_BITS = 44;
    private const int OFFSET_CYCLE_PROGRESS = 48;
    private const int OFFSET_FOOD_TO_HIBERNATE = 52;
    private const int OFFSET_MALNOURISHED = 53;
    private const int OFFSET_REGION = 54;
    public const int REGION_SIZE = 4;
    private const int OFFSET_FRAME_DATA = HEADER_SIZE;

    // action_bits (offset 44): one bit per player key. Must match rainworld_rl/shared_memory.py KEY_*
    // and the "Action bits" table in docs/PROTOCOL.md. Any combination may be set at once.
    // Game evidence (v1.11.8 decompile): RWInput.PlayerInputLogic reads Rewired actions
    // 0 Jump, 1 MoveHorizontal, 2 MoveVertical, 3 Take (pckp), 4 Throw, 11 Map, 34 Special
    // (RWInput.cs:187-207, RewiredConsts/Action.cs). Pause (action 5 / Escape) is deliberately
    // NOT a key: the agent must never pause or quit.
    public const uint KEY_LEFT = 1u << 0;    // MoveHorizontal negative -> InputPackage.x = -1
    public const uint KEY_RIGHT = 1u << 1;   // MoveHorizontal positive -> x = +1 (both held -> 0)
    public const uint KEY_UP = 1u << 2;      // MoveVertical positive   -> y = +1
    public const uint KEY_DOWN = 1u << 3;    // MoveVertical negative   -> y = -1 (both held -> 0)
    public const uint KEY_JUMP = 1u << 4;    // jmp  (UI: submit / "continue")
    public const uint KEY_GRAB = 1u << 5;    // pckp (pick up / eat / interact)
    public const uint KEY_THROW = 1u << 6;   // thrw (UI: cancel)
    public const uint KEY_MAP = 1u << 7;     // mp   (map; also "press to restart" on the game-over prompt)
    public const uint KEY_SPECIAL = 1u << 8; // spec (Watcher warp/camo, Saint ascension, Artificer pyro-jump)
    public const int KEY_COUNT = 9;
    public const uint KEY_ALL_MASK = (1u << KEY_COUNT) - 1;

    // Status bits (offset 3)
    public const byte STATUS_PLAYER_DEAD = 0x01;    // mod, edge-triggered
    public const byte STATUS_CONNECTED = 0x02;      // python
    public const byte STATUS_READY = 0x04;          // mod
    public const byte STATUS_HUMAN_OVERRIDE = 0x08; // mod
    public const byte STATUS_IN_GAME = 0x10;        // mod
    public const byte STATUS_MOD_ALIVE = 0x20;      // mod

    // game_flags bits (offset 14, mod-owned)
    public const byte GAME_FLAG_IN_SHELTER = 0x01;      // level
    public const byte GAME_FLAG_CYCLE_SURVIVED = 0x02;  // edge: hibernation succeeded this step
    public const byte GAME_FLAG_RAIN = 0x04;            // level
    public const byte GAME_FLAG_DIALOG_OPEN = 0x08;     // level: text/dialog overlay awaits input

    /// <summary>game_flags bits derived from per-step game state (cleared by <see cref="ClearGameState"/>).
    /// DIALOG_OPEN is owned by the dialog tracking code and is left alone.</summary>
    public const byte GAME_STATE_FLAGS_MASK = GAME_FLAG_IN_SHELTER | GAME_FLAG_CYCLE_SURVIVED | GAME_FLAG_RAIN;

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
        // the py->mod fields (action_bits, dims, command) are left untouched in case the
        // client attached first.
        WriteSyncFlag(SYNC_IDLE);
        UpdateStatusBits(MOD_OWNED_STATUS_MASK, 0);
        WriteCommandResult(RESULT_NONE);
        WriteGameFlags(0);
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

    /// <summary>Reads the raw-key action bitfield (offset 44), masked to the known keys.</summary>
    public uint ReadActionBits() => accessor.ReadUInt32(OFFSET_ACTION_BITS) & KEY_ALL_MASK;

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

    /// <summary>Writes the game_flags byte (offset 14). Mod-owned; write the whole byte.</summary>
    public void WriteGameFlags(byte flags) => accessor.Write(OFFSET_GAME_FLAGS, flags);

    public byte ReadGameFlags() => accessor.ReadByte(OFFSET_GAME_FLAGS);

    /// <summary>Sets or clears one game_flags bit (read-modify-write). Use this when
    /// several components own different bits of the byte.</summary>
    public void SetGameFlag(byte flag, bool value)
    {
        byte current = ReadGameFlags();
        byte next = value ? (byte)(current | flag) : (byte)(current & ~flag);
        if (next != current) WriteGameFlags(next);
    }

    /// <summary>Read-modify-write of several game_flags bits at once: the bits in
    /// <paramref name="mask"/> take the corresponding bits of <paramref name="value"/>.</summary>
    public void SetGameFlags(byte mask, byte value)
    {
        byte current = ReadGameFlags();
        byte next = (byte)((current & ~mask) | (value & mask));
        if (next != current) WriteGameFlags(next);
    }

    /// <summary>Writes the slugcat's maximum food pips (offset 27).</summary>
    public void WriteFoodMax(int foodMax) => accessor.Write(OFFSET_FOOD_MAX, ClampByte(foodMax));

    /// <summary>Writes the fraction of the rain cycle elapsed (offset 48).</summary>
    public void WriteCycleProgress(float progress) => accessor.Write(OFFSET_CYCLE_PROGRESS, progress);

    /// <summary>Writes the food pips needed to hibernate this cycle (offset 52).</summary>
    public void WriteFoodToHibernate(int pips) => accessor.Write(OFFSET_FOOD_TO_HIBERNATE, ClampByte(pips));

    /// <summary>Writes the malnourished level flag (offset 53).</summary>
    public void WriteMalnourished(bool malnourished) => accessor.Write(OFFSET_MALNOURISHED, malnourished ? (byte)1 : (byte)0);

    private readonly byte[] regionBuffer = new byte[REGION_SIZE];

    /// <summary>
    /// Writes the region acronym (offset 54, <see cref="REGION_SIZE"/> bytes): ASCII, truncated to
    /// the field size and NUL-padded. Null / empty writes all NULs ("unavailable").
    /// </summary>
    public void WriteRegion(string region)
    {
        Array.Clear(regionBuffer, 0, REGION_SIZE);
        if (!string.IsNullOrEmpty(region))
        {
            int n = Math.Min(region.Length, REGION_SIZE);
            for (int i = 0; i < n; i++)
            {
                char c = region[i];
                regionBuffer[i] = c < 128 ? (byte)c : (byte)'?';
            }
        }
        accessor.WriteArray(OFFSET_REGION, regionBuffer, 0, REGION_SIZE);
    }

    /// <summary>
    /// Writes the "unavailable" values for all game-state fields. Only the game-state bits of
    /// game_flags (<see cref="GAME_STATE_FLAGS_MASK"/>) are cleared; other bits keep their owner's value.
    /// </summary>
    public void ClearGameState()
    {
        WriteGameState(0, 0, 0, 0f, 0f, -1, -1);
        SetGameFlags(GAME_STATE_FLAGS_MASK, 0);
        WriteFoodMax(0);
        WriteCycleProgress(0f);
        WriteFoodToHibernate(0);
        WriteMalnourished(false);
        WriteRegion(null);
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
