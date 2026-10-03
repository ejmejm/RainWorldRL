using System;
using System.IO.MemoryMappedFiles;
using System.Threading;

/// <summary>
/// Manages shared memory communication between the Rain World mod and Python RL client.
/// 
/// Memory Layout:
/// Offset 0:  1 byte  - Sync flag (0=idle, 1=action_ready, 2=frame_ready, 3=processing)
///                      Python: idle -> action_ready. Mod: action_ready -> processing -> frame_ready.
///                      Python: frame_ready -> idle after reading the frame.
/// Offset 1:  1 byte  - Action bitfield (bits: 0=jump, 1=grab, 2=throw, 3-4=horizontal, 5-6=vertical)
/// Offset 2:  1 byte  - Ticks per step
/// Offset 3:  1 byte  - Status flags (bit 0=player_dead, bit 1=connected)
/// Offset 4:  4 bytes - Frame width (uint32)
/// Offset 8:  4 bytes - Frame height (uint32)
/// Offset 12: N bytes - RGB frame data (width * height * 3)
/// </summary>
public class SharedMemoryBridge : IDisposable
{
    public const string SHARED_MEMORY_NAME = "RainWorldRL";
    public const int HEADER_SIZE = 12;
    public const int MAX_FRAME_SIZE = 1920 * 1080 * 3; // Support up to 1080p

    // Sync flag values
    public const byte SYNC_IDLE = 0;
    public const byte SYNC_ACTION_READY = 1;
    public const byte SYNC_FRAME_READY = 2;
    public const byte SYNC_PROCESSING = 3;

    // Memory offsets
    private const int OFFSET_SYNC_FLAG = 0;
    private const int OFFSET_ACTION = 1;
    private const int OFFSET_TICKS_PER_STEP = 2;
    private const int OFFSET_STATUS = 3;
    private const int OFFSET_WIDTH = 4;
    private const int OFFSET_HEIGHT = 8;
    private const int OFFSET_FRAME_DATA = 12;

    // Action bitfield masks
    public const byte ACTION_JUMP = 0x01;
    public const byte ACTION_GRAB = 0x02;
    public const byte ACTION_THROW = 0x04;
    public const byte ACTION_HORIZONTAL_MASK = 0x18; // bits 3-4
    public const byte ACTION_VERTICAL_MASK = 0x60;   // bits 5-6

    // Status flags
    public const byte STATUS_PLAYER_DEAD = 0x01;
    public const byte STATUS_CONNECTED = 0x02;

    private MemoryMappedFile mmf;
    private MemoryMappedViewAccessor accessor;
    private bool disposed = false;

    public int FrameWidth { get; private set; }
    public int FrameHeight { get; private set; }
    public bool IsConnected => (ReadStatus() & STATUS_CONNECTED) != 0;

    public SharedMemoryBridge()
    {
        int totalSize = HEADER_SIZE + MAX_FRAME_SIZE;
        mmf = MemoryMappedFile.CreateOrOpen(SHARED_MEMORY_NAME, totalSize);
        accessor = mmf.CreateViewAccessor();

        // Initialize to idle state
        WriteSyncFlag(SYNC_IDLE);
    }

    /// <summary>
    /// Reads the current sync flag value.
    /// </summary>
    public byte ReadSyncFlag()
    {
        return accessor.ReadByte(OFFSET_SYNC_FLAG);
    }

    /// <summary>
    /// Writes a sync flag value.
    /// </summary>
    public void WriteSyncFlag(byte value)
    {
        accessor.Write(OFFSET_SYNC_FLAG, value);
    }

    /// <summary>
    /// Reads the action byte from shared memory.
    /// </summary>
    public byte ReadAction()
    {
        return accessor.ReadByte(OFFSET_ACTION);
    }

    /// <summary>
    /// Reads the ticks per step value.
    /// </summary>
    public byte ReadTicksPerStep()
    {
        byte ticks = accessor.ReadByte(OFFSET_TICKS_PER_STEP);
        return ticks == 0 ? (byte)1 : ticks; // Default to 1 if not set
    }

    /// <summary>
    /// Reads the status flags.
    /// </summary>
    public byte ReadStatus()
    {
        return accessor.ReadByte(OFFSET_STATUS);
    }

    /// <summary>
    /// Writes status flags.
    /// </summary>
    public void WriteStatus(byte status)
    {
        accessor.Write(OFFSET_STATUS, status);
    }

    /// <summary>
    /// Sets a specific status flag bit.
    /// </summary>
    public void SetStatusFlag(byte flag, bool value)
    {
        byte current = ReadStatus();
        if (value)
            current |= flag;
        else
            current &= (byte)~flag;
        WriteStatus(current);
    }

    /// <summary>
    /// Reads frame dimensions from shared memory. Call this to update FrameWidth/FrameHeight.
    /// </summary>
    public void ReadFrameDimensions()
    {
        FrameWidth = accessor.ReadInt32(OFFSET_WIDTH);
        FrameHeight = accessor.ReadInt32(OFFSET_HEIGHT);

        // Clamp to valid range
        if (FrameWidth <= 0 || FrameWidth > 1920)
            FrameWidth = 160;
        if (FrameHeight <= 0 || FrameHeight > 1080)
            FrameHeight = 90;
    }

    /// <summary>
    /// Writes frame data to shared memory.
    /// </summary>
    public void WriteFrameData(byte[] frameData)
    {
        if (frameData == null || frameData.Length == 0)
            return;

        int expectedSize = FrameWidth * FrameHeight * 3;
        int writeSize = Math.Min(frameData.Length, expectedSize);

        accessor.WriteArray(OFFSET_FRAME_DATA, frameData, 0, writeSize);
    }

    /// <summary>
    /// Waits for the sync flag to become a specific value.
    /// </summary>
    public bool WaitForSyncFlag(byte expectedValue, int timeoutMs = -1)
    {
        int elapsed = 0;
        while (ReadSyncFlag() != expectedValue)
        {
            Thread.Sleep(0); // Yield to other threads
            if (timeoutMs > 0)
            {
                elapsed++;
                if (elapsed > timeoutMs)
                    return false;
            }
        }
        return true;
    }

    /// <summary>
    /// Checks if an action is ready to be processed.
    /// </summary>
    public bool IsActionReady()
    {
        return ReadSyncFlag() == SYNC_ACTION_READY;
    }

    /// <summary>
    /// Signals that the pending action has been consumed and a step is running.
    /// Prevents the same action from being read again on subsequent Updates.
    /// </summary>
    public void SignalProcessing()
    {
        WriteSyncFlag(SYNC_PROCESSING);
    }

    /// <summary>
    /// Signals that the frame is ready to be read by Python.
    /// </summary>
    public void SignalFrameReady()
    {
        WriteSyncFlag(SYNC_FRAME_READY);
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

