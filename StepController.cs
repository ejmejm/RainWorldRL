using System;
using BepInEx.Logging;
using UnityEngine;

/// <summary>
/// Controls the game simulation step-by-step, driven by the Python RL agent.
/// Runs physics at a high timescale during a step and pauses (timescale 0) between steps
/// while the game is READY; while not READY (menus, loading, respawn) time runs at 1x so the
/// game can get itself back into a playable state.
///
/// Step lifecycle (sync flag transitions):
///   Python writes action + ACTION_READY
///   -> ProcessUpdate consumes action once, writes PROCESSING, unpauses
///   -> ProcessFixedUpdate counts ticks, pauses when ticksPerStep reached
///   -> ProcessPostRender captures frame, writes game state + status, step_counter++, writes FRAME_READY last
///   -> Python reads frame, writes IDLE
///
/// Human override (<see cref="SetPaused"/>): no new actions are consumed, timescale is 1 and the
/// keyboard passes through. A step already in flight finishes normally (at 1x) so Python still
/// gets its frame; a pending ACTION_READY is serviced once override is cleared.
///
/// Steps are serviced even when the game is not READY: the current rendered frame is returned
/// with zeroed state fields so Python never hangs.
/// </summary>
public class StepController
{
    /// <summary>Rain World treats fpsCap > 120 as "unlimited" (InitializationScreen.cs:619, OptionsMenu.cs:860).</summary>
    private const int FPS_CAP_UNLIMITED = 121;

    private readonly SharedMemoryBridge sharedMemory;
    private readonly InputInjector inputInjector;
    private readonly FrameCapture frameCapture;
    private readonly ManualLogSource log;

    private bool enabled = false;
    private bool paused = false;
    private int currentTick = 0;
    private int ticksPerStep = 1;
    private bool stepInProgress = false;         // Action consumed, physics running
    private bool waitingForFrameCapture = false; // Physics done, awaiting post-render

    // Cached game instance (set via OnRainWorldUpdate)
    private RainWorld rainWorld;
    private bool stateErrorLogged = false;

    // Death edge detection
    private AbstractCreature trackedPlayer = null;
    private bool prevDead = false;

    // Original settings to restore on disable
    private float originalTimeScale;
    private float originalMaxDeltaTime;
    private int originalVSyncCount;
    private int originalTargetFrameRate;
    private int originalCaptureFramerate;
    private int originalFpsCap;
    private bool fpsCapApplied = false;

    /// <summary>Game-time speed multiplier while a step is running.</summary>
    public float SpeedMultiplier { get; set; } = 50f;

    /// <summary>
    /// When true (game READY) the simulation is frozen between steps; when false time runs at 1x
    /// between steps so menus/fades/loading progress even if Python is not stepping.
    /// </summary>
    public bool HoldWhenIdle { get; set; } = false;

    public bool VerboseLogging { get; set; } = false;

    public bool IsEnabled => enabled;
    public bool IsPaused => paused;
    public int CurrentTick => currentTick;
    public RainWorld RainWorld => rainWorld;

    public StepController(SharedMemoryBridge sharedMemory, InputInjector inputInjector, FrameCapture frameCapture, ManualLogSource log)
    {
        this.sharedMemory = sharedMemory;
        this.inputInjector = inputInjector;
        this.frameCapture = frameCapture;
        this.log = log;
    }

    /// <summary>Called from the plugin's RainWorld.Update hook every frame.</summary>
    public void OnRainWorldUpdate(RainWorld self)
    {
        rainWorld = self;

        if (!enabled || self.options == null)
            return;

        // Keep the game's own FPS cap out of the way. The game applies options.fpsCap to
        // Application.targetFrameRate from its menus; >120 means unlimited.
        if (!fpsCapApplied)
        {
            originalFpsCap = self.options.fpsCap;
            self.options.fpsCap = FPS_CAP_UNLIMITED;
            fpsCapApplied = true;
        }
        if (Application.targetFrameRate != -1)
            Application.targetFrameRate = -1;
    }

    /// <summary>Enables step-locked simulation mode.</summary>
    public void Enable()
    {
        if (enabled)
            return;

        originalTimeScale = Time.timeScale;
        originalMaxDeltaTime = Time.maximumDeltaTime;
        originalVSyncCount = QualitySettings.vSyncCount;
        originalTargetFrameRate = Application.targetFrameRate;
        originalCaptureFramerate = Time.captureFramerate;

        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;
        Time.maximumDeltaTime = float.MaxValue;
        Time.captureFramerate = 1000;
        Time.timeScale = 1f; // idle timescale is applied in ProcessUpdate

        fpsCapApplied = false;
        if (rainWorld != null && rainWorld.options != null)
        {
            originalFpsCap = rainWorld.options.fpsCap;
            rainWorld.options.fpsCap = FPS_CAP_UNLIMITED;
            fpsCapApplied = true;
        }

        currentTick = 0;
        stepInProgress = false;
        waitingForFrameCapture = false;
        paused = false;
        trackedPlayer = null;
        prevDead = false;

        enabled = true;
        log?.LogInfo("[StepController] Enabled");
    }

    /// <summary>Disables step-locked simulation and restores normal timing.</summary>
    public void Disable()
    {
        if (!enabled)
            return;

        Time.timeScale = originalTimeScale <= 0f ? 1f : originalTimeScale;
        Time.maximumDeltaTime = originalMaxDeltaTime;
        QualitySettings.vSyncCount = originalVSyncCount;
        Application.targetFrameRate = originalTargetFrameRate;
        Time.captureFramerate = originalCaptureFramerate;

        if (fpsCapApplied && rainWorld != null && rainWorld.options != null)
        {
            rainWorld.options.fpsCap = originalFpsCap;
            Application.targetFrameRate = originalFpsCap > 120 ? -1 : originalFpsCap;
        }
        fpsCapApplied = false;

        // If a step was mid-flight, leave the handshake consistent: Python is waiting for a frame.
        // Returning IDLE would make it hang, so signal FRAME_READY with whatever is in the buffer.
        if (stepInProgress || waitingForFrameCapture)
        {
            sharedMemory.ClearGameState();
            sharedMemory.SignalFrameReady();
        }
        stepInProgress = false;
        waitingForFrameCapture = false;
        paused = false;

        enabled = false;
        log?.LogInfo("[StepController] Disabled");
    }

    /// <summary>Human override: pause step servicing and run at real time.</summary>
    public void SetPaused(bool value)
    {
        if (!enabled || paused == value)
            return;

        paused = value;
        if (paused)
        {
            Time.timeScale = 1f;
            inputInjector.ClearInput();
        }
        log?.LogInfo($"[StepController] Step servicing {(paused ? "paused (human override)" : "resumed")}");
    }

    /// <summary>
    /// Called every Unity Update. Consumes a pending action (once per step) and starts physics running.
    /// Returns true if a new step was started.
    /// </summary>
    public bool ProcessUpdate()
    {
        if (!enabled)
            return false;

        // A step is already running or awaiting capture - nothing to consume
        if (stepInProgress || waitingForFrameCapture)
            return false;

        if (paused)
        {
            Time.timeScale = 1f;
            return false;
        }

        if (!sharedMemory.IsActionReady())
        {
            Time.timeScale = HoldWhenIdle ? 0f : 1f;
            return false;
        }

        // Consume the action exactly once
        byte action = sharedMemory.ReadAction();
        ticksPerStep = sharedMemory.ReadTicksPerStep();
        sharedMemory.SignalProcessing();

        inputInjector.SetFromActionByte(action);

        currentTick = 0;
        stepInProgress = true;
        Time.timeScale = SpeedMultiplier;

        return true;
    }

    /// <summary>Called in FixedUpdate to track physics ticks. Returns true when the step completes.</summary>
    public bool ProcessFixedUpdate()
    {
        if (!enabled || !stepInProgress || Time.timeScale == 0f)
            return false;

        currentTick++;

        bool stepComplete = currentTick >= ticksPerStep;
        if (stepComplete)
        {
            // Pause immediately when the step is complete (unless a human has taken over)
            Time.timeScale = paused ? 1f : 0f;
            stepInProgress = false;
            waitingForFrameCapture = true;
        }

        return stepComplete;
    }

    /// <summary>Called after rendering to capture the frame, write state and signal Python.</summary>
    public void ProcessPostRender()
    {
        if (!enabled || !waitingForFrameCapture)
            return;

        // Update frame dimensions if changed
        sharedMemory.ReadFrameDimensions();
        if (frameCapture.Width != sharedMemory.FrameWidth ||
            frameCapture.Height != sharedMemory.FrameHeight)
        {
            frameCapture.Resize(sharedMemory.FrameWidth, sharedMemory.FrameHeight);
        }

        // Capture and send frame
        byte[] frameData = frameCapture.CaptureFrameFlipped();
        sharedMemory.WriteFrameData(frameData);

        // Game state + death edge
        bool deathEdge = WriteGameState();
        sharedMemory.SetStatusFlag(SharedMemoryBridge.STATUS_PLAYER_DEAD, deathEdge);

        sharedMemory.IncrementStepCounter();

        // FRAME_READY must be the very last write
        sharedMemory.SignalFrameReady();

        // Reset for next step
        currentTick = 0;
        waitingForFrameCapture = false;
        if (!paused)
            Time.timeScale = HoldWhenIdle ? 0f : 1f;
    }

    /// <summary>
    /// Writes karma/food/position/room/cycle (zeros / -1 when unavailable) and returns the death
    /// edge: true only on the step where player 0 went alive -> dead. Tracking resets whenever a
    /// different player instance appears (new game / respawn), so the next death is detected again.
    /// </summary>
    private bool WriteGameState()
    {
        try
        {
            RainWorldGame game = rainWorld?.processManager?.currentMainLoop as RainWorldGame;
            StoryGameSession story = game?.GetStorySession;
            AbstractCreature abstractPlayer = GameFlowController.GetPlayer0(rainWorld?.processManager);

            if (game == null || story == null || abstractPlayer == null)
            {
                trackedPlayer = null;
                prevDead = false;
                sharedMemory.ClearGameState();
                return false;
            }

            // Death edge
            bool dead = IsDead(abstractPlayer);
            bool edge = false;
            if (!ReferenceEquals(abstractPlayer, trackedPlayer))
            {
                trackedPlayer = abstractPlayer;
                prevDead = false; // fresh player: a death during its very first step still counts
            }
            if (dead && !prevDead)
                edge = true;
            prevDead = dead;

            // Save-state fields
            int karma = 0, karmaCap = 0, cycle = -1;
            SaveState save = story.saveState;
            if (save != null)
            {
                cycle = save.cycleNumber;
                if (save.deathPersistentSaveData != null)
                {
                    karma = save.deathPersistentSaveData.karma;
                    karmaCap = save.deathPersistentSaveData.karmaCap;
                }
            }

            // Player fields
            int food = 0;
            float x = 0f, y = 0f;
            int roomIndex = -1;

            Player player = abstractPlayer.realizedCreature as Player;
            if (player != null)
            {
                food = player.FoodInStomach;
                if (player.mainBodyChunk != null)
                {
                    x = player.mainBodyChunk.pos.x;
                    y = player.mainBodyChunk.pos.y;
                }
                if (player.room != null && player.room.abstractRoom != null)
                    roomIndex = player.room.abstractRoom.index;
                else
                    roomIndex = abstractPlayer.pos.room;
            }
            else
            {
                if (abstractPlayer.state is PlayerState ps)
                    food = ps.foodInStomach;
                roomIndex = abstractPlayer.pos.room;
            }

            sharedMemory.WriteGameState(karma, karmaCap, food, x, y, roomIndex, cycle);
            return edge;
        }
        catch (Exception ex)
        {
            if (!stateErrorLogged)
            {
                log?.LogError($"[StepController] Game state read failed: {ex}");
                stateErrorLogged = true;
            }
            sharedMemory.ClearGameState();
            return false;
        }
    }

    private static bool IsDead(AbstractCreature abstractPlayer)
    {
        // The abstract state persists even when the creature is not realized
        if (abstractPlayer.state != null && abstractPlayer.state.dead)
            return true;

        Player player = abstractPlayer.realizedCreature as Player;
        return player != null && player.dead;
    }
}
