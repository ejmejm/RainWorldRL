using System;
using UnityEngine;

/// <summary>
/// Controls the game simulation step-by-step, driven by Python RL agent.
/// Runs physics as fast as possible, pausing between steps to wait for Python.
///
/// Step lifecycle (sync flag transitions):
///   Python writes action + ACTION_READY
///   -> ProcessUpdate consumes action once, writes PROCESSING, unpauses
///   -> ProcessFixedUpdate counts ticks, pauses when ticksPerStep reached
///   -> ProcessPostRender captures frame, writes status, writes FRAME_READY
///   -> Python reads frame, writes IDLE
/// </summary>
public class StepController
{
    // Speed multiplier when running physics - higher = faster
    // 50x means physics runs 50x faster than real-time
    private const float SPEED_MULTIPLIER = 50f;

    private SharedMemoryBridge sharedMemory;
    private InputInjector inputInjector;
    private FrameCapture frameCapture;

    private bool enabled = false;
    private int currentTick = 0;
    private int ticksPerStep = 1;
    private bool stepInProgress = false;      // Action consumed, physics running
    private bool waitingForFrameCapture = false; // Physics done, awaiting post-render

    // Cached game instance (set by the RainWorld.Update hook)
    private RainWorld rainWorld;
    private bool deathCheckErrorLogged = false;

    // Original settings to restore on disable
    private float originalTimeScale;
    private float originalMaxDeltaTime;
    private int originalVSyncCount;
    private int originalTargetFrameRate;
    private int originalCaptureFramerate;
    private bool hooksInstalled = false;

    public bool IsEnabled => enabled;
    public int CurrentTick => currentTick;

    public StepController(SharedMemoryBridge sharedMemory, InputInjector inputInjector, FrameCapture frameCapture)
    {
        this.sharedMemory = sharedMemory;
        this.inputInjector = inputInjector;
        this.frameCapture = frameCapture;
    }

    /// <summary>
    /// Enables step-locked simulation mode.
    /// </summary>
    public void Enable()
    {
        if (enabled)
            return;

        // Store original settings
        originalTimeScale = Time.timeScale;
        originalMaxDeltaTime = Time.maximumDeltaTime;
        originalVSyncCount = QualitySettings.vSyncCount;
        originalTargetFrameRate = Application.targetFrameRate;
        originalCaptureFramerate = Time.captureFramerate;

        // Configure for maximum speed
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;
        Time.maximumDeltaTime = float.MaxValue;
        Time.captureFramerate = 1000;
        Time.timeScale = 0f; // Start paused

        // Install hook to bypass Rain World's FPS cap
        InstallSpeedHooks();

        currentTick = 0;
        stepInProgress = false;
        waitingForFrameCapture = false;

        enabled = true;
        Debug.Log("[StepController] RL mode enabled - waiting for Python");
    }

    /// <summary>
    /// Disables step-locked simulation and restores normal timing.
    /// </summary>
    public void Disable()
    {
        if (!enabled)
            return;

        UninstallSpeedHooks();

        // Restore original settings
        Time.timeScale = originalTimeScale;
        Time.maximumDeltaTime = originalMaxDeltaTime;
        QualitySettings.vSyncCount = originalVSyncCount;
        Application.targetFrameRate = originalTargetFrameRate;
        Time.captureFramerate = originalCaptureFramerate;

        enabled = false;
        Debug.Log("[StepController] RL mode disabled");
    }

    private void InstallSpeedHooks()
    {
        if (hooksInstalled)
            return;

        On.RainWorld.Update += RainWorld_Update;
        hooksInstalled = true;
    }

    private void UninstallSpeedHooks()
    {
        if (!hooksInstalled)
            return;

        On.RainWorld.Update -= RainWorld_Update;
        hooksInstalled = false;
    }

    /// <summary>
    /// Hook to bypass Rain World's internal FPS cap and cache the game instance.
    /// </summary>
    private void RainWorld_Update(On.RainWorld.orig_Update orig, RainWorld self)
    {
        rainWorld = self;

        if (enabled)
        {
            // Set FPS cap to unlimited (3 = unlimited in Rain World options)
            self.options.fpsCap = 3;
        }

        orig(self);
    }

    /// <summary>
    /// Called every Unity Update. Consumes a pending action (once per step) and
    /// starts physics running.
    /// </summary>
    public bool ProcessUpdate()
    {
        if (!enabled)
            return false;

        // A step is already running or awaiting capture - nothing to consume
        if (stepInProgress || waitingForFrameCapture)
            return false;

        // Check if Python has sent an action
        if (!sharedMemory.IsActionReady())
        {
            Time.timeScale = 0f;
            return false;
        }

        // Consume the action exactly once
        byte action = sharedMemory.ReadAction();
        ticksPerStep = sharedMemory.ReadTicksPerStep();
        sharedMemory.SignalProcessing();

        inputInjector.SetFromActionByte(action);

        currentTick = 0;
        stepInProgress = true;
        Time.timeScale = SPEED_MULTIPLIER;

        return true;
    }

    /// <summary>
    /// Called in FixedUpdate to track physics ticks.
    /// </summary>
    public bool ProcessFixedUpdate()
    {
        if (!enabled || !stepInProgress || Time.timeScale == 0f)
            return false;

        currentTick++;

        bool stepComplete = currentTick >= ticksPerStep;

        // Pause immediately when step is complete
        if (stepComplete)
        {
            Time.timeScale = 0f;
            stepInProgress = false;
            waitingForFrameCapture = true;
        }

        return stepComplete;
    }

    /// <summary>
    /// Called after rendering to capture the frame and signal Python.
    /// </summary>
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

        // Status flags
        sharedMemory.SetStatusFlag(SharedMemoryBridge.STATUS_PLAYER_DEAD, IsPlayerDead());

        sharedMemory.SignalFrameReady();

        // Reset for next step
        currentTick = 0;
        waitingForFrameCapture = false;
    }

    /// <summary>
    /// Returns true if player 0 is dead. Returns false when no game is running
    /// (e.g. in a menu) or the player is not present.
    /// </summary>
    private bool IsPlayerDead()
    {
        try
        {
            RainWorldGame game = rainWorld?.processManager?.currentMainLoop as RainWorldGame;
            if (game == null || game.Players == null || game.Players.Count == 0)
                return false;

            AbstractCreature abstractPlayer = game.Players[0];
            if (abstractPlayer == null)
                return false;

            // The abstract state persists even when the creature is not realized
            if (abstractPlayer.state != null && abstractPlayer.state.dead)
                return true;

            Player player = abstractPlayer.realizedCreature as Player;
            return player != null && player.dead;
        }
        catch (Exception ex)
        {
            if (!deathCheckErrorLogged)
            {
                Debug.LogError($"[StepController] Death check failed: {ex.Message}");
                deathCheckErrorLogged = true;
            }
            return false;
        }
    }
}
