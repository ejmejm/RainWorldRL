using System;
using UnityEngine;

/// <summary>
/// Controls the game simulation step-by-step, driven by Python RL agent.
/// Runs physics as fast as possible, pausing between steps to wait for Python.
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
    private bool waitingForFrameCapture = false;

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
    /// Hook to bypass Rain World's internal FPS cap.
    /// </summary>
    private void RainWorld_Update(On.RainWorld.orig_Update orig, RainWorld self)
    {
        if (enabled)
        {
            // Set FPS cap to unlimited (3 = unlimited in Rain World options)
            self.options.fpsCap = 3;
        }

        orig(self);
    }

    /// <summary>
    /// Called every Unity Update. Checks for actions and processes steps.
    /// </summary>
    public bool ProcessUpdate()
    {
        if (!enabled)
            return false;

        // Don't accept new action until previous frame is captured
        if (waitingForFrameCapture)
        {
            Time.timeScale = 0f;
            return false;
        }

        // Check if Python has sent an action
        if (!sharedMemory.IsActionReady())
        {
            Time.timeScale = 0f;
            return false;
        }

        // Action ready - run physics at high speed
        Time.timeScale = SPEED_MULTIPLIER;

        // Read action and ticks from shared memory
        byte action = sharedMemory.ReadAction();
        ticksPerStep = sharedMemory.ReadTicksPerStep();

        // Apply action to input system
        inputInjector.SetFromActionByte(action);

        return true;
    }

    /// <summary>
    /// Called in FixedUpdate to track physics ticks.
    /// </summary>
    public bool ProcessFixedUpdate()
    {
        if (!enabled || Time.timeScale == 0f)
            return false;

        currentTick++;

        bool stepComplete = currentTick >= ticksPerStep;

        // Pause immediately when step is complete
        if (stepComplete)
        {
            Time.timeScale = 0f;
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

        if (currentTick >= ticksPerStep)
        {
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
            sharedMemory.SignalFrameReady();

            // Reset for next step
            currentTick = 0;
            waitingForFrameCapture = false;

            inputInjector.UpdatePreviousState();
        }
    }

    /// <summary>
    /// Resets the tick counter and state.
    /// </summary>
    public void ResetTicks()
    {
        currentTick = 0;
        waitingForFrameCapture = false;
    }
}
