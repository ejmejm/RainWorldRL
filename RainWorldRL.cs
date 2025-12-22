using BepInEx;
using System;
using System.Security.Permissions;
using UnityEngine;

#pragma warning disable CS0618 // SecurityAction.RequestMinimum is obsolete. However, this does not apply to the mod, which still needs it. Suppress the warning indicating that it is obsolete.
[assembly: SecurityPermission(SecurityAction.RequestMinimum, SkipVerification = true)]
#pragma warning restore CS0618

/// <summary>
/// Rain World RL - Turns Rain World into a reinforcement learning environment.
/// Communicates with Python via shared memory for step-driven simulation.
/// </summary>
[BepInPlugin("rainworld.rl", "RainWorldRL", "0.1")]
public class RainWorldRL : BaseUnityPlugin
{
    // Core components
    private SharedMemoryBridge sharedMemory;
    private InputInjector inputInjector;
    private FrameCapture frameCapture;
    private StepController stepController;

    // State
    private bool initialized = false;
    private bool rlModeEnabled = false;

    // Default frame dimensions (will be updated from shared memory)
    private const int DEFAULT_FRAME_WIDTH = 160;
    private const int DEFAULT_FRAME_HEIGHT = 90;

    // Keybind for toggling RL mode
    private const KeyCode TOGGLE_KEY = KeyCode.F10;

    void Awake()
    {
        Logger.LogInfo("RainWorld RL initializing...");

        try
        {
            // Initialize shared memory bridge
            sharedMemory = new SharedMemoryBridge();
            Logger.LogInfo("Shared memory bridge created");

            // Initialize input injector
            inputInjector = new InputInjector();
            Logger.LogInfo("Input injector created");

            // Initialize frame capture with default dimensions
            frameCapture = new FrameCapture(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT);
            Logger.LogInfo($"Frame capture initialized ({DEFAULT_FRAME_WIDTH}x{DEFAULT_FRAME_HEIGHT})");

            // Initialize step controller
            stepController = new StepController(sharedMemory, inputInjector, frameCapture);
            Logger.LogInfo("Step controller created");

            initialized = true;
            Logger.LogInfo("RainWorld RL initialized successfully");
        }
        catch (Exception ex)
        {
            Logger.LogError($"Failed to initialize RainWorld RL: {ex.Message}");
            Logger.LogError(ex.StackTrace);
        }
    }

    void OnEnable()
    {
        if (!initialized)
            return;

        // Install input hooks
        inputInjector.Install();
        Logger.LogInfo("Input hooks installed");

        // Subscribe to camera post-render event
        Camera.onPostRender += OnCameraPostRender;
        Logger.LogInfo("Camera post-render hook installed");

        Logger.LogInfo($"RainWorld RL ready - Press {TOGGLE_KEY} to toggle RL mode");
    }

    void OnDisable()
    {
        DisableRLMode();

        // Unsubscribe from camera post-render event
        Camera.onPostRender -= OnCameraPostRender;

        if (inputInjector != null)
        {
            inputInjector.Uninstall();
            Logger.LogInfo("Input hooks removed");
        }
    }

    void OnDestroy()
    {
        DisableRLMode();

        // Cleanup
        frameCapture?.Dispose();
        sharedMemory?.Dispose();

        Logger.LogInfo("RainWorld RL cleaned up");
    }

    void Update()
    {
        if (!initialized)
            return;

        // Check for toggle keybind
        if (Input.GetKeyDown(TOGGLE_KEY))
        {
            ToggleRLMode();
        }

        if (!initialized || !rlModeEnabled)
            return;

        // Check for Python connection and process actions
        stepController.ProcessUpdate();
    }

    void FixedUpdate()
    {
        if (!initialized || !rlModeEnabled)
            return;

        // Track physics ticks
        stepController.ProcessFixedUpdate();
    }

    void LateUpdate()
    {
        if (!initialized || !rlModeEnabled)
            return;

        // This is called after all Update and FixedUpdate calls
        // We use this for frame capture timing
    }

    void OnCameraPostRender(Camera cam)
    {
        if (!initialized || !rlModeEnabled)
            return;

        // Only capture from the main camera
        if (cam != Camera.main)
            return;

        // Capture frame after rendering is complete
        stepController.ProcessPostRender();
    }

    /// <summary>
    /// Enables RL mode - step-driven simulation.
    /// </summary>
    public void EnableRLMode()
    {
        if (!initialized || rlModeEnabled)
            return;

        inputInjector.EnableOverride();
        stepController.Enable();
        rlModeEnabled = true;
        Logger.LogInfo("RL mode enabled - waiting for Python connection");
    }

    /// <summary>
    /// Disables RL mode - returns to normal game.
    /// </summary>
    public void DisableRLMode()
    {
        if (!rlModeEnabled)
            return;

        inputInjector.DisableOverride();
        stepController.Disable();
        rlModeEnabled = false;
        Logger.LogInfo("RL mode disabled");
    }

    /// <summary>
    /// Toggles RL mode on/off.
    /// </summary>
    public void ToggleRLMode()
    {
        if (rlModeEnabled)
            DisableRLMode();
        else
            EnableRLMode();
    }
}
