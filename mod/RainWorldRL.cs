using BepInEx;
using BepInEx.Configuration;
using System;
using System.Security.Permissions;
using UnityEngine;

#pragma warning disable CS0618 // SecurityAction.RequestMinimum is obsolete. However, this does not apply to the mod, which still needs it. Suppress the warning indicating that it is obsolete.
[assembly: SecurityPermission(SecurityAction.RequestMinimum, SkipVerification = true)]
#pragma warning restore CS0618

/// <summary>
/// Rain World RL - Turns Rain World into a reinforcement learning environment.
/// Communicates with Python via shared memory (protocol v4, docs/PROTOCOL.md).
///
/// RL mode is driven by Python's CONNECTED status bit: rising edge enters RL mode (swap to the
/// isolated RL save, auto-start a story game), falling edge exits it (save, return to the normal
/// save). F10 toggles HUMAN_OVERRIDE while RL mode is on. Actions are raw key bitfields
/// (SharedMemoryBridge.KEY_*) injected by InputInjector; the pause button is blocked while the
/// agent is in control.
/// </summary>
[BepInPlugin("rainworld.rl", "RainWorldRL", "0.2")]
public class RainWorldRL : BaseUnityPlugin
{
    private const KeyCode HUMAN_OVERRIDE_KEY = KeyCode.F10;

    // Core components
    private SharedMemoryBridge sharedMemory;
    private InputInjector inputInjector;
    private FrameCapture frameCapture;
    private StepController stepController;
    private SaveRedirector saveRedirector;
    private GameFlowController gameFlow;
    private WorkerThreadFix workerThreadFix;
    private readonly CoreWarmer coreWarmer = new CoreWarmer();

    // Config
    private ConfigEntry<string> cfgSlugcat;
    private ConfigEntry<string> cfgSaveName;
    private ConfigEntry<float> cfgSpeedMultiplier;
    private ConfigEntry<int> cfgRenderScale;
    private ConfigEntry<bool> cfgVerboseLogging;

    // State
    private bool initialized = false;
    private bool hooksInstalled = false;
    private bool modsInitHookSubscribed = false;
    private bool stepControllerEnabled = false;
    private bool humanOverride = false;
    private bool lastConnected = false;
    private RainWorld rainWorld;

    void Awake()
    {
        Logger.LogInfo("RainWorld RL initializing...");

        try
        {
            cfgSlugcat = Config.Bind("General", "Slugcat", "White",
                "Slugcat to play as in RL mode (ExtEnum name: White, Yellow, Red, or a DLC name such as Gourmand).");
            cfgSaveName = Config.Bind("General", "SaveName", "default",
                "Name of the isolated RL save profile. Stored under BepInEx/plugins/RainWorldRL/saves/<name>/.");
            cfgSpeedMultiplier = Config.Bind("Simulation", "SpeedMultiplier", 50f,
                new ConfigDescription("Game-time speed multiplier while a step is running.", new AcceptableValueRange<float>(1f, 1000f)));
            cfgRenderScale = Config.Bind("Simulation", "RenderScale", 2,
                new ConfigDescription("While the agent is in control the game renders straight into a texture this many times the " +
                    "observation size (same field of view), averaged down to the observation. 0 renders at the game's own 1366x768.",
                    new AcceptableValueRange<int>(0, 8)));
            cfgVerboseLogging = Config.Bind("Logging", "Verbose", false,
                "Log every process switch and other high-frequency diagnostics.");

            sharedMemory = new SharedMemoryBridge();
            Logger.LogInfo($"Shared memory bridge created ({sharedMemory.Location})");

            inputInjector = new InputInjector();
            frameCapture = new FrameCapture(SharedMemoryBridge.DEFAULT_FRAME_WIDTH, SharedMemoryBridge.DEFAULT_FRAME_HEIGHT)
            {
                RenderScale = cfgRenderScale.Value,
            };

            stepController = new StepController(sharedMemory, inputInjector, frameCapture, Logger)
            {
                SpeedMultiplier = cfgSpeedMultiplier.Value,
                VerboseLogging = cfgVerboseLogging.Value,
            };

            saveRedirector = new SaveRedirector(Logger)
            {
                SaveName = cfgSaveName.Value,
            };

            gameFlow = new GameFlowController(saveRedirector, sharedMemory, Logger)
            {
                SlugcatName = cfgSlugcat.Value,
                VerboseLogging = cfgVerboseLogging.Value,
            };
            workerThreadFix = new WorkerThreadFix(Logger);
            if (coreWarmer.Enabled)
                Logger.LogInfo($"Core warmer enabled ({CoreWarmer.ENV_VAR}=1)");

            cfgSpeedMultiplier.SettingChanged += (s, e) => stepController.SpeedMultiplier = cfgSpeedMultiplier.Value;
            cfgRenderScale.SettingChanged += (s, e) => frameCapture.RenderScale = cfgRenderScale.Value;
            cfgSlugcat.SettingChanged += (s, e) => gameFlow.SlugcatName = cfgSlugcat.Value;
            cfgSaveName.SettingChanged += (s, e) => saveRedirector.SaveName = cfgSaveName.Value;
            cfgVerboseLogging.SettingChanged += (s, e) =>
            {
                stepController.VerboseLogging = cfgVerboseLogging.Value;
                gameFlow.VerboseLogging = cfgVerboseLogging.Value;
            };

            sharedMemory.SetStatusFlag(SharedMemoryBridge.STATUS_MOD_ALIVE, true);

            initialized = true;
            Logger.LogInfo($"RainWorld RL initialized (slugcat={cfgSlugcat.Value}, save={cfgSaveName.Value}, speed={cfgSpeedMultiplier.Value}x)");
        }
        catch (Exception ex)
        {
            Logger.LogError($"Failed to initialize RainWorld RL: {ex}");
        }
    }

    void OnEnable()
    {
        if (!initialized)
            return;

        // Install game hooks at the conventional time (after the game registered its ExtEnums).
        if (!modsInitHookSubscribed)
        {
            On.RainWorld.OnModsInit += RainWorld_OnModsInit;
            modsInitHookSubscribed = true;
        }

        Camera.onPostRender += OnCameraPostRender;
        Logger.LogInfo($"RainWorld RL ready - connect from Python to enter RL mode; {HUMAN_OVERRIDE_KEY} toggles human override");
    }

    void OnDisable()
    {
        Camera.onPostRender -= OnCameraPostRender;
        ForceStopRLMode("plugin disabled");
    }

    void OnDestroy()
    {
        ForceStopRLMode("plugin destroyed");
        coreWarmer.SetActive(false);
        UninstallHooks();

        if (modsInitHookSubscribed)
        {
            On.RainWorld.OnModsInit -= RainWorld_OnModsInit;
            modsInitHookSubscribed = false;
        }

        if (sharedMemory != null)
        {
            sharedMemory.UpdateStatusBits(SharedMemoryBridge.MOD_OWNED_STATUS_MASK, 0);
            sharedMemory.Dispose();
        }
        frameCapture?.Dispose();

        Logger.LogInfo("RainWorld RL cleaned up");
    }

    private void RainWorld_OnModsInit(On.RainWorld.orig_OnModsInit orig, RainWorld self)
    {
        orig(self);
        rainWorld = self;
        InstallHooks();
    }

    private void InstallHooks()
    {
        if (hooksInstalled)
            return;

        try
        {
            On.RainWorld.Update += RainWorld_Update;
            inputInjector.Install();
            saveRedirector.Install();
            gameFlow.Install();
            workerThreadFix.Install();
            stepController.Install();
            hooksInstalled = true;
            Logger.LogInfo("Game hooks installed");
        }
        catch (Exception ex)
        {
            Logger.LogError($"Failed to install hooks: {ex}");
        }
    }

    private void UninstallHooks()
    {
        if (!hooksInstalled)
            return;

        workerThreadFix?.Uninstall();
        stepController?.Uninstall();
        gameFlow?.Uninstall();
        saveRedirector?.Uninstall();
        inputInjector?.Uninstall();
        On.RainWorld.Update -= RainWorld_Update;
        hooksInstalled = false;
        Logger.LogInfo("Game hooks removed");
    }

    private void RainWorld_Update(On.RainWorld.orig_Update orig, RainWorld self)
    {
        rainWorld = self;
        stepController.OnRainWorldUpdate(self);
        orig(self);
        if (stepControllerEnabled)
            stepController.RunFastSteps(self);
    }

    void Update()
    {
        if (!initialized)
            return;

        sharedMemory.IncrementHeartbeat();

        // Hot-reload friendliness: if we were loaded after OnModsInit already ran, install late.
        if (!hooksInstalled)
        {
            RainWorld rw = RWCustom.Custom.rainWorld;
            if (rw != null && rw.processManager != null)
            {
                rainWorld = rw;
                Logger.LogInfo("OnModsInit already passed; installing hooks now");
                InstallHooks();
            }
        }

        if (rainWorld == null)
            rainWorld = RWCustom.Custom.rainWorld;
        if (rainWorld == null || rainWorld.processManager == null)
            return;

        // RL mode follows Python's CONNECTED bit
        bool connected = sharedMemory.IsConnected;
        if (connected != lastConnected)
        {
            lastConnected = connected;
            Logger.LogInfo(connected ? "Python connected -> entering RL mode" : "Python disconnected -> exiting RL mode");
            gameFlow.SetDesired(connected);
        }

        gameFlow.Update(rainWorld);

        // Step controller + input override follow RL mode (including transitions)
        bool rlActive = gameFlow.IsActive;
        if (rlActive != stepControllerEnabled)
        {
            if (rlActive)
            {
                inputInjector.EnableOverride();
                stepController.Enable();
            }
            else
            {
                stepController.Disable();
                inputInjector.DisableOverride();
                humanOverride = false;
            }
            stepControllerEnabled = rlActive;
        }

        // F10: human override (only meaningful in RL mode)
        if (Input.GetKeyDown(HUMAN_OVERRIDE_KEY))
        {
            if (!rlActive)
                Logger.LogInfo($"{HUMAN_OVERRIDE_KEY}: RL mode is off (no Python client connected); nothing to override");
            else
                SetHumanOverride(!humanOverride);
        }

        // Commands
        byte command = sharedMemory.ReadCommand();
        if (command == SharedMemoryBridge.CMD_RESET && !gameFlow.ResetInProgress)
        {
            if (!gameFlow.RequestReset())
            {
                Logger.LogWarning("RESET command received but RL mode is not fully on; reporting ERROR");
                sharedMemory.WriteCommandResult(SharedMemoryBridge.RESULT_ERROR);
                sharedMemory.WriteCommand(SharedMemoryBridge.CMD_NONE);
            }
        }
        else if (command == SharedMemoryBridge.CMD_KILL_PLAYER)
        {
            // Debug/testing aid: kill player 0 synchronously and ack as soon as it is dead.
            // The respawn is NOT awaited - Python observes it through subsequent steps.
            bool killed = false;
            try
            {
                killed = gameFlow.KillPlayer(rainWorld);
            }
            catch (Exception ex)
            {
                Logger.LogError($"KILL_PLAYER failed: {ex}");
            }
            if (!killed)
                Logger.LogWarning("KILL_PLAYER command could not be applied; reporting ERROR");
            sharedMemory.WriteCommandResult(killed ? SharedMemoryBridge.RESULT_OK : SharedMemoryBridge.RESULT_ERROR);
            sharedMemory.WriteCommand(SharedMemoryBridge.CMD_NONE);
        }
        else if (command == SharedMemoryBridge.CMD_ENTER_SHELTER)
        {
            // Debug/testing aid: send player 0 into its den shelter with command_arg food pips and ack at once.
            // The arrival and any sleep are observed through subsequent steps.
            bool sent = false;
            try
            {
                sent = gameFlow.EnterShelter(rainWorld, sharedMemory.ReadCommandArg());
            }
            catch (Exception ex)
            {
                Logger.LogError($"ENTER_SHELTER failed: {ex}");
            }
            if (!sent)
                Logger.LogWarning("ENTER_SHELTER command could not be applied; reporting ERROR");
            sharedMemory.WriteCommandResult(sent ? SharedMemoryBridge.RESULT_OK : SharedMemoryBridge.RESULT_ERROR);
            sharedMemory.WriteCommand(SharedMemoryBridge.CMD_NONE);
        }
        else if (command == SharedMemoryBridge.CMD_HOP_ROOM || command == SharedMemoryBridge.CMD_SWITCH_REGION)
        {
            // Debug/testing aids: send player 0 into a neighbouring room / region (command_arg picks the exit / gate)
            // and ack once it is on its way. The arrival is observed through subsequent steps.
            bool hop = command == SharedMemoryBridge.CMD_HOP_ROOM;
            string name = hop ? "HOP_ROOM" : "SWITCH_REGION";
            bool sent = false;
            try
            {
                int arg = sharedMemory.ReadCommandArg();
                sent = hop ? gameFlow.HopRoom(rainWorld, arg) : gameFlow.SwitchRegion(rainWorld, arg);
            }
            catch (Exception ex)
            {
                Logger.LogError($"{name} failed: {ex}");
            }
            if (!sent)
                Logger.LogWarning($"{name} command could not be applied; reporting ERROR");
            sharedMemory.WriteCommandResult(sent ? SharedMemoryBridge.RESULT_OK : SharedMemoryBridge.RESULT_ERROR);
            sharedMemory.WriteCommand(SharedMemoryBridge.CMD_NONE);
        }
        else if (command != SharedMemoryBridge.CMD_NONE && command != SharedMemoryBridge.CMD_RESET)
        {
            Logger.LogWarning($"Unknown command {command}; reporting ERROR");
            sharedMemory.WriteCommandResult(SharedMemoryBridge.RESULT_ERROR);
            sharedMemory.WriteCommand(SharedMemoryBridge.CMD_NONE);
        }

        // Status bits (PLAYER_DEAD is written per step, MOD_ALIVE in Awake/OnDestroy)
        byte bits = 0;
        if (gameFlow.Ready) bits |= SharedMemoryBridge.STATUS_READY;
        if (gameFlow.InGame) bits |= SharedMemoryBridge.STATUS_IN_GAME;
        if (humanOverride) bits |= SharedMemoryBridge.STATUS_HUMAN_OVERRIDE;
        sharedMemory.UpdateStatusBits(
            (byte)(SharedMemoryBridge.STATUS_READY | SharedMemoryBridge.STATUS_IN_GAME | SharedMemoryBridge.STATUS_HUMAN_OVERRIDE),
            bits);

        // Freeze between steps only while the game is actually playable
        stepController.HoldWhenIdle = gameFlow.Ready;

        if (stepControllerEnabled)
            stepController.ProcessUpdate();
        stepController.UpdateRendering();
        frameCapture.SetAgentInControl(stepController.AgentInControl);
        coreWarmer.SetActive(stepController.AgentInControl);
    }

    void FixedUpdate()
    {
        if (!initialized || !stepControllerEnabled)
            return;

        stepController.ProcessFixedUpdate();
    }

    void OnCameraPostRender(Camera cam)
    {
        if (!initialized || !stepControllerEnabled)
            return;

        // Capture from the main camera; fall back to any camera if none is tagged.
        Camera main = Camera.main;
        if (main != null && cam != main)
            return;

        stepController.ProcessPostRender();
    }

    private void SetHumanOverride(bool value)
    {
        if (humanOverride == value)
            return;

        humanOverride = value;
        if (humanOverride)
        {
            stepController.SetPaused(true);
            inputInjector.DisableOverride(); // keyboard passes through
            Time.timeScale = 1f;
            Logger.LogInfo("Human override ON - you have control; Python is waiting");
        }
        else
        {
            inputInjector.EnableOverride();
            stepController.SetPaused(false);
            Logger.LogInfo("Human override OFF - agent resumes");
        }
        sharedMemory.SetStatusFlag(SharedMemoryBridge.STATUS_HUMAN_OVERRIDE, humanOverride);
    }

    /// <summary>
    /// Best-effort synchronous teardown of RL mode (used when the plugin is disabled/destroyed and
    /// the async exit state machine cannot run). The save swap cannot be completed here.
    /// </summary>
    private void ForceStopRLMode(string reason)
    {
        if (stepController != null && stepControllerEnabled)
        {
            stepController.Disable();
            stepControllerEnabled = false;
        }
        inputInjector?.DisableOverride();
        humanOverride = false;

        if (gameFlow != null && gameFlow.IsActive)
        {
            gameFlow.SetDesired(false);
            Logger.LogWarning($"RL mode force-stopped ({reason}). " +
                (saveRedirector != null && saveRedirector.Active
                    ? "The RL save is still the active progression; restart the game to return to the normal save."
                    : ""));
        }
        lastConnected = false;
    }
}
