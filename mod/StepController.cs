using System;
using System.Reflection;
using System.Threading;
using BepInEx.Logging;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Controls the game simulation step-by-step, driven by the Python RL agent.
///
/// While the game is READY and the agent has control (the normal case), steps run in
/// <see cref="RunFastSteps"/>, inside the RainWorld.Update hook: time stays frozen (timescale 0) and
/// each step runs exactly ticksPerStep game ticks, renders the game camera once and captures it, all
/// within one Unity frame; several steps can share a frame. Step lifecycle (sync flag transitions):
///   Python writes action_bits + ACTION_READY
///   -> RunFastSteps consumes the action, writes PROCESSING, runs the ticks (ProcessManager.Update)
///   -> renders the camera; ProcessPostRender (its post-render callback) captures the frame, writes
///      game state + status, step_counter++, writes FRAME_READY last
///   -> Python reads frame, writes IDLE
///
/// While not READY (menus, loading, respawn) or under human override, time runs at 1x between steps
/// so the game can get itself back into a playable state, and steps take the per-frame path:
///   ProcessUpdate consumes the action, unpauses -> ProcessFixedUpdate counts FixedUpdates, pauses
///   when ticksPerStep reached -> ProcessPostRender captures the next rendered frame.
///
/// Per-step game state (offsets 24..53 and the IN_SHELTER / RAIN / CYCLE_SURVIVED game_flags bits):
///   food_max / food_to_hibernate : StoryGameSession.characterStats.maxFood / .foodToHibernate
///                                  (SlugcatStats.cs:77/79; foodToHibernate == maxFood while malnourished, ctor)
///   malnourished                 : SaveState.malnourished (SaveState.cs:18, set by SessionEnded :294-297)
///   cycle_progress               : RainCycle.timer / cycleLength (RainCycle.cs:15/17); timer keeps counting
///                                  past cycleLength (:417) so the value exceeds 1 once the rain is falling.
///                                  (First cycle of a fresh save: the overseer tutorial pins timer = 2000 until
///                                  the player leaves the start rooms, OverseerTutorialBehavior.cs:1769/2015.)
///   IN_SHELTER                   : player.room.abstractRoom.shelter (AbstractRoom.cs:112)
///   RAIN                         : RainCycle.TimeUntilRain <= 0, i.e. RainCycle.RainGameOver (RainCycle.cs:55/139):
///                                  the cycle timer has expired and the lethal rain is falling
///   CYCLE_SURVIVED (edge)        : latched by the On.RainWorldGame.Win hook (RainWorldGame.cs:1578) when the
///                                  player hibernated with enough food (malnourished == false; ShelterDoor.cs:1788),
///                                  reported on the next frame written and then cleared. A starving sleep
///                                  (Win(malnourished: true), also via SleepScreen) does not count; it is visible as
///                                  cycle_number + 1 together with malnourished == 1.
///   region                       : game.overWorld.activeWorld.region.name (World.cs:278, Region.cs:148), the
///                                  region acronym ("SU", "HI", ...); empty while no world is loaded.
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

    /// <summary>OnDemandRendering.renderFrameInterval while nothing needs to be seen (see <see cref="UpdateRendering"/>).</summary>
    private const int SKIP_RENDER_INTERVAL = 1000;

    /// <summary>How long <see cref="RunFastSteps"/> waits inside the frame for the next action before
    /// handing the frame back (a fast agent's next action usually arrives well within this).</summary>
    private const double SPIN_WAIT_MS = 5.0;

    /// <summary>Longest stretch of steps in one frame, so Unity and the game's per-frame work
    /// (GameFlowController, commands, the F10 key) still run at least ~20 times a second.</summary>
    private const double FRAME_BUDGET_MS = 50.0;

    private static readonly FieldInfo timeStackerField =
        typeof(MainLoopProcess).GetField("myTimeStacker", BindingFlags.NonPublic | BindingFlags.Instance);

    // Futile turns the draw code's sprite changes into mesh data in its own Update (Redraw) and LateUpdate (mesh
    // upload), once per Unity frame. A fast step runs both before rendering by hand, or the render would show
    // the geometry from the start of the frame.
    private static readonly Action<Futile> futileUpdate = (Action<Futile>)Delegate.CreateDelegate(
        typeof(Action<Futile>), typeof(Futile).GetMethod("Update", BindingFlags.NonPublic | BindingFlags.Instance));
    private static readonly Action<Futile> futileLateUpdate = (Action<Futile>)Delegate.CreateDelegate(
        typeof(Action<Futile>), typeof(Futile).GetMethod("LateUpdate", BindingFlags.NonPublic | BindingFlags.Instance));

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
    private int forcedTicks = 0;                  // ticks the next RainWorldGame.RawUpdate runs (fast path)

    // Cached game instance (set via OnRainWorldUpdate)
    private RainWorld rainWorld;
    private bool stateErrorLogged = false;

    // Death edge detection
    private AbstractCreature trackedPlayer = null;
    private bool prevDead = false;

    // Cycle-survived edge: set by the RainWorldGame.Win hook, consumed by the next frame write
    private bool hooksInstalled = false;
    private bool pendingCycleSurvived = false;

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

    /// <summary>The agent drives the game: RL mode on, no human override, game READY (frozen between steps).</summary>
    public bool AgentInControl => enabled && !paused && HoldWhenIdle;

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

    /// <summary>Installs the RainWorldGame.Win hook used for the CYCLE_SURVIVED edge.</summary>
    public void Install()
    {
        if (hooksInstalled)
            return;

        On.RainWorldGame.Win += RainWorldGame_Win;
        On.MainLoopProcess.RawUpdate += MainLoopProcess_RawUpdate;
        hooksInstalled = true;
    }

    public void Uninstall()
    {
        if (!hooksInstalled)
            return;

        On.RainWorldGame.Win -= RainWorldGame_Win;
        On.MainLoopProcess.RawUpdate -= MainLoopProcess_RawUpdate;
        hooksInstalled = false;
    }

    /// <summary>
    /// RainWorldGame.Win is the hibernation path (ShelterDoor.cs:1788 and the Watcher warp/echo paths).
    /// It is a no-op while a process switch is already pending (RainWorldGame.cs:1581), so only latch
    /// when it actually ran. malnourished == true is the starving sleep and is not counted.
    /// </summary>
    private void RainWorldGame_Win(On.RainWorldGame.orig_Win orig, RainWorldGame self, bool malnourished, bool fromWarpPoint)
    {
        bool blocked = self.manager != null && self.manager.upcomingProcess != null;
        orig(self, malnourished, fromWarpPoint);

        if (!enabled || blocked)
            return;

        if (!malnourished)
        {
            pendingCycleSurvived = true;
            log?.LogInfo("[StepController] Cycle survived (hibernation)");
        }
        else
        {
            log?.LogInfo("[StepController] Starving sleep (malnourished); not counted as cycle survived");
        }
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
        pendingCycleSurvived = false;

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
        pendingCycleSurvived = false;

        enabled = false;
        OnDemandRendering.renderFrameInterval = 1;
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

        if (!sharedMemory.IsActionReady() || CanFastStep)
        {
            // While READY, RunFastSteps services the actions
            Time.timeScale = HoldWhenIdle ? 0f : 1f;
            return false;
        }

        ConsumeAction();
        stepInProgress = true;
        Time.timeScale = SpeedMultiplier;

        return true;
    }

    /// <summary>Consumes the pending action exactly once and applies it to the input override.</summary>
    private void ConsumeAction()
    {
        uint actionBits = sharedMemory.ReadActionBits();
        ticksPerStep = sharedMemory.ReadTicksPerStep();
        sharedMemory.SignalProcessing();

        inputInjector.SetFromActionBits(actionBits);

        // A pause menu left open (e.g. by a human before releasing F10) would freeze the world
        // while steps keep being serviced; close it the way its CONTINUE button does.
        if (inputInjector.DismissPauseMenu(rainWorld))
            log?.LogInfo("[StepController] Pause menu was open while the agent is in control; dismissing it");

        currentTick = 0;
    }

    /// <summary>The fast path applies: RL mode on, game READY, agent in control, a game camera to render.</summary>
    private bool CanFastStep => AgentInControl && Camera.main != null;

    /// <summary>
    /// Fast path, called from the RainWorld.Update hook once per frame: services whole steps inside this
    /// frame. Each step consumes the action, runs exactly ticksPerStep game ticks through
    /// ProcessManager.Update (see <see cref="MainLoopProcess_RawUpdate"/>), renders the game camera once
    /// and captures it (<see cref="ProcessPostRender"/>, via the camera's post-render callback). It then
    /// waits up to SPIN_WAIT_MS for the next action in the same frame, so a fast agent pays no Unity frame
    /// overhead per step. It hands the frame back when no action comes in time, after FRAME_BUDGET_MS,
    /// when a command is pending, or when the game leaves the playable state (death, game over, a process
    /// switch), which the per-frame flow (GameFlowController) handles.
    /// </summary>
    public void RunFastSteps(RainWorld rw)
    {
        if (!CanFastStep || stepInProgress || waitingForFrameCapture || rw.processManager == null)
            return;

        Camera cam = Camera.main;
        long frameStart = System.Diagnostics.Stopwatch.GetTimestamp();
        long waitStart = frameStart;
        while (true)
        {
            if (!sharedMemory.IsActionReady())
            {
                if (ElapsedMs(waitStart) >= SPIN_WAIT_MS || ElapsedMs(frameStart) >= FRAME_BUDGET_MS ||
                    sharedMemory.ReadCommand() != SharedMemoryBridge.CMD_NONE || !sharedMemory.IsConnected)
                    return;
                Thread.Yield();
                continue;
            }

            ConsumeAction();
            forcedTicks = ticksPerStep;
            rw.processManager.Update(ticksPerStep / 40f); // dt = the ticks' game time (shader clock, session timer)
            forcedTicks = 0;

            futileUpdate(Futile.instance);
            futileLateUpdate(Futile.instance);
            waitingForFrameCapture = true;
            cam.Render();
            if (waitingForFrameCapture)
                ProcessPostRender(); // the post-render callback did not capture (should not happen)

            if (!StillPlayable(rw) || ElapsedMs(frameStart) >= FRAME_BUDGET_MS)
                return;
            waitStart = System.Diagnostics.Stopwatch.GetTimestamp();
        }
    }

    private static double ElapsedMs(long since) =>
        (System.Diagnostics.Stopwatch.GetTimestamp() - since) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

    /// <summary>Story game running with a live player and no game-over prompt or process switch pending.</summary>
    private static bool StillPlayable(RainWorld rw)
    {
        ProcessManager pm = rw.processManager;
        if (!(pm.currentMainLoop is RainWorldGame game) || pm.upcomingProcess != null || game.GameOverModeActive)
            return false;
        AbstractCreature player = GameFlowController.GetPlayer0(pm);
        return player != null && player.realizedCreature != null && !IsDead(player);
    }

    /// <summary>
    /// During a fast step the game's RawUpdate runs exactly the step's ticks (with the Rewired update between
    /// ticks, as the original does) and one GrafUpdate at the unchanged time stacker. The original turns dt
    /// into ticks through a time accumulator, at most 3 per call; RainWorldGame.RawUpdate's own work before
    /// base.RawUpdate (session timer, slow-motion bookkeeping) still runs with the step's dt.
    /// </summary>
    private void MainLoopProcess_RawUpdate(On.MainLoopProcess.orig_RawUpdate orig, MainLoopProcess self, float dt)
    {
        if (forcedTicks <= 0 || !(self is RainWorldGame))
        {
            orig(self, dt);
            return;
        }

        int ticks = forcedTicks;
        forcedTicks = 0;
        for (int i = 0; i < ticks; i++)
        {
            if (i > 0)
                self.manager.rainWorld.RunRewiredUpdate();
            self.Update();
        }
        self.GrafUpdate((float)timeStackerField.GetValue(self));
    }

    /// <summary>
    /// Called every Unity Update, after <see cref="ProcessUpdate"/>, and right after a capture. While the game
    /// is frozen between steps and during a step's ticks nothing new needs to be seen, so rendering is skipped;
    /// only the capture is rendered. Matters most with software rendering (Linux/Wine without a GPU), where
    /// each frame costs several cores. Human override and the not-READY flow (menus, loading, respawn) render
    /// every frame. A new interval takes effect on the next frame, so the frame that completes a step is not
    /// rendered: the capture happens one frame later, which shows the same state because time is frozen
    /// (timeScale 0) as soon as the step completes.
    /// </summary>
    public void UpdateRendering()
    {
        bool render = !AgentInControl || waitingForFrameCapture;
        OnDemandRendering.renderFrameInterval = render ? 1 : SKIP_RENDER_INTERVAL;
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

        // Is an in-game prompt (dialog, game-over "press X to restart", pause menu) awaiting a key?
        sharedMemory.SetGameFlag(SharedMemoryBridge.GAME_FLAG_DIALOG_OPEN, InputInjector.IsPromptAwaitingInput(rainWorld));

        // Cycle-survived edge: reported on exactly one frame, whether or not the game is still
        // the current process (the SleepScreen -> Game redirect may already have happened).
        sharedMemory.SetGameFlag(SharedMemoryBridge.GAME_FLAG_CYCLE_SURVIVED, pendingCycleSurvived);
        pendingCycleSurvived = false;

        sharedMemory.IncrementStepCounter();

        // FRAME_READY must be the very last write
        sharedMemory.SignalFrameReady();

        // Reset for next step
        currentTick = 0;
        waitingForFrameCapture = false;
        if (!paused)
            Time.timeScale = HoldWhenIdle ? 0f : 1f;
        UpdateRendering(); // stop rendering from the next frame on
    }

    /// <summary>
    /// Writes karma/food/position/room/cycle, food_max/food_to_hibernate/malnourished, cycle_progress, region and
    /// the IN_SHELTER / RAIN flags (zeros / -1 when unavailable) and returns the death edge: true only on
    /// the step where player 0 went alive -> dead. Tracking resets whenever a different player instance
    /// appears (new game / respawn), so the next death is detected again.
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
            bool malnourished = false;
            SaveState save = story.saveState;
            if (save != null)
            {
                cycle = save.cycleNumber;
                malnourished = save.malnourished;
                if (save.deathPersistentSaveData != null)
                {
                    karma = save.deathPersistentSaveData.karma;
                    karmaCap = save.deathPersistentSaveData.karmaCap;
                }
            }

            // Food meter. characterStats is built from (saveStateNumber, malnourished) in the
            // StoryGameSession ctor, so foodToHibernate == maxFood while malnourished.
            int foodMax = 0, foodToHibernate = 0;
            SlugcatStats stats = story.characterStats;
            if (stats != null)
            {
                foodMax = stats.maxFood;
                foodToHibernate = stats.foodToHibernate;
            }
            else if (story.saveStateNumber != null)
            {
                var meter = SlugcatStats.SlugcatFoodMeter(story.saveStateNumber);
                foodMax = meter.x;
                foodToHibernate = malnourished ? meter.x : meter.y;
            }

            // Rain cycle (world may be null while a region loads)
            float cycleProgress = 0f;
            bool rain = false;
            World world = game.overWorld?.activeWorld;
            string region = world?.region?.name ?? "";
            RainCycle rainCycle = world?.rainCycle;
            if (rainCycle != null && rainCycle.cycleLength > 0)
            {
                cycleProgress = (float)rainCycle.timer / (float)rainCycle.cycleLength;
                rain = rainCycle.TimeUntilRain <= 0;
            }

            // Player fields
            int food = 0;
            float x = 0f, y = 0f;
            int roomIndex = -1;
            AbstractRoom abstractRoom = null;

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
                {
                    abstractRoom = player.room.abstractRoom;
                    roomIndex = abstractRoom.index;
                }
                else
                    roomIndex = abstractPlayer.pos.room;
            }
            else
            {
                if (abstractPlayer.state is PlayerState ps)
                    food = ps.foodInStomach;
                roomIndex = abstractPlayer.pos.room;
            }
            if (abstractRoom == null && world != null && roomIndex >= 0)
                abstractRoom = world.GetAbstractRoom(roomIndex);
            bool inShelter = abstractRoom != null && abstractRoom.shelter;

            sharedMemory.WriteGameState(karma, karmaCap, food, x, y, roomIndex, cycle);
            sharedMemory.WriteFoodMax(foodMax);
            sharedMemory.WriteFoodToHibernate(foodToHibernate);
            sharedMemory.WriteMalnourished(malnourished);
            sharedMemory.WriteCycleProgress(cycleProgress);
            sharedMemory.WriteRegion(region);
            sharedMemory.SetGameFlag(SharedMemoryBridge.GAME_FLAG_IN_SHELTER, inShelter);
            sharedMemory.SetGameFlag(SharedMemoryBridge.GAME_FLAG_RAIN, rain);
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
