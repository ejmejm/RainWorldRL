using System;
using BepInEx.Logging;
using UnityEngine;

/// <summary>
/// Owns "RL mode": entering/leaving it (save swap), auto-navigating menus into a story game,
/// skipping end-of-cycle screens, and the RESET command.
///
/// Everything asynchronous is a small state machine advanced from <see cref="Update"/>.
/// The plugin tells us whether RL mode is desired (Python's CONNECTED bit) via <see cref="SetDesired"/>
/// and the machine converges on it:
///
///   Off --(desired)--> EnteringLeaveGame --> EnteringSwap --> On
///   On  --(!desired)-> ExitingLeaveGame  --> ExitingSwap  --> Off
///
///   EnteringLeaveGame : if a RainWorldGame is running (human playing on the normal save) call
///                       game.ExitToMenu() (saves as quit) and wait until a menu is current.
///   EnteringSwap      : SaveRedirector.SwapToRLSave; wait for progressionLoaded.
///   On                : auto-navigation (menu -> story game as configured slugcat), screen skipping,
///                       RESET command.
///   ExitingLeaveGame  : if in game, game.ExitToMenu() (saves to the RL progression); wait for menu.
///   ExitingSwap       : SaveRedirector.SwapToNormalSave; wait for progressionLoaded.
///
/// Auto-navigation: the PostSwitchMainProcess hook only records that a new process arrived; the
/// actual RequestMainProcessSwitch must happen later from Update because ProcessManager.Update
/// clears processAfterModFinalization only AFTER PostSwitchMainProcess returns, and
/// ActualProcessSwitch silently drops requests while it is non-null (ProcessManager.cs 760-763, 881-883).
///
/// Screen skipping: RequestMainProcessSwitch(DeathScreen/SleepScreen/StarveScreen/Dream/GhostScreen/
/// KarmaToMaxScreen/KarmaToMinScreen/VengeanceGhostScreen/Tips) is rewritten to Game with
/// startGameCondition = Load and a 0.05s fade. This is exactly what the screens' CONTINUE button does
/// (KarmaLadderScreen.cs 245-278). Persistence is unaffected because both the sleep save
/// (SaveState.SessionEnded survived=true -> SaveWorldStateAndProgression, SaveState.cs:473) and the death
/// save (SessionEnded survived=false -> SaveProgressionAndDeathPersistentDataOfCurrentState, SaveState.cs:~523)
/// happen inside RainWorldGame.Win/GoToDeathScreen BEFORE the screen is requested.
///
/// RESET: WipeAll on the RL progression, wait for the async re-read, WipeSaveState(slugcat), start a
/// New story game, wait for Ready, then ack via command_result. Death is NOT a reset (nothing wiped).
/// </summary>
public class GameFlowController
{
    private enum FlowState
    {
        Off,
        EnteringLeaveGame,
        EnteringSwap,
        On,
        ExitingLeaveGame,
        ExitingSwap,
    }

    private enum ResetState
    {
        None,
        WaitForSwitchIdle,
        Wipe,
        WaitForWipe,
        WaitForNewGame,
    }

    private const float SKIP_SCREEN_FADE_SECONDS = 0.05f;
    private const float AUTO_START_RETRY_SECONDS = 3f;
    private const int AUTO_START_MAX_ATTEMPTS = 5;
    private const float RESET_TIMEOUT_SECONDS = 55f;

    private readonly SaveRedirector saves;
    private readonly SharedMemoryBridge sharedMemory;
    private readonly ManualLogSource log;

    private bool hooksInstalled = false;
    private bool desired = false;
    private FlowState state = FlowState.Off;

    // Enter/exit bookkeeping
    private bool exitToMenuRequested = false;

    // Auto-navigation debounce (per process instance)
    private MainLoopProcess lastAutoStartProcess = null;
    private float lastAutoStartTime = 0f;
    private int autoStartAttempts = 0;

    // Reset bookkeeping
    private ResetState resetState = ResetState.None;
    private RainWorldGame resetOldGame = null;
    private float resetStartTime = 0f;

    /// <summary>Slugcat to play as, by ExtEnum name (e.g. "White", "Yellow", "Red").</summary>
    public string SlugcatName { get; set; } = "White";

    public bool VerboseLogging { get; set; } = false;

    /// <summary>RL mode is on or in transition (anything but Off).</summary>
    public bool IsActive => state != FlowState.Off;

    /// <summary>RL mode fully entered (save swapped) and not exiting.</summary>
    public bool IsOn => state == FlowState.On;

    public bool ResetInProgress => resetState != ResetState.None;

    /// <summary>Set once per Update by <see cref="Update"/>.</summary>
    public bool Ready { get; private set; } = false;

    /// <summary>Current main process is a RainWorldGame (regardless of RL mode).</summary>
    public bool InGame { get; private set; } = false;

    public GameFlowController(SaveRedirector saves, SharedMemoryBridge sharedMemory, ManualLogSource log)
    {
        this.saves = saves;
        this.sharedMemory = sharedMemory;
        this.log = log;
    }

    // ------------------------------------------------------------------ hooks

    public void Install()
    {
        if (hooksInstalled)
            return;

        On.ProcessManager.PostSwitchMainProcess += ProcessManager_PostSwitchMainProcess;
        On.ProcessManager.RequestMainProcessSwitch_ProcessID_float += ProcessManager_RequestMainProcessSwitch;
        hooksInstalled = true;
    }

    public void Uninstall()
    {
        if (!hooksInstalled)
            return;

        On.ProcessManager.PostSwitchMainProcess -= ProcessManager_PostSwitchMainProcess;
        On.ProcessManager.RequestMainProcessSwitch_ProcessID_float -= ProcessManager_RequestMainProcessSwitch;
        hooksInstalled = false;
    }

    private void ProcessManager_PostSwitchMainProcess(
        On.ProcessManager.orig_PostSwitchMainProcess orig, ProcessManager self, ProcessManager.ProcessID ID)
    {
        orig(self, ID);

        // A new process instance exists: allow one auto-start attempt for it.
        // (The request itself is issued from Update - see class remarks.)
        lastAutoStartProcess = null;
        autoStartAttempts = 0;

        if (VerboseLogging)
            log?.LogInfo($"[GameFlow] Process switched to {ID} (now {self.currentMainLoop?.ID})");
    }

    private void ProcessManager_RequestMainProcessSwitch(
        On.ProcessManager.orig_RequestMainProcessSwitch_ProcessID_float orig, ProcessManager self,
        ProcessManager.ProcessID ID, float fadeOutSeconds)
    {
        if (state == FlowState.On && IsSkippableEndScreen(ID))
        {
            log?.LogInfo($"[GameFlow] Skipping {ID} -> Game (Load)");
            self.menuSetup.startGameCondition = ProcessManager.MenuSetup.StoryGameInitCondition.Load;
            ID = ProcessManager.ProcessID.Game;
            fadeOutSeconds = SKIP_SCREEN_FADE_SECONDS;
        }

        orig(self, ID, fadeOutSeconds);
    }

    private static bool IsSkippableEndScreen(ProcessManager.ProcessID ID)
    {
        if (ID == null)
            return false;

        if (ID == ProcessManager.ProcessID.DeathScreen ||
            ID == ProcessManager.ProcessID.SleepScreen ||
            ID == ProcessManager.ProcessID.StarveScreen ||
            ID == ProcessManager.ProcessID.Dream ||
            ID == ProcessManager.ProcessID.GhostScreen ||
            ID == ProcessManager.ProcessID.KarmaToMaxScreen)
            return true;

        // DLC enum values are null until their mod registers them.
        if (ModManager.MSC)
        {
            if (MoreSlugcats.MoreSlugcatsEnums.ProcessID.KarmaToMinScreen != null &&
                ID == MoreSlugcats.MoreSlugcatsEnums.ProcessID.KarmaToMinScreen)
                return true;
            if (MoreSlugcats.MoreSlugcatsEnums.ProcessID.VengeanceGhostScreen != null &&
                ID == MoreSlugcats.MoreSlugcatsEnums.ProcessID.VengeanceGhostScreen)
                return true;
        }
        if (ModManager.MMF)
        {
            if (MoreSlugcats.MMFEnums.ProcessID.Tips != null && ID == MoreSlugcats.MMFEnums.ProcessID.Tips)
                return true;
        }

        return false;
    }

    // ------------------------------------------------------------------ public API

    /// <summary>Tells the controller whether RL mode should be on. The state machine converges on it.</summary>
    public void SetDesired(bool on)
    {
        if (desired == on)
            return;

        desired = on;
        log?.LogInfo($"[GameFlow] RL mode {(on ? "requested" : "release requested")}");
    }

    /// <summary>Explicit entry point (equivalent to SetDesired(true)).</summary>
    public void EnterRLMode(RainWorld rw)
    {
        SetDesired(true);
        Update(rw);
    }

    /// <summary>Explicit exit point (equivalent to SetDesired(false)).</summary>
    public void ExitRLMode(RainWorld rw)
    {
        SetDesired(false);
        Update(rw);
    }

    /// <summary>
    /// Begins a RESET: wipe the RL save and start a fresh story game. Result is reported through
    /// shared memory (command_result = OK/ERROR, then command = NONE). Returns false if a reset is
    /// not possible right now (not in RL mode / already resetting) - the caller should report ERROR.
    /// </summary>
    public bool RequestReset()
    {
        if (state != FlowState.On || resetState != ResetState.None)
            return false;

        log?.LogInfo("[GameFlow] RESET requested");
        resetState = ResetState.WaitForSwitchIdle;
        resetStartTime = Time.realtimeSinceStartup;
        return true;
    }

    /// <summary>True when a story RainWorldGame is current and no process switch is pending.</summary>
    public static bool IsInStoryGame(ProcessManager pm)
    {
        if (pm == null || pm.upcomingProcess != null)
            return false;

        RainWorldGame game = pm.currentMainLoop as RainWorldGame;
        return game != null && game.session != null && game.IsStorySession;
    }

    /// <summary>Returns player 0's abstract creature, or null if no story game / no player yet.</summary>
    public static AbstractCreature GetPlayer0(ProcessManager pm)
    {
        try
        {
            RainWorldGame game = pm?.currentMainLoop as RainWorldGame;
            if (game == null || game.session == null || game.Players == null || game.Players.Count == 0)
                return null;
            return game.Players[0];
        }
        catch
        {
            return null;
        }
    }

    // ------------------------------------------------------------------ update loop

    /// <summary>Advances all state machines. Call once per Unity Update.</summary>
    public void Update(RainWorld rw)
    {
        if (rw == null || rw.processManager == null)
        {
            Ready = false;
            InGame = false;
            return;
        }

        ProcessManager pm = rw.processManager;
        saves.Update(rw);

        try
        {
            switch (state)
            {
                case FlowState.Off:
                    if (desired)
                    {
                        exitToMenuRequested = false;
                        state = FlowState.EnteringLeaveGame;
                        log?.LogInfo("[GameFlow] Entering RL mode");
                    }
                    break;

                case FlowState.EnteringLeaveGame:
                    if (LeaveGameStep(rw, pm))
                    {
                        saves.SwapToRLSave(rw);
                        state = FlowState.EnteringSwap;
                    }
                    break;

                case FlowState.EnteringSwap:
                    if (!saves.IsSwapping)
                    {
                        state = FlowState.On;
                        lastAutoStartProcess = null;
                        autoStartAttempts = 0;
                        log?.LogInfo("[GameFlow] RL mode on");
                    }
                    break;

                case FlowState.On:
                    if (!desired)
                    {
                        AbortReset("RL mode released");
                        exitToMenuRequested = false;
                        state = FlowState.ExitingLeaveGame;
                        log?.LogInfo("[GameFlow] Exiting RL mode");
                        break;
                    }
                    if (resetState != ResetState.None)
                        UpdateReset(rw, pm);
                    else
                        UpdateAutoNavigate(rw, pm);
                    break;

                case FlowState.ExitingLeaveGame:
                    if (LeaveGameStep(rw, pm))
                    {
                        saves.SwapToNormalSave(rw);
                        state = FlowState.ExitingSwap;
                    }
                    break;

                case FlowState.ExitingSwap:
                    if (!saves.IsSwapping)
                    {
                        state = FlowState.Off;
                        log?.LogInfo("[GameFlow] RL mode off");
                    }
                    break;
            }
        }
        catch (Exception ex)
        {
            log?.LogError($"[GameFlow] Error in state {state}: {ex}");
            if (resetState != ResetState.None)
                FinishReset(false);
        }

        InGame = pm.currentMainLoop is RainWorldGame;
        Ready = ComputeReady(pm);
    }

    private bool ComputeReady(ProcessManager pm)
    {
        if (state != FlowState.On || saves.IsSwapping || resetState != ResetState.None)
            return false;
        if (!IsInStoryGame(pm))
            return false;

        AbstractCreature player = GetPlayer0(pm);
        return player != null && player.realizedCreature != null;
    }

    /// <summary>
    /// If a game is running, asks it to exit to the menu (which saves). Returns true once it is
    /// safe to swap the progression: a menu other than Initialization is current, no switch or
    /// dialog is pending, and the current progression has finished loading. (The
    /// InitializationScreen constructs PlayerProgression itself several times - InitializationScreen.cs
    /// 197/580/676/699/708/747 - so we never swap underneath it.)
    /// </summary>
    private bool LeaveGameStep(RainWorld rw, ProcessManager pm)
    {
        MainLoopProcess current = pm.currentMainLoop;

        if (current is RainWorldGame game)
        {
            if (!exitToMenuRequested && pm.upcomingProcess == null)
            {
                log?.LogInfo("[GameFlow] Leaving running game via ExitToMenu()");
                game.ExitToMenu();
                exitToMenuRequested = true;
            }
            return false;
        }

        if (current == null || current.ID == ProcessManager.ProcessID.Initialization)
            return false;
        if (pm.upcomingProcess != null || pm.IsRunningAnyDialog)
            return false;
        if (!rw.OptionsReady || rw.progression == null || !rw.progression.progressionLoaded || rw.progression.SaveDataBusy)
            return false;

        exitToMenuRequested = false;
        return true;
    }

    // ------------------------------------------------------------------ auto-navigation

    private void UpdateAutoNavigate(RainWorld rw, ProcessManager pm)
    {
        MainLoopProcess current = pm.currentMainLoop;
        if (current == null || current is RainWorldGame)
            return;
        if (!IsAutoStartMenu(current))
            return;
        if (pm.upcomingProcess != null || pm.IsRunningAnyDialog)
            return;
        if (!rw.OptionsReady || rw.progression == null || !rw.progression.progressionLoaded || rw.progression.SaveDataBusy)
            return;

        // Debounce: one attempt per process instance, with a slow retry in case the request was dropped.
        if (ReferenceEquals(current, lastAutoStartProcess))
        {
            if (autoStartAttempts >= AUTO_START_MAX_ATTEMPTS)
                return;
            if (Time.realtimeSinceStartup - lastAutoStartTime < AUTO_START_RETRY_SECONDS)
                return;
            log?.LogWarning($"[GameFlow] Auto-start from {current.ID} did not take effect; retrying ({autoStartAttempts + 1})");
        }

        lastAutoStartProcess = current;
        lastAutoStartTime = Time.realtimeSinceStartup;
        autoStartAttempts++;

        log?.LogInfo($"[GameFlow] Auto-starting story game from {current.ID}");
        StartStoryGame(rw, forceNew: false);
    }

    /// <summary>Menus we auto-leave. Excludes Initialization and Dialog; any other Menu.Menu counts.</summary>
    private static bool IsAutoStartMenu(MainLoopProcess process)
    {
        if (!(process is global::Menu.Menu))
            return false;

        ProcessManager.ProcessID id = process.ID;
        if (id == null)
            return false;
        if (id == ProcessManager.ProcessID.Initialization || id == ProcessManager.ProcessID.Dialog)
            return false;

        return true;
    }

    /// <summary>
    /// Mirrors Menu.SlugcatSelectMenu.StartGame (SlugcatSelectMenu.cs 1730-1783) minus UI concerns:
    /// continue the saved RL game if one exists, else start fresh. Requests Game directly (never SlideShow).
    /// </summary>
    private void StartStoryGame(RainWorld rw, bool forceNew)
    {
        ProcessManager pm = rw.processManager;
        PlayerProgression prog = rw.progression;
        SlugcatStats.Name slugcat = ResolveSlugcat();

        if (!SlugcatStats.SlugcatUnlocked(slugcat, rw))
            log?.LogWarning($"[GameFlow] Slugcat {slugcat} is not unlocked on this save; starting anyway");

        rw.inGameSlugCat = slugcat;
        PlayerGraphics.customColors = null;
        pm.arenaSitting = null;
        prog.currentSaveState = null;
        prog.miscProgressionData.currentlySelectedSinglePlayerSlugcat = slugcat;

        bool continueSaved = !forceNew && prog.IsThereASavedGame(slugcat);
        if (!continueSaved)
            prog.WipeSaveState(slugcat);

        pm.menuSetup.startGameCondition = continueSaved
            ? ProcessManager.MenuSetup.StoryGameInitCondition.Load
            : ProcessManager.MenuSetup.StoryGameInitCondition.New;

        log?.LogInfo($"[GameFlow] Requesting Game as {slugcat} ({(continueSaved ? "Load" : "New")})");
        pm.RequestMainProcessSwitch(ProcessManager.ProcessID.Game);
    }

    private SlugcatStats.Name ResolveSlugcat()
    {
        string name = string.IsNullOrEmpty(SlugcatName) ? "White" : SlugcatName.Trim();
        if (ExtEnumBase.TryParse(typeof(SlugcatStats.Name), name, true, out ExtEnumBase parsed) && parsed is SlugcatStats.Name result)
            return result;

        log?.LogWarning($"[GameFlow] Unknown slugcat '{name}', falling back to White");
        return SlugcatStats.Name.White;
    }

    // ------------------------------------------------------------------ reset

    private void UpdateReset(RainWorld rw, ProcessManager pm)
    {
        if (Time.realtimeSinceStartup - resetStartTime > RESET_TIMEOUT_SECONDS)
        {
            log?.LogError("[GameFlow] RESET timed out");
            FinishReset(false);
            return;
        }

        PlayerProgression prog = rw.progression;

        switch (resetState)
        {
            case ResetState.WaitForSwitchIdle:
                // Let any in-flight switch (e.g. a death respawn) land first.
                if (pm.upcomingProcess != null || pm.IsRunningAnyDialog)
                    return;
                if (prog == null || !prog.progressionLoaded || prog.SaveDataBusy)
                    return;
                resetState = ResetState.Wipe;
                goto case ResetState.Wipe;

            case ResetState.Wipe:
                saves.WipeRLSave(rw);
                resetState = ResetState.WaitForWipe;
                return;

            case ResetState.WaitForWipe:
                // WipeAll writes the wiped file and re-reads it; wait for UserData to go idle.
                if (prog == null || prog.SaveDataBusy || pm.upcomingProcess != null)
                    return;
                resetOldGame = pm.currentMainLoop as RainWorldGame;
                StartStoryGame(rw, forceNew: true);
                lastAutoStartProcess = pm.currentMainLoop;
                lastAutoStartTime = Time.realtimeSinceStartup;
                autoStartAttempts = 1;
                resetState = ResetState.WaitForNewGame;
                return;

            case ResetState.WaitForNewGame:
                if (pm.upcomingProcess != null)
                    return;
                RainWorldGame game = pm.currentMainLoop as RainWorldGame;
                if (game == null || ReferenceEquals(game, resetOldGame) || !game.IsStorySession)
                    return;
                AbstractCreature player = GetPlayer0(pm);
                if (player == null || player.realizedCreature == null)
                    return;
                log?.LogInfo("[GameFlow] RESET complete");
                FinishReset(true);
                return;
        }
    }

    private void FinishReset(bool ok)
    {
        resetState = ResetState.None;
        resetOldGame = null;
        sharedMemory.WriteCommandResult(ok ? SharedMemoryBridge.RESULT_OK : SharedMemoryBridge.RESULT_ERROR);
        sharedMemory.WriteCommand(SharedMemoryBridge.CMD_NONE);
    }

    private void AbortReset(string reason)
    {
        if (resetState == ResetState.None)
            return;
        log?.LogWarning($"[GameFlow] RESET aborted: {reason}");
        FinishReset(false);
    }
}
