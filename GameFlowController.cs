using System;
using System.Collections.Generic;
using System.Reflection;
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
///
/// Fade skipping (RL mode on only): every RequestMainProcessSwitch fade-out is clamped to
/// SKIP_SCREEN_FADE_SECONDS, and the fade-from-black that a new RainWorldGame starts with is collapsed in
/// PostSwitchMainProcess (see <see cref="SkipGameFadeIn"/>), so the agent sees the room from its first step
/// instead of 2-5 s of black frames.
///
/// Death flow: Player.Die (Player.cs:6690) -> RainWorldGame.GameOver (RainWorldGame.cs:1465) only
/// puts the HUD into "game over" mode; the game stays current with the dead slugcat until a key is
/// pressed, 40 ticks later at the earliest (HUD.TextPrompt.Update, TextPrompt.cs:205-245, reads
/// Rewired directly so injected RL input cannot press it). In RL mode <see cref="UpdateGameOverAdvance"/>
/// presses that key for the agent: after the same 40 ticks it calls RainWorldGame.GoToDeathScreen
/// (RainWorldGame.cs:1914), which saves death-persistent data and requests DeathScreen - rewritten to
/// Game/Load by the skip hook above, so the slugcat respawns in the start-of-cycle shelter.
///
/// KILL_PLAYER (debug command): <see cref="KillPlayer"/> calls Player.Die on a realized player 0 and
/// the ordinary death flow above takes over. The ack only confirms the kill was applied.
///
/// ENTER_SHELTER (debug command): <see cref="EnterShelter"/> sets player 0's food and sends it through the
/// entrance pipe of its den shelter; the game's own shelter logic then decides whether and how it sleeps.
///
/// HOP_ROOM (debug command): <see cref="HopRoom"/> sends player 0 out through one of its room's exits into the
/// neighbouring room, the way walking through that pipe would.
///
/// SWITCH_REGION (debug command): <see cref="SwitchRegion"/> sends player 0 into a region gate room of its region;
/// <see cref="UpdateRegionSwitch"/> then starts the gate (skipping its karma check), lets the game load the next region
/// and sends the slugcat out of the gate room into it.
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

    private enum SwitchState
    {
        None,
        WaitForGateRoom,
        WaitForWorld,
    }

    /// <summary>Fade-to-black length for every process switch while RL mode is on (game default 0.45 s).</summary>
    private const float SKIP_SCREEN_FADE_SECONDS = 0.05f;
    /// <summary>Remaining fade-from-black (ProcessManager.blackFadeTime) forced on a new RainWorldGame.</summary>
    private const float GAME_FADE_IN_SECONDS = 0.05f;
    /// <summary>
    /// fadeToBlack value forced on a new RainWorldGame: small but positive, so ProcessManager.Update still
    /// takes its "fadeToBlack > 0" branch once and tears the fade sprite + Loading label down cleanly.
    /// </summary>
    private const float GAME_FADE_IN_REMAINDER = 0.001f;
    private const float AUTO_START_RETRY_SECONDS = 3f;
    private const int AUTO_START_MAX_ATTEMPTS = 5;
    private const float RESET_TIMEOUT_SECONDS = 55f;

    // ProcessManager keeps the fade-in timing in private fields and has no hookable getters for
    // MainLoopProcess.FadeInTime / InitialBlackSeconds (ProcessManager.cs 205-207, 1158-1159).
    private static readonly FieldInfo pmBlackDelay =
        typeof(ProcessManager).GetField("blackDelay", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo pmBlackFadeTime =
        typeof(ProcessManager).GetField("blackFadeTime", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly MethodInfo pmUpdateFade =
        typeof(ProcessManager).GetMethod("UpdateFade", BindingFlags.Instance | BindingFlags.NonPublic);

    /// <summary>
    /// Game ticks between the game-over prompt appearing and the restart key being accepted
    /// (TextPrompt.EnterGameOverMode sets restartNotAllowed = 40, TextPrompt.cs:497). Mirrored here
    /// because that field is private.
    /// </summary>
    private const int GAME_OVER_ADVANCE_TICKS = 40;

    private readonly SaveRedirector saves;
    private readonly SharedMemoryBridge sharedMemory;
    private readonly ManualLogSource log;

    private bool hooksInstalled = false;
    private bool fadeFieldsMissingLogged = false;
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

    // Game-over auto-advance bookkeeping (per RainWorldGame instance)
    private RainWorldGame gameOverGame = null;
    private int gameOverClock = 0;
    private bool gameOverAdvanced = false;

    // SWITCH_REGION bookkeeping
    private SwitchState switchState = SwitchState.None;
    private RainWorldGame switchGame = null;
    private AbstractRoom switchGateRoom = null;
    private RegionGate switchGate = null;

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

        if (state == FlowState.On && self.currentMainLoop is RainWorldGame)
            SkipGameFadeIn(self);

        if (VerboseLogging)
            log?.LogInfo($"[GameFlow] Process switched to {ID} (now {self.currentMainLoop?.ID})");
    }

    /// <summary>
    /// Collapses the fade-from-black every RainWorldGame starts with.
    ///
    /// ProcessManager.PostSwitchMainProcess copies RainWorldGame.FadeInTime (New: 2 s; Load:
    /// SaveState.SlowFadeIn, >= 0.8 s; RainWorldGame.cs 653-674) and InitialBlackSeconds (New: 3 s, 5.5 s as
    /// Red; Load: 0.75 s; RainWorldGame.cs 677-698) into its private blackFadeTime / blackDelay
    /// (ProcessManager.cs 1158-1159). ProcessManager.Update then holds fadeToBlack at 1 for blackDelay
    /// seconds of Time.deltaTime and lerps it down over blackFadeTime (ProcessManager.cs 766-782); because
    /// the step controller freezes time between steps this only advances during steps, so the agent saw
    /// ~50 black steps. Called right after that copy, this zeroes the delay, leaves a near-zero remainder so
    /// the next unpaused Update drives fadeToBlack below 0 and removes the fade sprite plus the "Loading..."
    /// label through the game's own code path, and re-applies the sprite alpha immediately so even the frame
    /// rendered before that Update already shows the room.
    /// </summary>
    private void SkipGameFadeIn(ProcessManager pm)
    {
        if (pmBlackDelay == null || pmBlackFadeTime == null)
        {
            if (!fadeFieldsMissingLogged)
            {
                log?.LogWarning("[GameFlow] ProcessManager.blackDelay/blackFadeTime not found; cannot skip the game fade-in");
                fadeFieldsMissingLogged = true;
            }
            return;
        }

        try
        {
            pmBlackDelay.SetValue(pm, 0f);
            pmBlackFadeTime.SetValue(pm, GAME_FADE_IN_SECONDS);
            if (pm.fadeToBlack > GAME_FADE_IN_REMAINDER)
                pm.fadeToBlack = GAME_FADE_IN_REMAINDER;

            if (pmUpdateFade != null)
                pmUpdateFade.Invoke(pm, null);
            else if (pm.fadeSprite != null)
                pm.fadeSprite.alpha = 0f;

            if (VerboseLogging)
                log?.LogInfo("[GameFlow] Skipped game fade-in");
        }
        catch (Exception ex)
        {
            log?.LogWarning($"[GameFlow] Could not skip the game fade-in: {ex.Message}");
        }
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
        }

        // Nobody watches the screen in RL mode: make every switch (auto-start, RESET, respawn, the game's
        // own Game requests such as RestartGame) fade out as fast as possible. The default is 0.45 s
        // (ProcessManager.cs 860-863); ActualProcessSwitch stores it in blackFadeTime (ProcessManager.cs 885).
        if (state == FlowState.On && fadeOutSeconds > SKIP_SCREEN_FADE_SECONDS)
            fadeOutSeconds = SKIP_SCREEN_FADE_SECONDS;

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

    /// <summary>
    /// Debug KILL_PLAYER command: kills player 0 right now, on the calling (main) thread, via
    /// Player.Die (Player.cs:6690). That runs RainWorldGame.GameOver (HUD game-over prompt) and
    /// Creature.Die (sets Creature.dead and the abstract state's alive = false, Creature.cs:926/976),
    /// so the next step reports the PLAYER_DEAD edge; the respawn happens later through
    /// <see cref="UpdateGameOverAdvance"/>. Returns true only when the player is dead afterwards.
    /// Returns false (caller reports ERROR) when RL mode is not fully on, a reset is in progress,
    /// no story game with a realized, in-room player 0 is current, or the player is already dead.
    /// </summary>
    public bool KillPlayer(RainWorld rw)
    {
        if (state != FlowState.On || resetState != ResetState.None || rw == null)
        {
            log?.LogWarning("[GameFlow] KILL_PLAYER rejected: RL mode not fully on or a RESET is in progress");
            return false;
        }

        ProcessManager pm = rw.processManager;
        if (!IsInStoryGame(pm))
        {
            log?.LogWarning("[GameFlow] KILL_PLAYER rejected: no story game is current (or a process switch is pending)");
            return false;
        }

        RainWorldGame game = pm.currentMainLoop as RainWorldGame;
        AbstractCreature abstractPlayer = GetPlayer0(pm);
        Player player = abstractPlayer?.realizedCreature as Player;
        if (player == null || player.room == null)
        {
            log?.LogWarning("[GameFlow] KILL_PLAYER rejected: player 0 is not realized in a room");
            return false;
        }
        if (player.dead || game.GameOverModeActive)
        {
            log?.LogWarning("[GameFlow] KILL_PLAYER rejected: player 0 is already dead");
            return false;
        }

        SaveState save = game.GetStorySession?.saveState;
        log?.LogInfo($"[GameFlow] KILL_PLAYER: killing player 0 in room {player.room.abstractRoom?.name} " +
                     $"(cycle {save?.cycleNumber}, karma {save?.deathPersistentSaveData?.karma})");
        player.Die();

        if (!player.dead)
        {
            // Player.Die returns early without dying when setupValues.invincibility is set.
            log?.LogWarning("[GameFlow] KILL_PLAYER: Player.Die() did not kill the player (invincibility?)");
            return false;
        }
        return true;
    }

    /// <summary>
    /// Debug ENTER_SHELTER command: sets player 0's food to <paramref name="food"/> pips and sends it into its den
    /// shelter (SaveState.denPosition, or SaveState.GetFinalFallbackShelter while the den is not a shelter, e.g. on a
    /// fresh save) through the entrance pipe (<see cref="SendThroughShortcut"/>), clearing Player.stillInStartShelter
    /// (Player.cs:6963). The game's own shelter logic (Player.cs:5719-5787) then decides the sleep. Returns false
    /// (caller reports ERROR) when <see cref="MovablePlayer0"/> rejects or the den shelter is not in the current region.
    /// </summary>
    public bool EnterShelter(RainWorld rw, int food)
    {
        Player player = MovablePlayer0(rw, "ENTER_SHELTER");
        if (player == null)
            return false;

        RainWorldGame game = player.room.game;
        SaveState save = game.GetStorySession.saveState;
        AbstractRoom shelter = string.IsNullOrEmpty(save.denPosition) ? null : game.world.GetAbstractRoom(save.denPosition);
        if (shelter == null || !shelter.shelter)
            shelter = game.world.GetAbstractRoom(SaveState.GetFinalFallbackShelter(save.saveStateNumber));
        if (shelter == null || !shelter.shelter)
        {
            log?.LogWarning($"[GameFlow] ENTER_SHELTER rejected: the den shelter is not in region {game.world.name}");
            return false;
        }

        player.playerState.foodInStomach = Mathf.Clamp(food, 0, player.MaxFoodInStomach);
        player.playerState.quarterFoodPoints = 0;
        log?.LogInfo($"[GameFlow] ENTER_SHELTER: player 0 {player.room.abstractRoom.name} -> {shelter.name} with {player.FoodInStomach} food");

        // Node 0 is a shelter's entrance (Player.cs:5768).
        SendThroughShortcut(player, shelter, 0);
        return true;
    }

    /// <summary>
    /// Debug HOP_ROOM command: sends player 0 through usable exit <paramref name="exit"/> % (number of usable exits) of its
    /// room (<see cref="UsableExits"/>) into the neighbouring room. It arrives from the pipe leading back, the entrance
    /// node ShortcutHandler.Update picks for a room exit (ShortcutHandler.cs:201). Returns false (caller reports ERROR)
    /// when <see cref="MovablePlayer0"/> rejects or the room has no usable exit.
    /// </summary>
    public bool HopRoom(RainWorld rw, int exit)
    {
        Player player = MovablePlayer0(rw, "HOP_ROOM");
        return player != null && HopThroughExit(player, exit, "HOP_ROOM");
    }

    private bool HopThroughExit(Player player, int exit, string command)
    {
        World world = player.room.game.world;
        AbstractRoom room = player.room.abstractRoom;
        List<int> exits = UsableExits(world, room);
        if (exits.Count == 0)
        {
            log?.LogWarning($"[GameFlow] {command}: room {room.name} has no usable exit");
            return false;
        }

        int node = exits[exit % exits.Count];
        AbstractRoom dest = world.GetAbstractRoom(room.connections[node]);
        log?.LogInfo($"[GameFlow] {command}: player 0 {room.name} -> {dest.name} (exit node {node}, {exits.Count} usable)");
        SendThroughShortcut(player, dest, dest.ExitIndex(room.index));
        return true;
    }

    /// <summary>
    /// Debug SWITCH_REGION command: takes player 0 through a region gate of its region (usable gate
    /// <paramref name="gate"/> % count, in world-file order) into the neighbouring region. The slugcat is sent into the
    /// gate room through its pipe on this region's side and holds still (Player.NullController) until it has left that
    /// room; <see cref="UpdateRegionSwitch"/> does the rest. Usable gates lead to a region the overworld knows and are not
    /// mid-cycle. Returns false (caller reports ERROR) when <see cref="MovablePlayer0"/> rejects or no gate is usable.
    /// </summary>
    public bool SwitchRegion(RainWorld rw, int gate)
    {
        Player player = MovablePlayer0(rw, "SWITCH_REGION");
        if (player == null)
            return false;

        RainWorldGame game = player.room.game;
        var gates = new List<AbstractRoom>();
        foreach (AbstractRoom room in game.world.abstractRooms)
        {
            RegionGate realized = room.realizedRoom?.regionGate;
            if (room.gate && UsableExits(game.world, room).Count > 0 && GateDestination(game, room) != null &&
                (realized == null || realized.mode == RegionGate.Mode.MiddleClosed))
                gates.Add(room);
        }
        if (gates.Count == 0)
        {
            log?.LogWarning($"[GameFlow] SWITCH_REGION rejected: region {game.world.name} has no usable gate");
            return false;
        }

        AbstractRoom gateRoom = gates[gate % gates.Count];
        log?.LogInfo($"[GameFlow] SWITCH_REGION: player 0 {player.room.abstractRoom.name} -> {gateRoom.name} -> region {GateDestination(game, gateRoom)}");
        SendThroughShortcut(player, gateRoom, UsableExits(game.world, gateRoom)[0]);
        player.controller = new Player.NullController();
        switchState = SwitchState.WaitForGateRoom;
        switchGame = game;
        switchGateRoom = gateRoom;
        return true;
    }

    /// <summary>
    /// Advances a SWITCH_REGION. Once player 0 is out of the pipe in the gate room, does what RegionGate.Update does when a
    /// player has stood still in the gate's zone with enough karma and energy (RegionGate.cs:317-323): close the airlock
    /// behind the player and call OverWorld.GateRequestsSwitchInitiation, which loads the next region's world. When it has
    /// loaded, OverWorld.WorldLoaded (OverWorld.cs:488-576) has moved the gate room and everything in it into the new world
    /// and called RegionGate.NewWorldLoaded; the slugcat then leaves through the gate room's far exit, the only one
    /// connected in the new world. Holding the slugcat still until then keeps it in the gate room while the world loads
    /// (it would otherwise leave through the pipe it came from: the airlock only closes it in once it is inside, and
    /// WorldLoaded needs the room realized) and away from the old side's pipe, DISCONNECTED in the new world.
    /// </summary>
    private void UpdateRegionSwitch(ProcessManager pm)
    {
        if (switchState == SwitchState.None)
            return;

        RainWorldGame game = pm.currentMainLoop as RainWorldGame;
        Player player = GetPlayer0(pm)?.realizedCreature as Player;
        if (!ReferenceEquals(game, switchGame))
        {
            EndRegionSwitch(null, "the game was restarted");
            return;
        }
        if (player == null || player.dead)
        {
            EndRegionSwitch(player, "player 0 died");
            return;
        }
        if (player.room == null)
            return; // still in the pipe

        if (switchState == SwitchState.WaitForGateRoom)
        {
            RegionGate gate = player.room.regionGate;
            if (player.room.abstractRoom != switchGateRoom || gate == null || gate.mode != RegionGate.Mode.MiddleClosed)
            {
                EndRegionSwitch(player, $"player 0 arrived in {player.room.abstractRoom.name}, not at a closed gate in {switchGateRoom.name}");
                return;
            }
            gate.letThroughDir = player.abstractCreature.pos.x < player.room.TileWidth / 2;
            gate.mode = RegionGate.Mode.ClosingAirLock;
            gate.goalDoorPositions[gate.letThroughDir ? 0 : 2] = 1f;
            game.overWorld.GateRequestsSwitchInitiation(gate);
            gate.waitingForWorldLoader = true;
            switchGate = gate;
            switchState = SwitchState.WaitForWorld;
            return;
        }

        if (switchGate.waitingForWorldLoader)
            return;
        if (player.room.regionGate != switchGate)
        {
            EndRegionSwitch(player, $"player 0 left the gate room for {player.room.abstractRoom.name}");
            return;
        }
        EndRegionSwitch(player, null);
        HopThroughExit(player, 0, "SWITCH_REGION");
    }

    private void EndRegionSwitch(Player player, string failure)
    {
        if (failure != null)
            log?.LogWarning($"[GameFlow] SWITCH_REGION abandoned: {failure}");
        if (player?.controller is Player.NullController)
            player.controller = null;
        switchState = SwitchState.None;
        switchGame = null;
        switchGateRoom = null;
        switchGate = null;
    }

    /// <summary>
    /// Player 0 when a debug command may send it to another room: RL mode fully on, no RESET, a story game current,
    /// player 0 alive in a room and not already on its way out (into a pipe, through a SWITCH_REGION, or through a gate
    /// whose next region is loading). Otherwise logs why and returns null.
    /// </summary>
    private Player MovablePlayer0(RainWorld rw, string command)
    {
        ProcessManager pm = rw?.processManager;
        if (state != FlowState.On || resetState != ResetState.None || !IsInStoryGame(pm))
        {
            log?.LogWarning($"[GameFlow] {command} rejected: RL mode not fully on, a RESET is in progress or no story game is current");
            return null;
        }

        RainWorldGame game = (RainWorldGame)pm.currentMainLoop;
        Player player = GetPlayer0(pm)?.realizedCreature as Player;
        if (player == null || player.dead || game.GameOverModeActive)
        {
            log?.LogWarning($"[GameFlow] {command} rejected: player 0 is not alive");
            return null;
        }
        if (player.room == null || player.enteringShortCut.HasValue || switchState != SwitchState.None ||
            (player.room.regionGate != null && player.room.regionGate.waitingForWorldLoader))
        {
            log?.LogWarning($"[GameFlow] {command} rejected: player 0 is on its way between rooms (in or entering a pipe, " +
                            "or a region switch is under way)");
            return null;
        }
        return player;
    }

    /// <summary>
    /// Takes player 0 out of its room into <paramref name="dest"/>, to come out of the pipe at node
    /// <paramref name="entranceNode"/>: it leaves with everything it holds like Creature.SuckedIntoShortCut
    /// (Creature.cs:1050-1063) and waits between rooms like after a room exit (ShortcutHandler.cs:198-203), so the game
    /// loads the room, moves the camera and spits it out there.
    /// </summary>
    private static void SendThroughShortcut(Player player, AbstractRoom dest, int entranceNode)
    {
        // A room script's controller stays with the room being left (e.g. the NullController of the fresh-save
        // intro, RoomSpecificScript.cs:102, which would otherwise keep the agent's input away for good).
        player.controller = null;
        Room room = player.room;
        var vessel = new ShortcutHandler.ShortCutVessel(new RWCustom.IntVector2(0, 0), player, dest, 0) { entranceNode = entranceNode };
        foreach (AbstractPhysicalObject obj in player.abstractCreature.GetAllConnectedObjects())
        {
            if (obj.realizedObject == null)
                continue;
            if (obj.realizedObject is Creature creature)
            {
                creature.inShortcut = true;
                creature.inShortcutVessel = vessel;
            }
            room.RemoveObject(obj.realizedObject);
        }
        room.game.shortcuts.betweenRoomsWaitingLobby.Add(vessel);
    }

    /// <summary>
    /// Exit nodes of <paramref name="room"/> that lead into a room of <paramref name="world"/> with a pipe back. Exits
    /// that lead nowhere (DISCONNECTED in the world file, e.g. a gate room's far side) are left out.
    /// </summary>
    private static List<int> UsableExits(World world, AbstractRoom room)
    {
        var exits = new List<int>();
        for (int i = 0; i < room.connections.Length; i++)
        {
            AbstractRoom dest = room.connections[i] > -1 ? world.GetAbstractRoom(room.connections[i]) : null;
            if (dest != null && dest.ExitIndex(room.index) > -1)
                exits.Add(i);
        }
        return exits;
    }

    /// <summary>
    /// The region <paramref name="gateRoom"/> leads to from the current world, worked out like
    /// OverWorld.GateRequestsSwitchInitiation (OverWorld.cs:328-343), or null if that is no region the overworld knows
    /// (the switch would then never load anything).
    /// </summary>
    private static string GateDestination(RainWorldGame game, AbstractRoom gateRoom)
    {
        string current = Region.GetVanillaEquivalentRegionAcronym(game.world.name);
        string[] parts = gateRoom.name.Split('_');
        if (parts.Length != 3)
            return null;
        string other = parts[1] != current ? parts[1] : parts[2];
        if (other == current)
            return null;
        other = Region.GetProperRegionAcronym(game.TimelinePoint, other);
        return game.overWorld.GetRegion(other) != null ? other : null;
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
                    {
                        UpdateReset(rw, pm);
                    }
                    else
                    {
                        UpdateAutoNavigate(rw, pm);
                        UpdateGameOverAdvance(pm);
                        UpdateRegionSwitch(pm);
                    }
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

    // ------------------------------------------------------------------ game over -> respawn

    /// <summary>
    /// Stands in for the "press SPACE to restart" key while the HUD is in game-over mode
    /// (RainWorldGame.GameOverModeActive). TextPrompt.Update only accepts that key once
    /// restartNotAllowed (40 ticks) has run down and reads the keyboard through Rewired, which the
    /// RL input override does not reach - so without this the agent would be stuck stepping a dead
    /// slugcat forever. After the same 40 game ticks (RainWorldGame.clock) this calls
    /// GoToDeathScreen exactly once per game instance; the skip hook turns the DeathScreen request
    /// into Game/Load, i.e. a respawn.
    /// </summary>
    private void UpdateGameOverAdvance(ProcessManager pm)
    {
        RainWorldGame game = pm.currentMainLoop as RainWorldGame;
        if (game == null || !game.IsStorySession || !game.GameOverModeActive)
        {
            gameOverGame = null;
            return;
        }

        if (!ReferenceEquals(game, gameOverGame))
        {
            gameOverGame = game;
            gameOverClock = game.clock;
            gameOverAdvanced = false;
            log?.LogInfo($"[GameFlow] Game over prompt active; advancing to the death screen in {GAME_OVER_ADVANCE_TICKS} ticks");
            return;
        }

        if (gameOverAdvanced || pm.upcomingProcess != null)
            return;
        if (game.clock - gameOverClock < GAME_OVER_ADVANCE_TICKS)
            return;

        gameOverAdvanced = true;
        log?.LogInfo("[GameFlow] Game over -> GoToDeathScreen (DeathScreen is skipped; respawning)");
        game.GoToDeathScreen();
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
