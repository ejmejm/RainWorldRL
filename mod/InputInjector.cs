using System;
using RWCustom;
using UnityEngine;

/// <summary>
/// Injects the RL agent's raw key presses into Rain World's input system.
///
/// The agent's action is a bitfield of held keys (<see cref="SharedMemoryBridge.KEY_LEFT"/> ...),
/// any combination at once. While the override is active (RL mode on, human override off) the
/// following game entry points are replaced so that every consumer of player-0 input sees the
/// agent's keys instead of the keyboard/controller:
///
///   * RWInput.PlayerInput(int)   - the Player's per-tick input (Player.cs:6059). The InputPackage is
///                                  built the way RWInput.PlayerInputLogic builds it for a keyboard
///                                  (RWInput.cs:185-275): x/y from the direction keys (opposite keys
///                                  cancel), analogueDir normalised, downDiagonal when down + a side key.
///   * RWInput.PlayerUIInput(int) - menu/dialog input (Menu.Menu.Update, Menu.cs:321). Only replaced
///                                  while a RainWorldGame is current or a Menu.Dialog side process is
///                                  running, so in-game prompts (Dialog, ArenaOverlay, ...) receive the
///                                  agent's keys but the auto-navigated menus between games are left to
///                                  GameFlowController. Mapping mirrors the UI category (RWInput.cs:209-223):
///                                  jump -> submit, throw -> cancel, map -> mp, directions -> x/y.
///   * Options.ControlSetup.GetButton(int) for player 0 - direct Rewired reads that bypass RWInput:
///                                  the game-over "press X to restart" prompt (HUD/TextPrompt.cs:230),
///                                  map fast-forward (RoomCamera.cs:966), chat-log fast display
///                                  (MoreSlugcats/ChatLogDisplay.cs:201). Same scope as PlayerUIInput.
///   * RWInput.CheckPauseButton(int, bool) - always false while the override is active, so neither the
///                                  Rewired Pause action nor the Escape fallback (RWInput.cs:138-142) can
///                                  open the pause menu (RainWorldGame.cs:2227) or back out of menus.
///
/// When the override is off (no client, or F10 human override) every hook calls the original.
/// </summary>
public class InputInjector
{
    // Rewired action ids (RewiredConsts/Action.cs)
    private const int ACTION_JUMP = 0;
    private const int ACTION_MOVE_HORIZONTAL = 1;
    private const int ACTION_MOVE_VERTICAL = 2;
    private const int ACTION_TAKE = 3;
    private const int ACTION_THROW = 4;
    private const int ACTION_PAUSE = 5;
    private const int ACTION_UI_SUBMIT = 8;
    private const int ACTION_UI_CANCEL = 9;
    private const int ACTION_MAP = 11;
    private const int ACTION_UI_CHEAT_HOLD_RIGHT = 13; // read as "mp" by the UI category
    private const int ACTION_SPECIAL = 34;

    private bool installed = false;
    private bool overrideActive = false;

    /// <summary>Currently held keys (SharedMemoryBridge.KEY_* bits).</summary>
    public uint ActionBits { get; private set; }

    public bool Left => (ActionBits & SharedMemoryBridge.KEY_LEFT) != 0;
    public bool Right => (ActionBits & SharedMemoryBridge.KEY_RIGHT) != 0;
    public bool Up => (ActionBits & SharedMemoryBridge.KEY_UP) != 0;
    public bool Down => (ActionBits & SharedMemoryBridge.KEY_DOWN) != 0;
    public bool Jump => (ActionBits & SharedMemoryBridge.KEY_JUMP) != 0;
    public bool Grab => (ActionBits & SharedMemoryBridge.KEY_GRAB) != 0;
    public bool Throw => (ActionBits & SharedMemoryBridge.KEY_THROW) != 0;
    public bool Map => (ActionBits & SharedMemoryBridge.KEY_MAP) != 0;
    public bool Special => (ActionBits & SharedMemoryBridge.KEY_SPECIAL) != 0;

    /// <summary>-1 left, 0 none/both, +1 right.</summary>
    public int HorizontalAxis => (Right ? 1 : 0) - (Left ? 1 : 0);

    /// <summary>-1 down, 0 none/both, +1 up.</summary>
    public int VerticalAxis => (Up ? 1 : 0) - (Down ? 1 : 0);

    /// <summary>Installs the input hooks. Safe to call once; hooks are inert until <see cref="EnableOverride"/>.</summary>
    public void Install()
    {
        if (installed)
            return;
        On.RWInput.PlayerInput_int += RWInput_PlayerInput_int;
        On.RWInput.PlayerUIInput_int += RWInput_PlayerUIInput_int;
        On.RWInput.CheckPauseButton_int_bool += RWInput_CheckPauseButton_int_bool;
        On.Options.ControlSetup.GetButton += ControlSetup_GetButton;
        installed = true;
    }

    public void Uninstall()
    {
        if (!installed)
            return;
        On.RWInput.PlayerInput_int -= RWInput_PlayerInput_int;
        On.RWInput.PlayerUIInput_int -= RWInput_PlayerUIInput_int;
        On.RWInput.CheckPauseButton_int_bool -= RWInput_CheckPauseButton_int_bool;
        On.Options.ControlSetup.GetButton -= ControlSetup_GetButton;
        installed = false;
        overrideActive = false;
    }

    /// <summary>Agent takes control of player 0's input (and the pause button is blocked).</summary>
    public void EnableOverride()
    {
        overrideActive = true;
    }

    /// <summary>Returns control to the keyboard/controller and releases all keys.</summary>
    public void DisableOverride()
    {
        overrideActive = false;
        ClearInput();
    }

    /// <summary>Sets the held keys from the action_bits field.</summary>
    public void SetFromActionBits(uint bits)
    {
        ActionBits = bits & SharedMemoryBridge.KEY_ALL_MASK;
    }

    /// <summary>Releases every key.</summary>
    public void ClearInput()
    {
        ActionBits = 0;
    }

    // ------------------------------------------------------------------ prompt detection

    /// <summary>
    /// True when an in-game overlay is waiting for a key press: a Menu.Dialog side process
    /// (ProcessManager.IsRunningAnyDialog), the game-over "press X to restart" prompt
    /// (RainWorldGame.GameOverModeActive -> HUD.TextPrompt.gameOverMode), or an open pause menu.
    /// Cheap: a few null checks, no allocation. Conversation text (HUD.DialogBox) is deliberately
    /// excluded: it advances on a timer and never waits for input (HUD/DialogBox.cs:149-212).
    /// </summary>
    public static bool IsPromptAwaitingInput(RainWorld rw)
    {
        ProcessManager pm = rw?.processManager;
        if (pm == null)
            return false;
        if (pm.IsRunningAnyDialog)
            return true;
        RainWorldGame game = pm.currentMainLoop as RainWorldGame;
        if (game == null)
            return false;
        if (game.pauseMenu != null)
            return true;
        try
        {
            return game.GameOverModeActive;
        }
        catch
        {
            return false; // cameras not built yet
        }
    }

    /// <summary>
    /// If the pause menu is open while the agent is in control (e.g. a human paused, then released
    /// the F10 override), dismiss it the way its CONTINUE button does (PauseMenu.Singal "CONTINUE").
    /// Returns true if a close was requested.
    /// </summary>
    public bool DismissPauseMenu(RainWorld rw)
    {
        if (!overrideActive)
            return false;
        RainWorldGame game = rw?.processManager?.currentMainLoop as RainWorldGame;
        if (game?.pauseMenu == null)
            return false;
        try
        {
            game.pauseMenu.Singal(null, "CONTINUE");
            return true;
        }
        catch
        {
            return false;
        }
    }

    // ------------------------------------------------------------------ package building

    /// <summary>
    /// Builds an InputPackage from the held keys, following RWInput.PlayerInputLogic's keyboard path
    /// (RWInput.cs:185-275). <paramref name="ui"/>: menu/dialog input, the UI category (RWInput.cs:209-223):
    /// jump = submit, throw = cancel, map = mp; pckp/spec are not part of it and stay false.
    /// </summary>
    private Player.InputPackage BuildInput(bool ui)
    {
        int x = HorizontalAxis;
        int y = VerticalAxis;

        Player.InputPackage input = new Player.InputPackage(
            gamePad: false,
            controllerType: Options.ControlSetup.Preset.KeyboardSinglePlayer,
            x: x, y: y,
            jmp: Jump, thrw: Throw, pckp: Grab && !ui, mp: Map,
            crouchToggle: false, // never set by any input path in v1.11.8 (only RWInput.cs:319, always false)
            spec: Special && !ui);

        input.analogueDir = (x != 0 || y != 0) ? new Vector2(x, y).normalized : Vector2.zero;

        // Down + a side key = downward diagonal (crawl / roll direction). Matches RWInput.cs:252-262 (MMF path).
        input.downDiagonal = (y < 0 && x != 0) ? x : 0;
        return input;
    }

    // ------------------------------------------------------------------ hooks

    /// <summary>
    /// UI-side injection applies only where an in-game prompt could be consuming it: a RainWorldGame
    /// is current, or a Menu.Dialog side process is running (over the game or over a menu, since the
    /// auto-navigation waits for dialogs to close). Plain menus keep the keyboard so GameFlowController's
    /// programmatic navigation is not fought by stray agent presses.
    /// </summary>
    private bool UIInjectionApplies()
    {
        if (!overrideActive)
            return false;
        ProcessManager pm = Custom.rainWorld?.processManager;
        if (pm == null)
            return false;
        return pm.currentMainLoop is RainWorldGame || pm.IsRunningAnyDialog;
    }

    private Player.InputPackage RWInput_PlayerInput_int(On.RWInput.orig_PlayerInput_int orig, int playerNumber)
    {
        if (!overrideActive || playerNumber != 0)
            return orig(playerNumber);
        return BuildInput(ui: false);
    }

    private Player.InputPackage RWInput_PlayerUIInput_int(On.RWInput.orig_PlayerUIInput_int orig, int playerNumber)
    {
        // playerNumber is -1 ("any player") from Menu.Menu; in RL mode there is only player 0.
        if (!UIInjectionApplies() || playerNumber > 0)
            return orig(playerNumber);
        return BuildInput(ui: true);
    }

    private bool RWInput_CheckPauseButton_int_bool(On.RWInput.orig_CheckPauseButton_int_bool orig, int playerNumber, bool inMenu)
    {
        if (overrideActive)
            return false; // the agent can never pause / back out
        return orig(playerNumber, inMenu);
    }

    private bool ControlSetup_GetButton(On.Options.ControlSetup.orig_GetButton orig, Options.ControlSetup self, int actionID)
    {
        if (!UIInjectionApplies() || self == null || self.index != 0)
            return orig(self, actionID);

        switch (actionID)
        {
            case ACTION_JUMP:
            case ACTION_UI_SUBMIT:
                return Jump;
            case ACTION_THROW:
            case ACTION_UI_CANCEL:
                return Throw;
            case ACTION_TAKE:
                return Grab;
            case ACTION_MAP:
            case ACTION_UI_CHEAT_HOLD_RIGHT:
                return Map;
            case ACTION_SPECIAL:
                return Special;
            case ACTION_PAUSE:
                // Pause is never pressable. The one place the game asks for it as a "continue" key is
                // the game-over prompt on non-default / gamepad bindings (HUD/TextPrompt.cs:230); let MAP
                // serve there too so the prompt is always dismissable. The pause menu cannot open while
                // the prompt is up (RainWorldGame.cs:2228 checks !gameOverMode) and CheckPauseButton is
                // separately blocked, so this cannot pause the game.
                return Map && IsGameOverPrompt();
            default:
                return false; // the real keyboard is ignored for player 0 while the agent is in control
        }
    }

    private static bool IsGameOverPrompt()
    {
        try
        {
            RainWorldGame game = Custom.rainWorld?.processManager?.currentMainLoop as RainWorldGame;
            return game != null && game.GameOverModeActive;
        }
        catch
        {
            return false;
        }
    }
}
