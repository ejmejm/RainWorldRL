using System;
using UnityEngine;

/// <summary>
/// Injects RL agent actions into Rain World's input system.
/// Hooks RWInput.PlayerInput to override controller/keyboard input with programmatic values.
/// </summary>
public class InputInjector
{
    private bool installed = false;
    private bool overrideActive = false;

    // Current action state (set by RL agent)
    public bool Jump { get; set; }
    public bool Grab { get; set; }  // Also used for pickup
    public bool Throw { get; set; }
    public int HorizontalAxis { get; set; } // -1 = left, 0 = none, 1 = right
    public int VerticalAxis { get; set; }   // -1 = down, 0 = none, 1 = up

    // Previous frame state for edge detection
    private bool prevJump;
    private bool prevGrab;
    private bool prevThrow;

    public bool IsInstalled => installed;
    public bool IsOverrideActive => overrideActive;

    /// <summary>
    /// Installs hooks into Rain World's input system.
    /// </summary>
    public void Install()
    {
        // Hook the version that takes player number only (older API)
        On.RWInput.PlayerInput_int += RWInput_PlayerInput_int;
        installed = true;
    }

    /// <summary>
    /// Removes hooks from Rain World's input system.
    /// </summary>
    public void Uninstall()
    {
        On.RWInput.PlayerInput_int -= RWInput_PlayerInput_int;
        installed = false;
        overrideActive = false;
    }

    /// <summary>
    /// Enables input override - RL agent takes control of input.
    /// </summary>
    public void EnableOverride()
    {
        overrideActive = true;
    }

    /// <summary>
    /// Disables input override - returns control to player.
    /// </summary>
    public void DisableOverride()
    {
        overrideActive = false;
        ClearInput();
    }

    /// <summary>
    /// Sets the input state from an action byte.
    /// </summary>
    public void SetFromActionByte(byte action)
    {
        Jump = (action & SharedMemoryBridge.ACTION_JUMP) != 0;
        Grab = (action & SharedMemoryBridge.ACTION_GRAB) != 0;
        Throw = (action & SharedMemoryBridge.ACTION_THROW) != 0;

        // Horizontal: bits 3-4, 0=none, 1=left, 2=right
        int hRaw = (action & SharedMemoryBridge.ACTION_HORIZONTAL_MASK) >> 3;
        HorizontalAxis = hRaw == 1 ? -1 : (hRaw == 2 ? 1 : 0);

        // Vertical: bits 5-6, 0=none, 1=down, 2=up
        int vRaw = (action & SharedMemoryBridge.ACTION_VERTICAL_MASK) >> 5;
        VerticalAxis = vRaw == 1 ? -1 : (vRaw == 2 ? 1 : 0);
    }

    /// <summary>
    /// Clears all input state.
    /// </summary>
    public void ClearInput()
    {
        Jump = false;
        Grab = false;
        Throw = false;
        HorizontalAxis = 0;
        VerticalAxis = 0;
    }

    /// <summary>
    /// Updates the previous frame state. Call this after input has been consumed.
    /// </summary>
    public void UpdatePreviousState()
    {
        prevJump = Jump;
        prevGrab = Grab;
        prevThrow = Throw;
    }

    /// <summary>
    /// Hook for RWInput.PlayerInput(int) - replaces game input with RL actions.
    /// </summary>
    private Player.InputPackage RWInput_PlayerInput_int(
        On.RWInput.orig_PlayerInput_int orig,
        int playerNumber)
    {
        if (!overrideActive || playerNumber != 0)
        {
            // RL mode not active or not controlling this player, use original input
            return orig(playerNumber);
        }

        // Build input package from RL action state
        Player.InputPackage input = new Player.InputPackage();

        // Directional input
        input.x = HorizontalAxis;
        input.y = VerticalAxis;

        // Analog direction (normalized)
        if (HorizontalAxis != 0 || VerticalAxis != 0)
        {
            Vector2 dir = new Vector2(HorizontalAxis, VerticalAxis).normalized;
            input.analogueDir = dir;
        }
        else
        {
            input.analogueDir = Vector2.zero;
        }

        // Button states
        input.jmp = Jump;
        input.pckp = Grab;
        input.thrw = Throw;

        // Map buttons - Rain World uses specific mappings
        input.mp = false; // Map button

        // Downward diagonal for crawling
        input.downDiagonal = VerticalAxis < 0 && HorizontalAxis != 0 ? HorizontalAxis : 0;

        return input;
    }
}

