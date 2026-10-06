using System;
using System.IO;
using BepInEx;
using BepInEx.Logging;

/// <summary>
/// Redirects Rain World's progression save file to an isolated "RL save" directory while
/// RL mode is active, so training never touches the player's real campaign.
///
/// Mechanism: <c>PlayerProgression(RainWorld, bool, bool, string overrideBaseDir)</c> uses
/// <c>overrideBaseDir</c> in place of <c>Application.persistentDataPath</c> when mounting the
/// <c>sav</c>/<c>sav2</c>/<c>sav3</c> file (PlayerProgression.cs: ctor + Update -> UserData.Mount).
/// We hook that constructor and substitute our directory whenever <see cref="Active"/> is set, so
/// every progression the game constructs in RL mode (including the game's own
/// <c>RainWorld.ReloadProgression()</c>) lands in the RL directory.
///
/// Swapping at runtime mirrors Menu.OptionsMenu's save-slot change:
/// <c>progression.Destroy(oldSlot); progression = new PlayerProgression(rw, true, false);</c>
/// then wait for <c>progression.progressionLoaded</c> (the load is asynchronous and driven by
/// <c>RainWorld.Update -> progression.Update()</c>).
///
/// Caveats (not redirected, still live in persistentDataPath):
///   * <c>options</c> (settings file) - shared with the normal profile.
///   * <c>SJ_&lt;slot&gt;</c> (per-slot screenshot/journal folder) - <c>WipeAll</c>/<c>WipeSaveState</c>
///     delete it for the active slot.
///   * Expedition files (<c>exp*</c>) and anything else written outside PlayerProgression.
/// </summary>
public class SaveRedirector
{
    private const string RL_ROOT_FOLDER = "RainWorldRL";
    private const string RL_SAVES_FOLDER = "saves";

    private readonly ManualLogSource log;
    private bool hooksInstalled = false;
    private bool swapping = false;

    /// <summary>Name of the RL save profile (sub-directory). Defaults to "default".</summary>
    public string SaveName { get; set; } = "default";

    /// <summary>While true, newly constructed PlayerProgression instances use the RL save directory.</summary>
    public bool Active { get; private set; } = false;

    /// <summary>True from a swap request until the new progression has finished loading.</summary>
    public bool IsSwapping => swapping;

    /// <summary>If set, overrides <see cref="SaveDirectory"/>. The Linux launcher points it into each
    /// instance's Wine prefix so parallel instances never share a save.</summary>
    public const string SAVE_DIR_ENV_VAR = "RAINWORLD_RL_SAVE_DIR";

    /// <summary>Absolute directory the RL save file lives in.</summary>
    public string SaveDirectory
    {
        get
        {
            string overrideDir = Environment.GetEnvironmentVariable(SAVE_DIR_ENV_VAR);
            if (!string.IsNullOrEmpty(overrideDir))
                return overrideDir;
            string name = string.IsNullOrEmpty(SaveName) ? "default" : SaveName;
            foreach (char c in Path.GetInvalidFileNameChars())
                name = name.Replace(c, '_');
            return Path.Combine(Path.Combine(Path.Combine(Paths.PluginPath, RL_ROOT_FOLDER), RL_SAVES_FOLDER), name);
        }
    }

    public SaveRedirector(ManualLogSource log)
    {
        this.log = log;
    }

    public void Install()
    {
        if (hooksInstalled)
            return;

        On.PlayerProgression.ctor_RainWorld_bool_bool_string += PlayerProgression_ctor;
        hooksInstalled = true;
    }

    public void Uninstall()
    {
        if (!hooksInstalled)
            return;

        On.PlayerProgression.ctor_RainWorld_bool_bool_string -= PlayerProgression_ctor;
        hooksInstalled = false;
    }

    private void PlayerProgression_ctor(
        On.PlayerProgression.orig_ctor_RainWorld_bool_bool_string orig,
        PlayerProgression self, RainWorld rainWorld, bool tryLoad, bool saveAfterLoad, string overrideBaseDir)
    {
        if (Active)
        {
            string dir = SaveDirectory;
            try
            {
                Directory.CreateDirectory(dir);
            }
            catch (Exception ex)
            {
                log?.LogError($"[SaveRedirector] Could not create RL save directory '{dir}': {ex.Message}");
            }
            overrideBaseDir = dir;
            log?.LogInfo($"[SaveRedirector] Progression redirected to '{dir}'");
        }

        orig(self, rainWorld, tryLoad, saveAfterLoad, overrideBaseDir);
    }

    /// <summary>
    /// Destroys the current (normal) progression and constructs a new one backed by the RL save
    /// directory. Poll <see cref="Update"/> / <see cref="IsSwapping"/> until the load completes.
    /// Must not be called while a RainWorldGame is running.
    /// </summary>
    public void SwapToRLSave(RainWorld rw)
    {
        if (Active && !swapping)
            return;

        Active = true;
        Swap(rw, "RL");
    }

    /// <summary>Destroys the RL progression and reloads the player's normal progression.</summary>
    public void SwapToNormalSave(RainWorld rw)
    {
        if (!Active && !swapping)
            return;

        Active = false;
        Swap(rw, "normal");
    }

    private void Swap(RainWorld rw, string label)
    {
        if (rw == null)
            throw new InvalidOperationException("RainWorld instance not available");

        int slot = rw.options != null ? rw.options.saveSlot : 0;
        if (rw.progression != null)
            rw.progression.Destroy(slot);

        // The ctor hook applies the RL directory when Active; otherwise the game's default path is used.
        rw.progression = new PlayerProgression(rw, true, false);
        swapping = true;
        log?.LogInfo($"[SaveRedirector] Swapping to {label} save (slot {slot})");
    }

    /// <summary>Advances the swap state; call every Update.</summary>
    public void Update(RainWorld rw)
    {
        if (!swapping || rw == null || rw.progression == null)
            return;

        if (rw.progression.progressionLoaded && !rw.progression.SaveDataBusy)
        {
            swapping = false;
            log?.LogInfo($"[SaveRedirector] Progression loaded ({(Active ? "RL" : "normal")} save)");
        }
    }

    /// <summary>
    /// Wipes the RL save (everything except misc progression) via <c>PlayerProgression.WipeAll()</c>.
    /// Only valid while the RL progression is active and loaded. WipeAll writes the wiped file and
    /// then re-reads it asynchronously; callers should wait for <c>!SaveDataBusy</c> afterwards.
    /// </summary>
    public void WipeRLSave(RainWorld rw)
    {
        if (!Active)
            throw new InvalidOperationException("RL save is not active; refusing to wipe the normal save");
        if (swapping || rw?.progression == null || !rw.progression.progressionLoaded)
            throw new InvalidOperationException("RL progression is not loaded");

        log?.LogInfo("[SaveRedirector] Wiping RL save");
        rw.progression.WipeAll();
    }
}
