using System.Threading;

/// <summary>
/// Keeps one core busy while the agent is in control.
///
/// Each step hands work between several threads (the game's main thread, Unity's render worker, Wine's D3D
/// thread, the GPU driver) within a couple of milliseconds. With every core idle between hand-offs the CPU
/// drops to low clocks / sleep states and each wake-up costs time. RoomPreparer used to spin a core by
/// accident (see <see cref="WorkerThreadFix"/>), and on a cluster node with an L40S GPU that was worth
/// ~1.75x steps/s (VirtualGL 260 -> 465; CPU route 259 -> 287). This spins on purpose instead, only while
/// the agent plays, so nothing spins in menus, under human override, or after Python disconnects.
///
/// Off unless <see cref="ENV_VAR"/> is "1". The Linux launcher sets it except on WSL2, where the Windows
/// host manages clocks and the spinning core just costs one of the game's cores (~10% slower).
/// </summary>
public class CoreWarmer
{
    public const string ENV_VAR = "RAINWORLD_RL_CORE_WARMER";

    public readonly bool Enabled = System.Environment.GetEnvironmentVariable(ENV_VAR) == "1";

    private volatile bool active;
    private volatile int generation;

    public void SetActive(bool value)
    {
        value &= Enabled;
        if (value == active)
            return;
        active = value;
        generation++;  // only the main thread writes; a spinner exits once its generation is stale
        if (value)
        {
            int gen = generation;
            new Thread(() => Spin(gen)) { IsBackground = true, Name = "RainWorldRL core warmer" }.Start();
        }
    }

    private void Spin(int gen)
    {
        while (active && generation == gen)
            Thread.SpinWait(1000);
    }
}
