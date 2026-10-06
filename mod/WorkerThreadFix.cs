using System;
using System.Reflection;
using System.Threading;
using BepInEx.Logging;
using Mono.Cecil.Cil;
using MonoMod.Cil;

/// <summary>
/// Stops RoomPreparer's worker thread from spinning a core while it waits on the main thread.
///
/// RoomPreparer.UpdateThread (RoomPreparer.cs) loops over <c>switch (status)</c> without ever sleeping. In two
/// states it only waits for RoomPreparer.Update on the main thread to consume a request and advance status:
/// status 0 with requestShortcutsReady set, and status 4 (requestReadyForAI). While the game is frozen between
/// RL steps nothing consumes them, so the thread burns a core; and when a game is torn down mid-preparation
/// (e.g. leaving RL mode) nothing ever will, so the thread spins forever.
///
/// An IL hook calls <see cref="WaitIfIdle"/> just before that switch, which sleeps 1 ms in exactly those two
/// waiting states and passes the status through. Working states are untouched, so preparation is not slowed.
/// If the IL pattern is not found the hook logs a warning and leaves the method alone.
/// </summary>
public class WorkerThreadFix
{
    private static readonly FieldInfo requestShortcutsReady =
        typeof(RoomPreparer).GetField("requestShortcutsReady", BindingFlags.Instance | BindingFlags.NonPublic);

    private readonly ManualLogSource log;

    public WorkerThreadFix(ManualLogSource log)
    {
        this.log = log;
    }

    public void Install() => IL.RoomPreparer.UpdateThread += RoomPreparer_UpdateThread;

    public void Uninstall() => IL.RoomPreparer.UpdateThread -= RoomPreparer_UpdateThread;

    private void RoomPreparer_UpdateThread(ILContext il)
    {
        var c = new ILCursor(il);
        if (requestShortcutsReady == null || !c.TryGotoNext(MoveType.Before, i => i.OpCode == OpCodes.Switch))
        {
            log.LogWarning("[WorkerThreadFix] RoomPreparer.UpdateThread not as expected; its thread will busy-wait");
            return;
        }
        // The status is on the stack for the switch: (status, this) -> WaitIfIdle -> status.
        c.Emit(OpCodes.Ldarg_0);
        c.EmitDelegate<Func<int, RoomPreparer, int>>(WaitIfIdle);
        log.LogInfo("[WorkerThreadFix] RoomPreparer.UpdateThread patched");
    }

    private static int WaitIfIdle(int status, RoomPreparer self)
    {
        if (status == 4 || (status == 0 && (bool)requestShortcutsReady.GetValue(self)))
            Thread.Sleep(1);
        return status;
    }
}
