# Rain World RL

This is a project to turn Rain World into a reinforcement learning environment.

A BepInEx mod drives the game step-by-step over shared memory; a Gymnasium env in `rainworld_rl/`
talks to it. See `docs/PYTHON_API.md` for usage, `docs/PROTOCOL.md` for the wire format,
and `tests/README.md` for the end-to-end test harness.
Baseline agents (PPO, random) live in `baselines/`; see `baselines/README.md`.

## Quick start

You need your own copy of Rain World (v1.11.8). The Steam client does not need to be running.

**Windows**

```
pip install -e .
rainworld-rl setup --game-dir "Z:/SteamLibrary/steamapps/common/Rain World"
rainworld-rl doctor                        # launch the game, run random steps, print steps/s
python -m pytest tests --e2e -q            # build the mod, launch the game, run everything
```

**Linux, WSL2 and clusters**: the game runs under Wine inside an Apptainer image (`container/`).
You need Apptainer (`module load apptainer` on Compute Canada).

```
pip install -e .
rainworld-rl download --user <steam_user>  # SteamCMD into ~/.local/share/rainworld_rl/game (or copy the game there)
rainworld-rl setup                         # pull the image, install the mod, write ~/.config/rainworld_rl/rainworld_rl.toml
rainworld-rl doctor
```

If the image isn't published yet, build it yourself:
`docker build -t rainworld-rl container && apptainer build ~/.local/share/rainworld_rl/rainworld-rl.sif docker-daemon://rainworld-rl:latest`.

On Linux the `renderer` config key picks how frames are drawn. The default `auto` uses `virtualgl`
(the GPU, through VirtualGL) when an NVIDIA GPU is visible, `wsl` (WSL2's GPU) on WSL2, and `cpu`
(Mesa on the CPU) otherwise. A pip install ships a prebuilt mod DLL, so the .NET SDK is only needed
to change the mod.

## TODO
- [x] Immediately load into a preconfigured save state when the python env is connected/resets
- [x] Make keyboard action space where multiple keys can be pressed at once (`MultiBinary(9)`, see docs/PROTOCOL.md)
- [x] Remake the discrete action space, but as a wrapper over the keyboard action space (`rainworld_rl.wrappers.DiscreteActions`, optional)
- [x] Stop exit buttons from working, or immediately load save when main menu is loaded
- [x] Add a configurable reward function for things like dying, eating, sleeping, finding new areas, etc. (`rainworld_rl.rewards`, see docs/REWARDS.md)
- [x] Automatically launch the game from the python script
- [x] Skip the fade-in after reset so the first observations are not black frames
- [x] Scripted death (`env.debug_kill()`) so the death-edge e2e test can run
- [x] Live test for the cycle-survived reward edge (real fed and starving hibernations via `env.debug_enter_shelter()`, `tests/e2e/test_sleep.py`)
- [x] Harden connect() against the game's synchronous initial load (occasional heartbeat false-negative right after launch)
- [x] Headless mode (Linux: Wine + Xvfb, see Quick start)
- [x] Running multiple environments in parallel (Linux: `instance`)
