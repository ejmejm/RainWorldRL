# Rain World RL

This is a project to turn Rain World into a reinforcement learning environment.

A BepInEx mod (C# source in `mod/`) drives the game step-by-step over shared memory; a Gymnasium
env in `rainworld_rl/` talks to it. See `docs/PYTHON_API.md` for usage, `docs/REWARDS.md` for the
default reward, `docs/PROTOCOL.md` for the wire format, and `tests/README.md` for the tests.
Baseline agents (PPO, PPO+RND, DCEO) live in
[RainWorldRL-baselines](https://github.com/ejmejm/RainWorldRL-baselines).

## Quick start

You need your own copy of Rain World (v1.11.8). The Steam client does not need to be running.

**Windows**

```
pip install git+https://github.com/ejmejm/RainWorldRL
rainworld-rl setup --game-dir "C:/path/to/Rain World"   # omit --game-dir for Steam's default location
rainworld-rl doctor                                     # launch the game, run random steps, print steps/s
```

**Linux, WSL2 and clusters**: the game runs under Wine inside an Apptainer image (`container/`).
You need Apptainer (on HPC clusters, e.g. `module load apptainer`).

```
pip install git+https://github.com/ejmejm/RainWorldRL
rainworld-rl download --user <steam_user>  # SteamCMD into ~/.local/share/rainworld_rl/game (or copy the game there)
rainworld-rl setup                         # pull the image, install the mod, write ~/.config/rainworld_rl/rainworld_rl.toml
rainworld-rl doctor
```

If the image isn't published yet, build it yourself:
`docker build -t rainworld-rl container && apptainer build ~/.local/share/rainworld_rl/rainworld-rl.sif docker-daemon://rainworld-rl:latest`.

On Linux the `renderer` config key picks how frames are drawn. The default `auto` uses `virtualgl`
(the GPU, through VirtualGL) when the job can use an NVIDIA GPU, `wsl` (WSL2's GPU) on WSL2, and `cpu`
(Mesa on the CPU) otherwise. A pip install ships a prebuilt mod DLL, so the .NET SDK is only needed
to change the mod.

## Development

```
git clone https://github.com/ejmejm/RainWorldRL && cd RainWorldRL
pip install -e ".[test]"
python -m rainworld_rl.launcher --build-only   # build mod/ (needs the .NET SDK), deploy it, refresh rainworld_rl/mod/
python -m pytest tests --e2e -q                # build the mod, launch the game, run everything
```

## License

MIT, see [LICENSE](LICENSE).
