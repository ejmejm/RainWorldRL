# Rain World RL

This is a project to turn Rain World into a reinforcement learning environment.

A BepInEx mod drives the game step-by-step over shared memory; a Gymnasium env in `rainworld_rl/`
talks to it. See `docs/PYTHON_API.md` for usage, `docs/PROTOCOL.md` for the wire format,
and `tests/README.md` for the end-to-end test harness.
Baseline agents (PPO, random) live in `baselines/`; see `baselines/README.md`.

## Quick start

```
cp rainworld_rl.example.toml rainworld_rl.toml   # set game_dir if not on Z:
python -m pytest tests --e2e -q                   # build, launch the game, run everything
python -m rainworld_rl.test_env --launch   # random agent demo (run from the repo's parent dir)
```

## TODO
- [x] Immediately load into a preconfigured save state when the python env is connected/resets
- [x] Make keyboard action space where multiple keys can be pressed at once (`MultiBinary(9)`, see docs/PROTOCOL.md)
- [x] Remake the discrete action space, but as a wrapper over the keyboard action space (`rainworld_rl.wrappers.DiscreteActions`, optional)
- [x] Stop exit buttons from working, or immediately load save when main menu is loaded
- [x] Add a configurable reward function for things like dying, eating, sleeping, finding new areas, etc. (`rainworld_rl.rewards`, see docs/REWARDS.md)
- [x] Automatically launch the game from the python script
- [x] Skip the fade-in after reset so the first observations are not black frames
- [x] Scripted death (`env.debug_kill()`) so the death-edge e2e test can run
- [ ] Live test for the cycle-survived reward edge (needs a shortened rain cycle or a scripted sleep)
- [ ] Harden connect() against the game's synchronous initial load (occasional heartbeat false-negative right after launch)
- [ ] Look into headless mode
- [ ] Look into running multiple environments in parallel
