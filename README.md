# Rain World RL

This is a project to turn Rain World into a reinforcement learning environment.

A BepInEx mod drives the game step-by-step over shared memory; a Gymnasium env in `python/`
talks to it. See `docs/PYTHON_API.md` for usage, `docs/PROTOCOL.md` for the wire format,
and `tests/README.md` for the end-to-end test harness.

## Quick start

```
cp rainworld_rl.example.toml rainworld_rl.toml   # set game_dir if not on Z:
python -m pytest tests --e2e -q                   # build, launch the game, run everything
python -m rainworld_rl.python.test_env --launch   # random agent demo (run from the repo's parent dir)
```

## TODO
- [x] Immediately load into a preconfigured save state when the python env is connected/resets
- [ ] Make keyboard action space where multiple keys can be pressed at once
- [ ] Remake the discrete action space, but as a wrapper over the keyboard action space
- [x] Stop exit buttons from working, or immediately load save when main menu is loaded
- [ ] Add a configurable reward function for things like dying, eating, sleeping, finding new areas, etc.
- [x] Automatically launch the game from the python script
- [ ] Skip the fade-in after reset so the first observations are not black frames
- [ ] Scripted death (kill command or hazard room) so the death-edge e2e test can run
- [ ] Look into headless mode
- [ ] Look into running multiple environments in parallel
