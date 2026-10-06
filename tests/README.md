# Rain World RL tests

```
tests/
  conftest.py   --e2e/--no-launch/--no-build; game/env/fresh_env/reattach_shared_env fixtures
  unit/         game-free tests of the package (fake_mapping.py plays the mod's side of the shared memory)
  e2e/          drive the real game; auto-marked `e2e`, skipped unless --e2e (helpers in e2e/helpers.py)
```

Install with `pip install -e ".[test]"` and run from the repo root (the pytest settings are in
`pyproject.toml`). On Linux the e2e tests use the config written by `rainworld-rl setup`
(`game_dir`, `container`, `renderer`) and need `apptainer` on PATH.

| Command | What happens |
|---|---|
| `python -m pytest -q` | Unit tests; the e2e tests are reported as skipped. |
| `python -m pytest tests --e2e -q` | Builds + deploys the mod, launches Rain World, runs everything, kills the game at the end. |
| `python -m pytest tests --e2e --no-build -q` | Same without the C# build/deploy. |
| `python -m pytest tests --e2e --no-launch -q` | Connects to a running game with the mod and leaves it running. |
| `python -m pytest tests/e2e/test_stepping.py --e2e --no-launch -q -k map` | One file / one test. |

A failed launch aborts the session with the `LaunchError` message (it includes the tail of the
BepInEx log); `--no-launch` without a running game aborts with the `GameNotRunningError` message.

## Fixtures

| Fixture | Scope | Cost | Use when |
|---|---|---|---|
| `game` | session | launch once (~30-90 s with build) | you need the shared `RainWorldEnv` itself |
| `env` | function | free | state carry-over between tests is fine (most tests) |
| `fresh_env` | function | `reset()` = wipe save + reload, ~10-30 s | the test needs a known starting state |
| `reattach_shared_env` | function | ~1-5 s teardown | the test creates its own second `RainWorldEnv` against the running game |

## Adding a test

- Game-free: `tests/unit/`, with `FakeMapping` standing in for the mod.
- Needs the game: `tests/e2e/`. Prefer `env`; reset only when the test needs a wiped save. Copy frames
  you keep across steps (`np.array(obs, copy=True)`; the env may reuse its buffer).
