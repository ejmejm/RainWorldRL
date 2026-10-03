# Rain World RL test harness

```
tests/
  conftest.py        sys.path wiring, --e2e/--no-launch/--no-build, game/env/fresh_env fixtures
  harness.py         game-free helpers: step_n(), status bits, read_state(), info contract
  unit/              run anywhere, no game needed (harness smoke tests, FakeEnv)
  e2e/               drive the real game; auto-marked `e2e`, skipped unless --e2e
```

Requirements: Python 3.13, `pip install pytest numpy gymnasium` (see `python/requirements.txt`).
Run from the repo root (`E:\projects\rainworld_rl`); `pytest.ini` there sets the test paths.

## Running

| Command | What happens |
|---|---|
| `python -m pytest tests -q` | Unit tests only; every e2e test is reported as skipped (`-ra` prints the reason). |
| `python -m pytest tests --e2e -q` | Builds + deploys the mod, launches Rain World, waits for READY, runs everything, kills the game at the end. |
| `python -m pytest tests --e2e --no-build -q` | Same, but skips the C# build/deploy. |
| `python -m pytest tests --e2e --no-launch -q` | Game already running with the mod: just connect, run, and leave the game running. |
| `python -m pytest tests/e2e/test_stepping.py --e2e --no-launch -q -k counter` | One file / one test. |
| `python -m pytest tests --e2e --collect-only -q` | Check that everything collects (no game needed). |

The conftest imports `rainworld_rl.*` lazily inside fixtures, so collection never
fails while the client is being rewritten; a missing API module aborts the session with a
clear message only when an e2e test actually runs.

If launching fails, the whole session aborts with the `LaunchError` message (it includes the
tail of the BepInEx log). With `--no-launch` and no running game, the session aborts with the
`GameNotRunningError` message.

## Fixtures

| Fixture | Scope | Cost | Use when |
|---|---|---|---|
| `game` | session | launch once (~30-90 s with build) | you need the raw shared `RainWorldEnv` |
| `env` | function | free | state carry-over between tests is fine (most tests) |
| `fresh_env` | function | `reset()` = wipe save + reload, ~10-30 s | the test needs a known starting state |
| `api` | session | free | you need `RainWorldEnv`/`launch`/`kill_game`/... classes |
| `reattach_shared_env` | function | ~1-5 s teardown | the test creates its *own* second env against the running game |

Helpers in `tests/harness.py`:

- `step_n(env, n, action_fn=None) -> [(obs, info), ...]` (no-op when `action_fn` is omitted; `action_fn(i)` otherwise)
- `infos(results, "step_counter")` to pull one info key out of a `step_n` result
- `assert_info_contract(info)` checks the info keys/types of the API contract
- `read_state(env)`, `state_field(state, "heartbeat")`, `status_bit(state, "ready")` for raw header access
- `wait_for(predicate, timeout)`

## Expected timings (ticks_per_step=1, 160x90)

- Launch with build: 30-90 s. `--no-build`: 15-45 s. `--no-launch` connect: < 5 s.
- One step: a few ms up to ~50 ms; 200 no-op steps < 10 s typically (hard limit 120 s in the test).
- `reset()`: 10-30 s. The suite currently performs 4 resets (test_reset x3, test_death x1).
- Whole `--e2e` run after launch: roughly 1.5-3 minutes.

## Adding a test

1. Game-free? Put it in `tests/unit/`. It must not import `rainworld_rl` at module level
   unless it uses `pytest.importorskip`, so the suite keeps collecting while the client is unfinished.
2. Needs the game? Put it in `tests/e2e/` - it is marked `e2e` and skipped without `--e2e` automatically.
   - Prefer `env` (shared, no reset). Use `fresh_env` only when you really need a wiped save.
   - Use `step_n` + `infos` instead of hand-written loops; use `np.array(obs, copy=True)` if you keep a frame
     across steps (the env may reuse its buffer).
   - Anything that can legitimately fail because of game geometry / randomness goes under
     `@pytest.mark.xfail(strict=False, reason=...)`.
   - If you create a second `RainWorldEnv`, request `reattach_shared_env` so the session env is restored.
   - Protocol constants (status bits, action ids, fresh-save cycle) live in `tests/harness.py`; keep them in
     sync with `docs/PROTOCOL.md`.
3. Run `python -m pytest tests -q` (unit + skip check) and, if you can, `python -m pytest tests --e2e --no-launch -q`.
