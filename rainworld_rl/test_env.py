"""
Smoke test for the Rain World RL environment with a random agent.

By default this attaches to a game that is already running with the
RainWorldRL mod (``env.connect()``). Pass ``--launch`` to build the mod,
(re)start the game and connect (``env.launch()``; Steam must be running).

Actions are sampled from the native ``MultiBinary`` key space (any key
combination). Pass ``--discrete`` to wrap the env in ``DiscreteActions`` and
sample from the classic 18-action set instead.

Usage:
    python -m rainworld_rl.test_env [--launch] [--no-wipe] [--discrete] [--steps N] ...
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from .launcher import kill_game
from .rainworld_env import RainWorldEnv
from .shared_memory import KEY_NAMES, NUM_KEYS, GameNotRunningError, action_to_bits, pressed_key_names
from .wrappers import ACTION_NAMES, DiscreteActions


def main() -> int:
    parser = argparse.ArgumentParser(description = "Test Rain World RL environment")
    parser.add_argument("--width", type = int, default = 320, help = "Frame width (default: 320)")
    parser.add_argument("--height", type = int, default = 240, help = "Frame height (default: 240)")
    parser.add_argument("--ticks", type = int, default = 4, help = "Ticks per step (default: 4)")
    parser.add_argument("--steps", type = int, default = 1000, help = "Number of steps to run (default: 1000)")
    parser.add_argument("--ready-timeout", type = float, default = 60.0, help = "Seconds to wait for READY (default: 60)")
    parser.add_argument("--launch", action = "store_true", help = "Build the mod and (re)start the game before connecting")
    parser.add_argument("--no-build", action = "store_true", help = "With --launch: skip dotnet build")
    parser.add_argument("--no-wipe", action = "store_true", help = "Attach to the game in progress instead of sending RESET")
    parser.add_argument("--discrete", action = "store_true", help = "Use the optional Discrete(18) DiscreteActions wrapper instead of raw keys")
    parser.add_argument("--info-every", type = int, default = 100, help = "Print info fields every N steps (default: 100)")
    parser.add_argument("--render", action = "store_true", help = "Save frames as PNG images")
    parser.add_argument("--debug", action = "store_true", help = "Print detailed timing debug info")
    parser.add_argument("--kill", action = "store_true", help = "Close the game when the script finishes")
    args = parser.parse_args()

    logging.basicConfig(level = logging.INFO, format = "%(levelname)s %(name)s: %(message)s")

    print(f"Creating environment ({args.width}x{args.height}, {args.ticks} ticks/step)")
    base_env = RainWorldEnv(
        frame_width = args.width,
        frame_height = args.height,
        ticks_per_step = args.ticks,
        ready_timeout = args.ready_timeout,
        render_mode = "rgb_array" if args.render else None,
        debug_timing = args.debug,
    )
    env = DiscreteActions(base_env) if args.discrete else base_env
    print(f"Action space: {env.action_space}" + ("" if args.discrete else f"  keys={KEY_NAMES}"))

    step_times = []
    discrete_counts = np.zeros(len(ACTION_NAMES), dtype = np.int64)
    key_counts = np.zeros(NUM_KEYS, dtype = np.int64)
    combo_counts: dict = {}
    deaths = 0
    steps_done = 0

    try:
        if args.launch:
            print("Launching Rain World (build + restart)...")
            base_env.launch(build = not args.no_build)
        else:
            print("Connecting to a running Rain World...")
            try:
                base_env.connect(wait_ready = True)
            except GameNotRunningError as e:
                print(f"\n{e}\n\nTip: pass --launch to start it from here.")
                return 1
        print("Connected.")

        wipe = not args.no_wipe
        print("Resetting (fresh game)..." if wipe else "Attaching to game in progress (no wipe)...")
        obs, info = env.reset(options = {"wipe": wipe})
        print(f"Initial observation shape: {obs.shape}")
        print_info(0, info)
        if args.render:
            save_frame(obs, 0)

        print(f"\nRunning {args.steps} steps with random actions...")
        print("-" * 60)

        for step in range(args.steps):
            action = env.action_space.sample()
            if args.discrete:
                discrete_counts[int(action)] += 1
                bits = env.action(action)  # the key vector the base env will receive
            else:
                bits = action
            bits = action_to_bits(bits)
            for i in range(NUM_KEYS):
                if bits & (1 << i):
                    key_counts[i] += 1
            combo_counts[bits] = combo_counts.get(bits, 0) + 1

            start_time = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_times.append(time.perf_counter() - start_time)
            steps_done = step + 1

            if info["player_dead"]:
                deaths += 1
                print(
                    f"  ** player_dead at step {steps_done} "
                    f"(cycle {info['cycle_number']}, room {info['room_index']}, karma {info['karma']})"
                )

            if steps_done % args.info_every == 0:
                avg_time = float(np.mean(step_times[-args.info_every:]))
                fps = 1.0 / avg_time if avg_time > 0 else 0.0
                print(f"Step {steps_done}/{args.steps} | avg step {avg_time * 1000:.2f}ms | {fps:.1f} steps/s")
                print_info(steps_done, info)

            if args.render and steps_done % 10 == 0:
                save_frame(obs, steps_done)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        if args.kill:
            kill_game()
            print("\nEnvironment closed, game killed")
        else:
            print("\nEnvironment closed (game left running)")

    if step_times:
        print("-" * 60)
        print(f"Steps: {steps_done}  Deaths: {deaths}")
        print(f"Average step time: {np.mean(step_times) * 1000:.2f}ms  ({1.0 / np.mean(step_times):.1f} steps/s)")
        print(f"Min/Max step time: {np.min(step_times) * 1000:.2f}ms / {np.max(step_times) * 1000:.2f}ms")
        if args.discrete:
            print("\nDiscrete action distribution:")
            for i, count in enumerate(discrete_counts):
                if count > 0:
                    print(f"  {ACTION_NAMES[i]:12s} {count:5d} ({100 * count / steps_done:.1f}%)")
        print("\nKey press frequency (fraction of steps each key was held):")
        for i, count in enumerate(key_counts):
            print(f"  {KEY_NAMES[i]:8s} {count:5d} ({100 * count / steps_done:.1f}%)")
        print("\nMost common key combinations:")
        for bits, count in sorted(combo_counts.items(), key = lambda kv: -kv[1])[:10]:
            names = "+".join(pressed_key_names(bits)) or "(no keys)"
            print(f"  {names:32s} {count:5d} ({100 * count / steps_done:.1f}%)")
    return 0


def print_info(step: int, info: dict) -> None:
    x, y = info["player_pos"]
    print(
        f"  [info @ {step}] in_game={info['in_game']} ready={info['ready']} override={info['human_override']} "
        f"cycle={info['cycle_number']} room={info['room_index']} pos=({x:.0f}, {y:.0f}) "
        f"karma={info['karma']}/{info['karma_cap']} food={info['food']} dialog={info['dialog_open']} "
        f"step_counter={info['step_counter']}"
    )


def save_frame(frame: np.ndarray, step: int) -> None:
    """Save a frame as a PNG image."""
    try:
        from PIL import Image
    except ImportError:
        print("Warning: PIL not installed, cannot save frames")
        return
    Image.fromarray(frame).save(f"frame_{step:06d}.png")


if __name__ == "__main__":
    raise SystemExit(main())
