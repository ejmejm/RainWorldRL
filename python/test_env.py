"""
Test script for the Rain World RL environment.

This script connects to a running Rain World game with the RainWorldRL mod
and runs a simple random agent to verify the environment is working.

Usage:
    1. Start Rain World with the RainWorldRL mod installed
    2. Load into a game (start a new game or continue)
    3. Run this script: python -m rainworld_rl.python.test_env
"""

import argparse
import time

import numpy as np

from .rainworld_env import RainWorldEnv


def main():
    parser = argparse.ArgumentParser(description = "Test Rain World RL environment")
    parser.add_argument(
        "--width", type = int, default = 320, help = "Frame width (default: 320)"
    )
    parser.add_argument(
        "--height", type = int, default = 240, help = "Frame height (default: 240)"
    )
    parser.add_argument(
        "--ticks", type = int, default = 4, help = "Ticks per step (default: 4)"
    )
    parser.add_argument(
        "--steps", type = int, default = 1000, help = "Number of steps to run (default: 1000)"
    )
    parser.add_argument(
        "--timeout", type = float, default = 30.0, help = "Connection timeout (default: 30)"
    )
    parser.add_argument(
        "--render", action = "store_true", help = "Save frames as images"
    )
    parser.add_argument(
        "--debug", action = "store_true", help = "Print detailed timing debug info"
    )
    args = parser.parse_args()

    print(f"Creating environment ({args.width}x{args.height}, {args.ticks} ticks/step)")

    env = RainWorldEnv(
        frame_width = args.width,
        frame_height = args.height,
        ticks_per_step = args.ticks,
        connection_timeout = args.timeout,
        render_mode = "rgb_array" if args.render else None,
        debug_timing = args.debug,
    )

    try:
        print(f"Connecting to Rain World (timeout: {args.timeout}s)...")
        obs, info = env.reset()
        print(f"Connected! Initial observation shape: {obs.shape}")

        if args.render:
            save_frame(obs, 0)

        # Statistics
        step_times = []
        action_counts = np.zeros(18, dtype = np.int32)

        print(f"\nRunning {args.steps} steps with random actions...")
        print("-" * 50)

        for step in range(args.steps):
            # Random action
            action = env.action_space.sample()
            action_counts[action] += 1

            # Step
            start_time = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.perf_counter() - start_time
            step_times.append(step_time)

            # Progress report every 100 steps
            if (step + 1) % 100 == 0:
                avg_time = np.mean(step_times[-100:])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"Step {step + 1}/{args.steps} | Avg step time: {avg_time*1000:.2f}ms | FPS: {fps:.1f}")

            if args.render and (step + 1) % 10 == 0:
                save_frame(obs, step + 1)

            # Check if player died
            if info.get("player_dead", False):
                print(f"  Player died at step {step + 1}")

        print("-" * 50)
        print("\nTest complete!")
        print(f"Total steps: {args.steps}")
        print(f"Average step time: {np.mean(step_times)*1000:.2f}ms")
        print(f"Effective FPS: {1.0/np.mean(step_times):.1f}")
        print(f"Min step time: {np.min(step_times)*1000:.2f}ms")
        print(f"Max step time: {np.max(step_times)*1000:.2f}ms")

        # Action distribution
        print("\nAction distribution:")
        action_names = [
            "No-op", "Left", "Right", "Up", "Down",
            "Jump", "Grab", "Throw",
            "Left+Jump", "Right+Jump", "Up+Jump", "Down+Jump",
            "Left+Grab", "Right+Grab", "Up+Grab", "Down+Grab",
            "Crawl Left", "Crawl Right",
        ]
        for i, count in enumerate(action_counts):
            if count > 0:
                print(f"  {action_names[i]}: {count} ({100*count/args.steps:.1f}%)")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        env.close()
        print("\nEnvironment closed")


def save_frame(frame: np.ndarray, step: int):
    """Save a frame as a PNG image."""
    try:
        from PIL import Image
        img = Image.fromarray(frame)
        img.save(f"frame_{step:06d}.png")
    except ImportError:
        print("Warning: PIL not installed, cannot save frames")


if __name__ == "__main__":
    main()

