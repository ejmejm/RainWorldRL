"""
Record a wall-clock-bounded PPO training run on the real game as a video plus a
reward-over-time plot.

The agent trains exactly as in ``train_ppo`` (same model, rollout and update
code), but the loop stops after ``--seconds`` of training time (the game launch
and JIT warm-up happen before the clock starts). While it runs, a tap just
above the reward wrapper records what the agent sees at ``--video_width x
--video_height`` RGB (the policy itself still gets the usual 96x54 grayscale
stack, produced by resizing those frames), samples that stream at ``--fps`` in
wall-clock time, overlays live stats, and streams it to an mp4. Policy updates
freeze the picture for their real duration so the video stays real-time.

Outputs in ``--out`` (default ``baselines/runs/ppo_<timestamp>/``):

* ``video.mp4``          real-time recording with overlay
* ``reward_plot.png``    per-step reward (rolling mean), per-rollout mean,
                         cumulative reward-term totals, policy entropy
* ``steps.csv``          one row per env step (t, step, reward, each reward term)
* ``updates.csv``        one row per PPO update (losses, entropy, timing)
* ``summary.json``       totals and settings
* ``latest.eqx``         final checkpoint (+ ``model_config.json``)

Usage::

    python -m baselines.ppo.record_run --seconds 300 --launch --kill_game
    python -m baselines.ppo.record_run --seconds 20 --fake        # pipeline check
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont

from baselines.common.env import FakeRainWorldEnv, FrameStack, GrayscaleObs, attach
from baselines.common.metrics import RolloutMetrics
from baselines.ppo.train_ppo import (
    collect_rollout,
    init_experiment,
    model_config_from_args,
    parse_args as ppo_parse_args,
    sample_action,
    save_checkpoint,
    train_step,
    value_of,
)

logger = logging.getLogger("record_run")

DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
ROLLING_WINDOW = 500   # steps, for the rolling mean reward in the overlay and plot


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class ResizeObs(gym.ObservationWrapper):
    """RGB uint8 ``(H, W, 3)`` -> RGB uint8 ``(h, w, 3)`` via bilinear resize."""

    def __init__(self, env: gym.Env, width: int, height: int):
        super().__init__(env)
        self.size = (width, height)
        self.observation_space = spaces.Box(0, 255, (height, width, 3), dtype = np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(obs).resize(self.size, Image.BILINEAR), dtype = np.uint8)


class Tap(gym.Wrapper):
    """Passes everything through; hands (frame, reward, info) to the recorder."""

    def __init__(self, env: gym.Env, recorder: "Recorder"):
        super().__init__(env)
        self.recorder = recorder

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.recorder.on_reset(obs, info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.recorder.on_step(obs, float(reward), info)
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

def _font(size: int):
    try:
        return ImageFont.load_default(size = size)
    except TypeError:            # very old Pillow
        return ImageFont.load_default()


class Recorder:
    """Real-time video sampler + per-step reward log."""

    TEXT_BAR = 84

    def __init__(self, out_dir: Path, fps: int, scale: int, frame_shape: Tuple[int, int]):
        import imageio.v2 as imageio

        self.out_dir = out_dir
        self.fps = fps
        self.period = 1.0 / fps
        self.scale = scale
        h, w = frame_shape
        self.canvas_size = (w * scale, h * scale + self.TEXT_BAR)
        self.writer = imageio.get_writer(
            str(out_dir / "video.mp4"), fps = fps, codec = "libx264", quality = 7,
            pixelformat = "yuv420p", macro_block_size = 1,
        )
        self.font = _font(18)
        self.small = _font(15)

        self.t0: Optional[float] = None
        self.next_frame_time = 0.0
        self.frames_written = 0
        self.last_frame: Optional[np.ndarray] = None
        self.last_info: Dict[str, Any] = {}
        self.label = ""

        self.env_steps = 0
        self.updates = 0
        self.total_reward = 0.0
        self.recent: Deque[float] = collections.deque(maxlen = ROLLING_WINDOW)
        self.term_totals: Dict[str, float] = {}
        self.rooms: set = set()
        self.deaths = 0
        self.slept = 0
        self.last_event = ""
        self.last_event_time = -1e9

        self.steps_log: List[Dict[str, Any]] = []
        self.rollout_marks: List[Tuple[int, float]] = []   # (env_step, mean reward of rollout)

    # -- lifecycle -----------------------------------------------------------

    def start_clock(self) -> None:
        self.t0 = time.perf_counter()
        self.next_frame_time = self.t0

    def elapsed(self) -> float:
        return 0.0 if self.t0 is None else time.perf_counter() - self.t0

    def close(self) -> None:
        if self.t0 is not None and self.last_frame is not None:
            self._catch_up(time.perf_counter(), "finished")
        self.writer.close()

    # -- env callbacks ---------------------------------------------------------

    def on_reset(self, frame: np.ndarray, info: Dict[str, Any]) -> None:
        self.last_frame = frame
        self.last_info = info
        self._note_room(info)

    def on_step(self, frame: np.ndarray, reward: float, info: Dict[str, Any]) -> None:
        self.last_frame = frame
        self.last_info = info
        self.env_steps += 1
        self.total_reward += reward
        self.recent.append(reward)
        terms = info.get("reward_terms", {}) or {}
        for name, value in terms.items():
            self.term_totals[name] = self.term_totals.get(name, 0.0) + float(value)
            if abs(float(value)) > 1e-9:
                self.last_event = f"{name} {float(value):+.2f}"
                self.last_event_time = self.elapsed()
        self._note_room(info)
        if info.get("player_dead"):
            self.deaths += 1
        if info.get("cycle_survived"):
            self.slept += 1
        if self.t0 is not None:
            row = {"t": round(self.elapsed(), 3), "step": self.env_steps, "reward": reward}
            row.update({f"term/{k}": float(v) for k, v in terms.items()})
            self.steps_log.append(row)
            self._catch_up(time.perf_counter(), "")

    def on_update_done(self, seconds: float, entropy: float) -> None:
        self.updates += 1
        self._catch_up(time.perf_counter(), f"policy update #{self.updates}  ({seconds:.1f}s, entropy {entropy:.2f})")

    def mark_rollout(self, mean_reward: float) -> None:
        self.rollout_marks.append((self.env_steps, mean_reward))

    # -- internals -------------------------------------------------------------

    def _note_room(self, info: Dict[str, Any]) -> None:
        if info.get("ready") and int(info.get("room_index", -1)) >= 0:
            self.rooms.add((info.get("region", ""), int(info["room_index"])))

    def _catch_up(self, now: float, label: str) -> None:
        """Write frames so the video keeps pace with wall-clock time."""
        if self.last_frame is None or self.t0 is None:
            return
        if now - self.next_frame_time > 3.0 * self.period and label == "":
            # Normal stepping should never fall far behind; if it did (e.g. a
            # stall), keep real time by emitting repeated frames below anyway.
            pass
        wrote = 0
        while self.next_frame_time <= now and wrote < self.fps * 60:
            self.writer.append_data(self._compose(label, self.next_frame_time - self.t0))
            self.next_frame_time += self.period
            self.frames_written += 1
            wrote += 1

    def _compose(self, label: str, t: float) -> np.ndarray:
        frame = Image.fromarray(self.last_frame).resize(
            (self.last_frame.shape[1] * self.scale, self.last_frame.shape[0] * self.scale), Image.NEAREST
        )
        canvas = Image.new("RGB", self.canvas_size, (12, 12, 12))
        canvas.paste(frame, (0, 0))
        d = ImageDraw.Draw(canvas)
        y = frame.height + 6
        info = self.last_info
        mean_recent = float(np.mean(self.recent)) if self.recent else 0.0
        mm, ss = divmod(int(t), 60)
        line1 = (f"t {mm:02d}:{ss:02d}   steps {self.env_steps:>6d}   updates {self.updates:>3d}   "
                 f"reward/step (last {ROLLING_WINDOW}) {mean_recent:+.4f}   total {self.total_reward:+.2f}")
        line2 = (f"rooms {len(self.rooms):>2d}   food {int(info.get('food', 0))}/{int(info.get('food_max', 0))}   "
                 f"karma {int(info.get('karma', 0))}   cycle {int(info.get('cycle_number', -1))}   "
                 f"deaths {self.deaths}   slept {self.slept}   "
                 f"{'in shelter' if info.get('in_shelter') else ''}{' RAIN' if info.get('rain') else ''}")
        d.text((8, y), line1, fill = (235, 235, 235), font = self.font)
        d.text((8, y + 26), line2, fill = (200, 200, 200), font = self.font)
        status = label
        if not status and self.elapsed() - self.last_event_time < 2.0 and self.last_event:
            status = self.last_event
        if status:
            color = (255, 200, 80) if label else (120, 230, 120)
            d.text((8, y + 54), status, fill = color, font = self.small)
        return np.asarray(canvas, dtype = np.uint8)


# ---------------------------------------------------------------------------
# Plot / files
# ---------------------------------------------------------------------------

def write_outputs(rec: Recorder, updates: List[Dict[str, Any]], out_dir: Path, summary: Dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # steps.csv
    term_keys = sorted({k for row in rec.steps_log for k in row if k.startswith("term/")})
    with open(out_dir / "steps.csv", "w", newline = "") as f:
        w = csv.DictWriter(f, fieldnames = ["t", "step", "reward"] + term_keys)
        w.writeheader()
        for row in rec.steps_log:
            w.writerow({k: row.get(k, 0.0) for k in w.fieldnames})
    # updates.csv
    if updates:
        with open(out_dir / "updates.csv", "w", newline = "") as f:
            w = csv.DictWriter(f, fieldnames = list(updates[0].keys()))
            w.writeheader()
            w.writerows(updates)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent = 2)

    if not rec.steps_log:
        return
    steps = np.array([r["step"] for r in rec.steps_log])
    t = np.array([r["t"] for r in rec.steps_log])
    rewards = np.array([r["reward"] for r in rec.steps_log], dtype = np.float64)
    win = min(ROLLING_WINDOW, len(rewards))
    rolling = np.convolve(rewards, np.ones(win) / win, mode = "valid")
    rolling_steps = steps[win - 1:]

    fig, axes = plt.subplots(3, 1, figsize = (11, 10), sharex = True,
                             gridspec_kw = {"height_ratios": [3, 2, 1.5]})
    ax = axes[0]
    ax.plot(rolling_steps, rolling, color = "#1f77b4", lw = 1.8, label = f"reward / step, rolling mean ({win} steps)")
    if rec.rollout_marks:
        ms, mr = zip(*rec.rollout_marks)
        ax.plot(ms, mr, "o-", color = "#ff7f0e", ms = 4, lw = 1, alpha = 0.9, label = "mean reward per rollout")
    ax.axhline(0, color = "#999", lw = 0.8)
    ax.set_ylabel("average reward per step")
    ax.set_title(f"PPO on Rain World: {summary['seconds']:.0f} s of training, {steps[-1]} env steps, "
                 f"{summary['updates']} updates, total reward {rewards.sum():+.2f}")
    ax.legend(loc = "upper left")
    ax.grid(alpha = 0.3)

    ax = axes[1]
    for k in term_keys:
        series = np.cumsum([r.get(k, 0.0) for r in rec.steps_log])
        ax.plot(steps, series, lw = 1.5, label = k.split("/", 1)[1])
    ax.set_ylabel("cumulative reward by term")
    ax.legend(loc = "upper left", ncol = 3, fontsize = 9)
    ax.grid(alpha = 0.3)

    ax = axes[2]
    if updates:
        us = [u["env_step"] for u in updates]
        ax.plot(us, [u["entropy"] for u in updates], color = "#2ca02c", lw = 1.5, label = "policy entropy (nats, max 6.24)")
        ax.set_ylabel("entropy")
        ax.legend(loc = "lower left")
    ax.set_xlabel("env steps")
    ax.grid(alpha = 0.3)

    secax = axes[0].secondary_xaxis("top", functions = (
        lambda s: np.interp(s, steps, t), lambda tt: np.interp(tt, t, steps)))
    secax.set_xlabel("wall-clock seconds")
    fig.tight_layout()
    fig.savefig(out_dir / "reward_plot.png", dpi = 140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, argparse.Namespace]:
    p = argparse.ArgumentParser(
        prog = "python -m baselines.ppo.record_run",
        description = "Record a wall-clock-bounded PPO training run as video + reward plot. "
                      "Unknown flags are passed to train_ppo (e.g. --lr, --ent_coef, --seed).",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seconds", type = float, default = 300.0, help = "Training wall-clock budget")
    p.add_argument("--out", type = str, default = None, help = "Output directory (default baselines/runs/ppo_<timestamp>)")
    p.add_argument("--video_width", type = int, default = 320)
    p.add_argument("--video_height", type = int, default = 180)
    p.add_argument("--video_scale", type = int, default = 2, help = "Integer upscale of the recorded frame")
    p.add_argument("--fps", type = int, default = 30)
    p.add_argument("--rollout_steps", type = int, default = 512)
    p.add_argument("--fake", action = "store_true", help = "Use FakeRainWorldEnv (pipeline check)")
    p.add_argument("--launch", action = "store_true", help = "(Re)start Rain World before training")
    p.add_argument("--kill_game", action = "store_true", help = "Close the game at the end")
    p.add_argument("--no_wipe", action = "store_true", help = "Attach to the game in progress instead of RESET")
    mine, rest = p.parse_known_args(argv)
    ppo = ppo_parse_args(rest + ["--total_steps", str(10**9), "--rollout_steps", str(mine.rollout_steps)])
    return mine, ppo


def build_env(mine: argparse.Namespace, ppo: argparse.Namespace, rec: Recorder) -> gym.Env:
    from rainworld_rl.rewards import make_default_reward_env

    if mine.fake:
        base: gym.Env = FakeRainWorldEnv(mine.video_width, mine.video_height, ppo.ticks_per_step, seed = ppo.seed)
    else:
        from rainworld_rl import RainWorldEnv
        base = RainWorldEnv(mine.video_width, mine.video_height, ppo.ticks_per_step)
    env: gym.Env = make_default_reward_env(base)
    env = Tap(env, rec)
    env = ResizeObs(env, ppo.frame_width, ppo.frame_height)
    if not ppo.rgb:
        env = GrayscaleObs(env)
    return FrameStack(env, ppo.frame_stack)


def warm_up(train_state, ppo: argparse.Namespace, obs_shape: Tuple[int, ...]) -> None:
    """Trigger JIT compilation before the clock starts (on a throwaway state)."""
    dummy_obs = jnp.zeros(obs_shape, dtype = jnp.uint8)
    key = jr.PRNGKey(0)
    jax.block_until_ready(sample_action(train_state.model, dummy_obs, key))
    jax.block_until_ready(value_of(train_state.model, dummy_obs))
    n = ppo.rollout_steps
    batch = {
        "obs": jnp.zeros((n,) + obs_shape, dtype = jnp.uint8),
        "actions": jnp.zeros((n, 9), dtype = jnp.float32),
        "log_probs": jnp.zeros(n, dtype = jnp.float32),
        "values": jnp.zeros(n, dtype = jnp.float32),
        "rewards": jnp.zeros(n, dtype = jnp.float32),
        "last_value": jnp.zeros((), dtype = jnp.float32),
    }
    throwaway = init_experiment(ppo)          # identical structure -> same compiled executable
    _, m = train_step(throwaway, batch)
    jax.block_until_ready(m.total_loss)


def main(argv: Optional[List[str]] = None) -> int:
    mine, ppo = parse_args(argv)
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = Path(mine.out) if mine.out else DEFAULT_RUNS_ROOT / time.strftime("ppo_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents = True, exist_ok = True)
    logger.info("output -> %s", out_dir)

    rec = Recorder(out_dir, mine.fps, mine.video_scale, (mine.video_height, mine.video_width))
    env = build_env(mine, ppo, rec)
    train_state = init_experiment(ppo)
    obs_shape = tuple(env.observation_space.shape)

    updates: List[Dict[str, Any]] = []
    metrics = RolloutMetrics()
    summary: Dict[str, Any] = {}
    try:
        attach(env, launch = mine.launch)
        logger.info("warming up JIT...")
        tw = time.perf_counter()
        warm_up(train_state, ppo, obs_shape)
        logger.info("warm-up took %.1fs", time.perf_counter() - tw)

        obs, info = env.reset(options = {"wipe": not mine.no_wipe})
        metrics.start(info)
        rng = jr.PRNGKey(ppo.seed + 1)

        rec.start_clock()
        logger.info("training for %.0f s", mine.seconds)
        while rec.elapsed() < mine.seconds:
            t0 = time.perf_counter()
            batch, obs, rng, info = collect_rollout(
                env, train_state.model, obs, rng, ppo.rollout_steps, metrics, ppo.reward_scale
            )
            t1 = time.perf_counter()
            rec.mark_rollout(float(np.mean(np.asarray(batch["rewards"]))) / ppo.reward_scale)
            train_state, sm = train_step(train_state, batch)
            jax.block_until_ready(sm.total_loss)
            t2 = time.perf_counter()
            rec.on_update_done(t2 - t1, float(sm.entropy))
            row = {
                "update": rec.updates, "env_step": rec.env_steps, "t": round(rec.elapsed(), 2),
                "policy_loss": float(sm.policy_loss), "value_loss": float(sm.value_loss),
                "entropy": float(sm.entropy), "approx_kl": float(sm.approx_kl), "clipfrac": float(sm.clipfrac),
                "grad_norm": float(sm.grad_norm), "explained_variance": float(sm.explained_variance),
                "rollout_seconds": round(t1 - t0, 3), "update_seconds": round(t2 - t1, 3),
                "env_steps_per_second": round(ppo.rollout_steps / max(1e-9, t1 - t0), 1),
                "rollout_mean_reward": rec.rollout_marks[-1][1],
            }
            updates.append(row)
            logger.info("update %d | step %d | t %.0fs | r/step %+.4f | entropy %.3f | rooms %d | sps %.0f | upd %.1fs",
                        row["update"], row["env_step"], row["t"], row["rollout_mean_reward"], row["entropy"],
                        len(rec.rooms), row["env_steps_per_second"], row["update_seconds"])
    except KeyboardInterrupt:
        logger.warning("interrupted")
    finally:
        seconds = rec.elapsed()
        rec.close()
        ckpt = save_checkpoint(train_state.model, out_dir, rec.env_steps, model_config_from_args(ppo))
        summary = {
            "seconds": round(seconds, 1), "env_steps": rec.env_steps, "updates": rec.updates,
            "total_reward": round(rec.total_reward, 4), "reward_terms": {k: round(v, 4) for k, v in rec.term_totals.items()},
            "rooms_discovered": len(rec.rooms), "rooms": sorted(f"{r}:{i}" for r, i in rec.rooms),
            "deaths": rec.deaths, "cycles_survived": rec.slept,
            "video_frames": rec.frames_written, "fps": mine.fps,
            "env_steps_per_second_mean": round(float(np.mean([u["env_steps_per_second"] for u in updates])), 1) if updates else None,
            "update_seconds_mean": round(float(np.mean([u["update_seconds"] for u in updates])), 2) if updates else None,
            "checkpoint": str(ckpt), "ppo_args": vars(ppo), "record_args": vars(mine),
        }
        write_outputs(rec, updates, out_dir, summary)
        try:
            env.close()
        finally:
            if mine.kill_game and not mine.fake:
                from rainworld_rl.launcher import kill_game
                kill_game()
        logger.info("done: %s", json.dumps({k: summary[k] for k in ("seconds", "env_steps", "updates", "total_reward", "rooms_discovered", "deaths")}))
        logger.info("video: %s  plot: %s", out_dir / "video.mp4", out_dir / "reward_plot.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
