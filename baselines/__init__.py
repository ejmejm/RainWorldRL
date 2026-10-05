"""Baseline agents for the Rain World RL environment (JAX + Equinox + Optax + MLflow)."""

import os

# MLflow >= 3.16 prints an agent hint on import; silence it for every baselines entry point.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
