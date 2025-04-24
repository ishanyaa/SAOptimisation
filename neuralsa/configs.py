# configs.py
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING


@dataclass
class TrainingConfig:
    method: str = "ppo"
    reward: str = "immediate"
    n_epochs: int = 100
    lr: float = 0.0002  # learning rate
    batch_size: int = 1024
    optimizer: str = "adam"  # "adam" or "sgd"
    # PPO params
    ppo_epochs: int = 10
    trace_decay: float = 0.9
    eps_clip: float = 0.25
    gamma: float = 0.9
    weight_decay: float = 0.01
    # ES params
    momentum: float = 0.9
    stddev: float = 0.05
    population: int = 16
    milestones: list = field(default_factory=lambda: [0.9])


@dataclass
class SAConfig:
    init_temp: float = 1.0
    stop_temp: float = 0.1
    outer_steps: int = 40  # number of steps at which temperature changes
    inner_steps: int = 1   # number of steps at a specific temperature
    alpha: float = MISSING  # will be set in main.py as a function of init/stop temps


@dataclass
class NeuralSAExperiment:
    # Problem settings
    n_problems: int = 256        # number of problems in a batch
    problem_dim: int = 20
    embed_dim: int = 16          # size of hidden layer in the actor network
    problem: str = "tsp"         # default changed from "knapsack" → "tsp"
    capacity: Optional[float] = field(default=None)

    # Training and SA sub-configs
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sa: SAConfig = field(default_factory=SAConfig)

    # Paths & device
    device: str = "cuda:0"
    model_path: Optional[str] = field(default=None)
    results_path: str = "results"
    data_path: str = "datasets"
    save_path: str = "outputs"   # ◀ newly added!

    # Reproducibility
    seed: int = 42
