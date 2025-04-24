import os
import random
import time  # Added for timing ES
import csv

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
# Unified optimizer imports
from torch.optim import SGD, Adam, AdamW, Adagrad, RMSprop, RAdam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from neuralsa.configs import NeuralSAExperiment
from neuralsa.model import (
    BinPackingActor,
    BinPackingCritic,
    KnapsackActor,
    KnapsackCritic,
    TSPActor,
    TSPCritic,
)
from neuralsa.problem import TSP, BinPacking, Knapsack
from neuralsa.sa import sa
from neuralsa.training import EvolutionStrategies
from neuralsa.training.ppo import ppo
from neuralsa.training.replay import Replay

# For reproducibility on GPU
torch.backends.cudnn.deterministic = True


def create_folder(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created: {dirname}")


def get_optimizer(name, params, lr, weight_decay=0.0, momentum=0.0):
    """
    Returns an optimizer instance based on name.
    Supports: sgd, adam, adamw, adagrad, rmsprop, radam
    """
    name = name.lower()
    if name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "adagrad":
        return Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif name == "radam":
        return RAdam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def train_es(actor, problem, init_x, es, cfg, epoch, log_writer):
    start_time = time.time()  # Added timing
    with torch.no_grad():
        es.zero_updates()
        epoch_objectives = []

        for _ in range(es.population):
            es.perturb(antithetic=True)
            results = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False)
            loss = torch.mean(results[cfg.training.reward])
            epoch_objectives.append(loss.item())
            es.collect(loss)

        es.step(reshape_fitness=True)

    mean_obj = np.mean(epoch_objectives)
    std_obj = np.std(epoch_objectives)
    best_obj = np.min(epoch_objectives)
    elapsed = time.time() - start_time

    train_loss = torch.tensor(mean_obj)
    # Enhanced logging for ES
    log_writer.writerow({
        "Epoch": epoch + 1,
        "TrainLoss": train_loss.item(),
        "MeanObjective": mean_obj,
        "BestObjective": best_obj,
        "FitnessStd": std_obj,
        "Stddev": cfg.training.stddev,
        "LR": es.optimizer.param_groups[0]['lr'],
        "TimeSec": elapsed,
        "Optimizer": cfg.training.optimizer
    })

    return train_loss


def train_ppo(actor, critic, actor_opt, critic_opt, problem, init_x, cfg):
    replay = Replay(cfg.sa.outer_steps * cfg.sa.inner_steps)
    sa(actor, problem, init_x, cfg, replay=replay, baseline=False, greedy=False)
    ppo(actor, critic, replay, actor_opt, critic_opt, cfg)


cs = ConfigStore.instance()
cs.store(name="base_config", node=NeuralSAExperiment, group="experiment")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: NeuralSAExperiment) -> None:
    if "cuda" in cfg.device and not torch.cuda.is_available():
        cfg.device = "cpu"
        print("CUDA device not found. Running on CPU.")

    # Temperature decay parameter
    alpha = np.log(cfg.sa.stop_temp) - np.log(cfg.sa.init_temp)
    cfg.sa.alpha = np.exp(alpha / cfg.sa.outer_steps).item()

    print(OmegaConf.to_yaml(cfg))

    # Set seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Initialize problem, actor, critic
    if cfg.problem == "knapsack":
        problem = Knapsack(
            cfg.problem_dim, cfg.n_problems, device=cfg.device, params={"capacity": cfg.capacity}
        )
        actor = KnapsackActor(cfg.embed_dim, device=cfg.device)
        critic = KnapsackCritic(cfg.embed_dim, device=cfg.device)
    elif cfg.problem == "binpacking":
        problem = BinPacking(cfg.problem_dim, cfg.n_problems, device=cfg.device)
        actor = BinPackingActor(cfg.embed_dim, device=cfg.device)
        critic = BinPackingCritic(cfg.embed_dim, device=cfg.device)
    elif cfg.problem == "tsp":
        problem = TSP(cfg.problem_dim, cfg.n_problems, device=cfg.device)
        actor = TSPActor(cfg.embed_dim, device=cfg.device)
        critic = TSPCritic(cfg.embed_dim, device=cfg.device)
    else:
        raise ValueError("Invalid problem name.")

    problem.manual_seed(cfg.seed)

    # Optimizer and scheduler setup
    es_log_writer = None
    if cfg.training.method == "ppo":
        # Unified optimizer selection for PPO
        actor_opt = get_optimizer(
            cfg.training.optimizer,
            actor.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.training.momentum
        )
        critic_opt = get_optimizer(
            cfg.training.optimizer,
            critic.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.training.momentum
        )
        # CSV logging for PPO (original behavior)
        log_file = os.path.join(os.getcwd(), "optimizer_comparison.csv")
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["optimizer", "epoch", "train_loss"]);
    elif cfg.training.method == "es":
        # Use unified get_optimizer for ES too
        optimizer = get_optimizer(
            cfg.training.optimizer,
            actor.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.training.momentum
        )
        es = EvolutionStrategies(optimizer, cfg.training.stddev, cfg.training.population)
        milestones = [int(cfg.training.n_epochs * m) for m in cfg.training.milestones]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        # Enhanced ES logging setup
        log_dir = os.path.join(os.getcwd(), "outputs")
        create_folder(log_dir)
        log_path = os.path.join(log_dir, "es_train_log.csv")
        log_file_handle = open(log_path, mode='w', newline='')
        fieldnames = [
            "Epoch", "TrainLoss", "MeanObjective", "BestObjective",
            "FitnessStd", "Stddev", "LR", "TimeSec", "Optimizer"
        ]
        es_log_writer = csv.DictWriter(log_file_handle, fieldnames=fieldnames)
        es_log_writer.writeheader()
    else:
        raise ValueError("Invalid training method.")

    # Training loop
    with tqdm(range(cfg.training.n_epochs)) as t:
        for i in t:
            params = problem.generate_params()
            params = {k: v.to(cfg.device) for k, v in params.items()}
            problem.set_params(**params)
            init_x = problem.generate_init_x()
            actor.manual_seed(cfg.seed)

            if cfg.training.method == "ppo":
                train_ppo(actor, critic, actor_opt, critic_opt, problem, init_x, cfg)
                train_out = sa(actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False)
                train_loss = torch.mean(train_out[cfg.training.reward])
                # Append PPO log
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([cfg.training.optimizer, i+1, train_loss.item()])
            elif cfg.training.method == "es":
                train_loss = train_es(actor, problem, init_x, es, cfg, i, es_log_writer)
                scheduler.step()

            t.set_description(f"Training loss: {train_loss:.4f}")

            # Save model checkpoint
            path = os.path.join(os.getcwd(), "models")
            name = f"{cfg.problem}{cfg.problem_dim}-{cfg.training.method}.pt"
            create_folder(path)
            torch.save(actor.state_dict(), os.path.join(path, name))

    # Close ES log file
    if cfg.training.method == "es" and es_log_writer is not None:
        log_file_handle.close()


if __name__ == "__main__":
    main()
