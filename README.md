# Neural Simulated Annealing (Neural SA)

This repository is a detailed implementation of the Neural Simulated Annealing (Neural SA) framework as proposed in the paper **"Neural Simulated Annealing"** by Correia et al., presented at AISTATS 2023. This framework is designed for solving combinatorial optimization problems, such as the Knapsack problem, using techniques like Proximal Policy Optimization (PPO) and Simulated Annealing (SA).

---

## 🧰 Requirements and Setup

- **Python version**: `Python 3.11` is required.
- Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🛠 Directory Structure (Important Folders)

- `scripts/`: Contains the main training, evaluation, and result printing scripts.
- `scripts/conf/experiment/`: YAML configuration files for each experiment.
- `neuralsa/`: Core implementation of Neural SA, PPO, and configuration management.
- `outputs/models/`: Stores trained model checkpoints.
- `outputs/results/`: Stores evaluation results and logs.

---

## 🧪 How to Train (or Retrain) Models

Run the main script with the desired experiment configuration:

```bash
python scripts/main.py +experiment=<config_file>
```

### 🔹 Example (Knapsack using PPO)

```bash
python scripts/main.py +experiment=knapsack_ppo
```

### 🔸 Custom Configuration Using Hydra Overrides

You can modify variables inline without creating a new config file:

```bash
python scripts/main.py +experiment=knapsack_ppo ++problem_dim=100 ++sa.outer_steps=500
```

- Use `sa.` prefix for parameters from `SAConfig`
- Use `training.` prefix for parameters from `TrainingConfig`

#### 🔄 Model Output Location

Models are saved at:

```
outputs/models/<problem><problem_dim>-<training.method>
```

---

## 📊 Evaluation of Models

Evaluate trained models with the same setting used in the paper:

```bash
python scripts/eval.py +experiment=knapsack_ppo
```

This performs evaluation across multiple:
- SA step sizes (`sa.outer_steps`)
- Methods: PPO, vanilla SA, Greedy SA

### 🗂 Results Output

```
outputs/results/<problem>
```

---

## 📈 Print and Aggregate Results

After evaluation, print and compare results using:

```bash
python scripts/print_results.py +experiment=knapsack_ppo
```

This script summarizes results from multiple runs and outputs comparative tables.

---

## 🧾 YAML Configuration Files

You can find the default experiment configurations in:

```
scripts/conf/experiment/
```

Each YAML file follows the format `<problem>_<method>.yaml`, e.g.:
- `knapsack_ppo.yaml`
- `knapsack_es.yaml`

---

## 🧠 Reference

If you find our work or this implementation useful, please cite:

```bibtex
@inproceedings{correia2023neural,
  title={Neural simulated annealing},
  author={Correia, Alvaro HC and Worrall, Daniel E and Bondesan, Roberto},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={4946--4962},
  year={2023},
  organization={PMLR}
}
```

---

## 👥 Credits and Acknowledgements

This implementation and documentation were prepared by:

- **Ishanya** (21329)
- **Hari Krishna** (22236)
- **Hiba** (22146)
- **Astha** (22063)

Special thanks to the original authors of the Neural SA framework and Hydra configuration library.

---

## ✏️ Notes for Developers

- You are encouraged to add or modify YAML files in `scripts/conf/experiment/` to explore new configurations.
- Feel free to tune hyperparameters via command-line overrides.
- Always verify the Python version compatibility (tested on Python 3.11).

---

