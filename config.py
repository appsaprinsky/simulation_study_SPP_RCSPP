import os
from pathlib import Path

# --- Directory Structure ---
BASE_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = BASE_DIR / "graphs"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

for d in [GRAPHS_DIR, FIGURES_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Global Experiment Settings ---
GRAPH_SIZES = [5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
NUM_RUNS = 1
TIME_LIMIT_SEC = 500  # 5 minutes
MASTER_SEED = 42

# --- Metaheuristic Hyperparameters ---
PENALTY_MULTIPLIER = 1000.0  # For soft constraint violations

GA_PARAMS = {
    "pop_size": 100,
    "generations": 200,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8
}

PSO_PARAMS = {
    "swarm_size": 80,
    "iterations": 200,
    "w": 0.729,   # Inertia
    "c1": 1.494,  # Cognitive
    "c2": 1.494   # Social
}

SQA_PARAMS = {
    "num_reads": 100,
    "num_sweeps": 1000,
    "beta_range": (0.1, 10.0),
    "flow_penalty": 150.0,
    "cap_penalty": 50.0
}