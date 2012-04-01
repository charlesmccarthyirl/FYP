from all_experiments import experiment as all_experiment
from utils import sub_pairs

names = ["Uncertainty Sampling", "Sparsity Minimization", "Diversity + Density Maximization", "Certainty + Sparsity Minimization"]

experiment = all_experiment.create_sub_experiment(names)