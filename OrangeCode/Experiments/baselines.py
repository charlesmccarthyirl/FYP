from all import experiment as all_experiment
from ..utils import sub_pairs

names = ["Random Selection", "Uncertainty Sampling", "Margin Sampling", "Maximum Diversity Sampling",
         "Sparsity Minimization", "Diversity + Density Maximization", "Certainty + Sparsity Minimization"]

experiment = all_experiment.create_sub_experiment(names)