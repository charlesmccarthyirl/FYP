from all_experiments import experiment as all_experiment
from utils import sub_pairs

names = ["Random Selection", "Uncertainty Sampling", "Margin Sampling", "Maximum Diversity Sampling"]

experiment = all_experiment.create_sub_experiment(names)