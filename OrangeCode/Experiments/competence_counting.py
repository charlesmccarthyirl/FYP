from all_experiments import experiment as all_experiment
from utils import sub_pairs

names = ["Uncertainty Sampling", "Local Reachability Total Counting Cross-Label Total Minimization",
         "Local Liability Total Counting Cross-Label Deviation Minimization",
         "CompStrat - Global Liability (Counting)"]

experiment = all_experiment.create_sub_experiment(names)