from all_experiments import experiment as all_experiment
from utils import sub_pairs

names = ["Local Reachability Total Counting Cross-Label Total Minimization",
         "CompStrat - Global Liability (Similarity)",
         "Local Coverage Total Direct Similarity Cross-Label Total Minimization",
         "Local Reachability Total Direct Similarity Cross-Label Total Minimization"]

experiment = all_experiment.create_sub_experiment(names)