from all_experiments import experiment as all_experiment
from utils import sub_pairs

names = ["Certainty + Sparsity Minimization",
         "Local Reachability Total Direct Similarity (With Sparsity) Cross-Label Total Minimization",
         "Local Coverage Total Direct Similarity (With Sparsity) Cross-Label Deviation Minimization",
         "Local Coverage Average All-Pairs Similarity (incl source) (With Sparsity) Cross-Label Deviation Minimization"]

experiment = all_experiment.create_sub_experiment(names)