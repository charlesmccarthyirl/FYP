from all_experiments import experiment as all_experiment
from utils import sub_pairs

names = ["Local Reachability Total Direct Similarity (With Sparsity) Cross-Label Total Minimization",
         "Local Coverage Total All-Pairs Similarity (excl source) (With Sparsity) Cross-Label Deviation Minimization",
         "Local Liability + Reachability + Coverage Total All-Pairs Similarity (incl source) (With Sparsity) Cross-Label Deviation Minimization",
         "Local Liability + Coverage Total Counting Cross-Label Total Minimization"]

experiment = all_experiment.create_sub_experiment(names)