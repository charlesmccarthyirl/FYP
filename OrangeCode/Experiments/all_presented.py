from all_experiments import experiment as all_experiment
from utils import sub_pairs, uniqueify
import baselines, baseline_sparsity, competence_counting, competence_hybrids, competence_with_similarity, competence_with_sparsity

names = (baselines.names + baseline_sparsity.names + competence_counting.names + competence_hybrids.names 
         + competence_with_similarity.names + competence_with_sparsity.names)

names = uniqueify(names)

experiment = all_experiment.create_sub_experiment(names)