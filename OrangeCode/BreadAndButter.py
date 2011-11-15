import orange, orngStat, orngTest
from Eval import *
import glob, os
from functools import partial
import collections


RANDOM_SEED = 42
DATASETS_DIR = "../Datasets/"
DATASET_EXTENSIONS = [".csv", ".tab", ".arff"]

oracle_generator = lambda *args, **kwargs: Oracle(orange.Example.get_class)
classifier_generator = lambda training_data, *args, **kwargs: orange.kNNLearner(training_data, k=5, rankWeight=False) 

def get_training_test_sets_extractor(rand_seed):
    return lambda data: n_fold_cross_validation(data, 10, rand_seed)

def create_named_experiment_variations(named_selection_strategy_generators):
    return dict([(name, ExperimentVariation(classifier_generator, selection_strategy_generator))
                for (name, selection_strategy_generator) in named_selection_strategy_generators.items()])
    
def create_experiment(stopping_condition_generator, named_experiment_variations, rand_seed=RANDOM_SEED):
    return Experiment(oracle_generator, 
                    stopping_condition_generator, 
                    get_training_test_sets_extractor(rand_seed), 
                    named_experiment_variations)
    
data_files = [os.path.join(DATASETS_DIR, filename)
              for filename in os.listdir(DATASETS_DIR) 
              if os.path.splitext(filename)[1].lower() in DATASET_EXTENSIONS]

data_files_dict = dict([(os.path.splitext(os.path.basename(df))[0], df) for df in data_files])

def load_data(base_filename, random_seed=RANDOM_SEED):
    d = orange.ExampleTable(data_files_dict[base_filename], randomGenerator=orange.RandomGenerator(random_seed))
    d.shuffle() # Could all be clustered together in the file. Some of my operations might 
                # (and do . . .) go in order - so can skew the results *a lot*.
    return d


def create_named_data_set_generators(*base_data_set_names):
    base_data_set_names = [name_pair if name_pair is collections.Iterable 
                                     else (name_pair,) 
                           for name_pair in base_data_set_names]
    return [(data_set_info[0], partial(load_data, *data_set_info)) for data_set_info in base_data_set_names]
    

