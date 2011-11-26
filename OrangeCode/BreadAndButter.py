import orange, orngStat, orngTest
from Eval import *
import glob, os
from functools import partial
import collections
import logging
import itertools

RANDOM_SEED = 42
DATASETS_DIR = "../Datasets/"
DATASET_EXTENSIONS = [".csv", ".tab", ".arff"]

oracle_generator = lambda *args, **kwargs: Oracle(orange.Example.get_class)
def classifier_generator(training_data, distance_constructor, *args, **kwargs): 
    return orange.kNNLearner(training_data, k=5, rankWeight=False, distanceConstructor=distance_constructor()) 

def get_training_test_sets_extractor(rand_seed):
    def split_data(data): 
        splits = n_fold_cross_validation(data, 10, rand_seed)
        return splits
    return split_data

def create_named_experiment_variations_generator(named_selection_strategy_generators):
    return lambda *args, **kwargs: dict([(name, 
                                          ExperimentVariation(classifier_generator, 
                                                              selection_strategy_generator))
                                         for (name, selection_strategy_generator) 
                                         in named_selection_strategy_generators.items()])
    
def create_experiment(stopping_condition_generator, named_experiment_variations, rand_seed=RANDOM_SEED):
    return Experiment(oracle_generator, 
                    stopping_condition_generator, 
                    get_training_test_sets_extractor(rand_seed), 
                    named_experiment_variations)
    
data_files = [os.path.join(DATASETS_DIR, filename)
              for filename in os.listdir(DATASETS_DIR) 
              if os.path.splitext(filename)[1].lower() in DATASET_EXTENSIONS]

data_files_dict = dict([(os.path.splitext(os.path.basename(df))[0], df) for df in data_files])

def euclidean_distance_constructor_generator(data):
    return orange.ExamplesDistanceConstructor_Euclidean

def load_data_distance_constructor_pair(
        base_filename, 
        random_seed=RANDOM_SEED, 
        #distance_constructor_generator=generate_example_distance_constructor_generator()):
        distance_constructor_generator=generate_precomputed_example_distance_constructor_generator()):
        #distance_constructor_generator=euclidean_distance_constructor_generator):
    d = orange.ExampleTable(data_files_dict[base_filename], randomGenerator=orange.RandomGenerator(random_seed))
    
    if not d.domain.has_meta("ex_id"):
        var = orange.FloatVariable("ex_id")
        varId = orange.newmetaid()
        d.domain.add_meta(varId, var)
        i = 0
        for ex in d:
            ex.set_meta(varId, i)
            i += 1
        
    distance_constructor = distance_constructor_generator(d)
    
    d.shuffle() # Could all be clustered together in the file. Some of my operations might 
                # (and do . . .) go in order - so can skew the results *a lot*.
    return (d, distance_constructor)


def create_named_data_set_generators(base_data_set_infos):
    base_data_set_infos = [info if isinstance(info, dict) 
                                else {"base_filename": info} 
                           for info in base_data_set_infos]
    return [(data_set_info["base_filename"], partial(load_data_distance_constructor_pair, **data_set_info)) 
            for data_set_info in base_data_set_infos]
    

