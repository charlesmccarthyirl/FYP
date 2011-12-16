from __future__ import division
import orange, orngStat, orngTest
from SelectionStrategyEvaluator import *
from PrecomputedDistance import *
from Knn import *
import glob, os
from functools import partial
import collections
import logging
import itertools
import random
import functools
from collections import defaultdict

RANDOM_SEED = 42
DATASETS_DIR = "../Datasets/"
DATASET_EXTENSIONS = [".csv", ".tab", ".arff"]

true_oracle = lambda ex: ex.get_class().value
oracle_generator = lambda *args, **kwargs: Oracle(true_oracle)

k=5

def get_knn_for(data, distance_constructor, possible_classes):
    return KNN(data, k, distance_constructor()(data), true_oracle, possible_classes)

def get_orange_example_based_knn_action(data, distance_constructor, thing_to_do):
    knn = get_knn_for(data, distance_constructor, None)
    def thing(ex):
        if knn.possible_classes is None:
            knn.possible_classes = ex.domain.class_var.values.native()
        return thing_to_do(knn, ex)
    return thing

def classifier_generator(training_data, distance_constructor, *args, **kwargs):
    return get_orange_example_based_knn_action(training_data, distance_constructor, KNN.classify)

def probability_generator(training_data, distance_constructor, *args, **kwargs):
    return get_orange_example_based_knn_action(training_data, distance_constructor, KNN.get_probabilities)

def get_training_test_sets_extractor(rand_seed=None):
    def split_data(data):   
        splits = n_fold_cross_validation(data, 10, true_oracle, rand_seed=rand_seed)
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
                      true_oracle,
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
    
    prev_len = len(d)
    d.remove_duplicates()
    new_len = len(d)
    
    if (prev_len != new_len):
        logging.info("Removed %d duplicates contained in %s" % (prev_len - new_len, base_filename))
    
    # Have to add these after . . .
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