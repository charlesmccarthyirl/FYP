from __future__ import division
from SelectionStrategyEvaluator import *
from PrecomputedDistance import *
from Knn import *
from utils import stream_getter
from functools import partial
from DataInfoLoaders import get_data_info
import os, logging, itertools, random
from SelectionStrategy import SingleCompetenceSelectionStrategy
from CaseProfiling import CaseProfileBuilder

RANDOM_SEED = 42
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "../Datasets/")

oracle_generator_generator = lambda true_oracle: lambda data, possible_classes, *args, **kwargs: Oracle(true_oracle)
random_seed_generator=lambda: RANDOM_SEED

k=5

def get_knn_for(data, distance_constructor, possible_classes, oracle):
    return KNN(data, k, distance_constructor(data), 
               oracle, possible_classes)

def get_knn_action(data, distance_constructor, possible_classes, oracle, thing_to_do):
    return lambda ex: thing_to_do(get_knn_for(data, distance_constructor, possible_classes, oracle), ex)

def classifier_generator(training_data, distance_constructor, possible_classes, oracle, *args, **kwargs):
    return get_knn_action(training_data, distance_constructor, possible_classes, oracle, KNN.classify)

def probability_generator(training_data, distance_constructor, possible_classes, oracle, *args, **kwargs):
    return get_knn_action(training_data, distance_constructor, possible_classes, oracle, KNN.get_probabilities)

def nns_getter_generator(training_data, distance_constructor, possible_classes, oracle, *args, **kwargs):
    return get_knn_action(training_data, distance_constructor, possible_classes, oracle, KNN.find_nearest)

def get_training_test_sets_extractor(random_seed_generator=random_seed_generator):
    def split_data(data, true_oracle):
        splits = n_fold_cross_validation(data, 10, true_oracle=true_oracle, 
                                         random_seed=random_seed_generator())
        return splits
    return split_data

def create_named_experiment_variations_generator(named_selection_strategy_generators):
    return lambda *args, **kwargs: dict([(name, 
                                          ExperimentVariation(classifier_generator, 
                                                              probability_generator, 
                                                              nns_getter_generator,
                                                              selection_strategy_generator))
                                         for (name, selection_strategy_generator) 
                                         in named_selection_strategy_generators.items()])
    
def create_experiment(stopping_condition_generator, named_experiment_variations, random_seed_generator=random_seed_generator):
    return Experiment(oracle_generator_generator,
                      stopping_condition_generator, 
                      get_training_test_sets_extractor(random_seed_generator), 
                      named_experiment_variations)
    
data_files = [os.path.join(DATASETS_DIR, filename)
              for filename in os.listdir(DATASETS_DIR)]

data_files_dict = dict(((os.path.basename(df), df) for df in data_files))

def load_data_info(
        base_filename, 
        random_seed_generator=random_seed_generator, 
        distance_constructor=None,
        **kwargs):
    filename = data_files_dict[base_filename]
    
    logging.info("Beginning load_data_info of %s" % base_filename)
    data_info = get_data_info(filename, distance_constructor)
    data_info = data_info.get_precached()
    logging.info("Ending load_data_info")
    
    if random_seed_generator is not None:
        random_seed = random_seed_generator()
        data_info.data = list(data_info.data)
        random_function = random.Random(random_seed).random
        random.shuffle(data_info.data, random_function)
    
    for (k, v) in kwargs.items():
        setattr(data_info, k, v)
    
    return data_info

def _translate_bit(bit):
    if isinstance(bit, dict) :
        return bit
    elif isinstance(bit, str):
        return  {"base_filename": bit}
    elif isinstance(bit, tuple):
        d = dict(bit[1])
        d['base_filename'] = bit[0]
        return d

def create_named_data_set_generators(base_data_set_infos):
    base_data_set_infos = map(_translate_bit, base_data_set_infos)
    return [((data_set_info["base_filename"]).split(".")[0], partial(load_data_info, **data_set_info)) 
            for data_set_info in base_data_set_infos]

def gen_case_profile_ss_generator(case_profile_measure_generator, op=SingleCompetenceSelectionStrategy.take_maximum):
    def case_profile_selection_strategy_generator(*args, **kwargs): 
        case_profile_builder=CaseProfileBuilder(k, *args, **kwargs)
        return SingleCompetenceSelectionStrategy(
            case_profile_measure_generator, 
            op, 
            *args,
            case_profile_builder=case_profile_builder,
            on_selection_action=lambda example: case_profile_builder.put(example),
            **kwargs)
    
    return case_profile_selection_strategy_generator

def gen_case_profile_ss_generator2(competence_ss, *args, **kwargs):
    def case_profile_selection_strategy_generator(*args, **kwargs): 
        case_profile_builder=CaseProfileBuilder(k, *args, **kwargs)
        return competence_ss(
            *args,
            case_profile_builder=case_profile_builder,
            **kwargs)
    
    return case_profile_selection_strategy_generator