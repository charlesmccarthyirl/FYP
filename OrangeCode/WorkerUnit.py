'''
Created on Mar 19, 2012

@author: charles
'''

from utils import getitem
from operator import xor
from main_runner_utils import *
import logging
from collections import defaultdict
from SelectionStrategyEvaluator import MultiResultSet
from itertools import groupby

def main_gen_results_set_on_fold(experiment, named_data_sets, data_set_name, variation_name, fold_num):
    experiment_obj = get_experiment_obj(experiment)
    named_data_sets_obj = get_named_data_sets_obj(named_data_sets)
    named_data_sets_dict = dict(named_data_sets_obj)
    data_set_generator = named_data_sets_dict[data_set_name]
    l_experiment, data_info_generator = get_exp_ds_pair(experiment_obj, data_set_generator)
    named_experiment_variations = l_experiment.named_experiment_variations
    named_experiment_variations_dict = dict(named_experiment_variations)
    
    data_info = data_info_generator()
    variation = named_experiment_variations_dict[variation_name]
    evaluator = l_experiment.get_selection_strategy_evaluator(data_info, variation)
    
    training_test_tuples = l_experiment.training_test_sets_extractor(data_info.data, data_info.oracle)
    training_test_tuple = getitem(training_test_tuples, fold_num)
    
    return evaluator.generate_results(*training_test_tuple)

def main_gen_work_unit_result(work_unit):
    result = main_gen_results_set_on_fold(work_unit.variation_info.exp_name,
                                        work_unit.variation_info.data_file_name, 
                                        work_unit.variation_info.data_set_name, 
                                        work_unit.variation_info.variation_name, 
                                        work_unit.fold_num)
    return WorkUnitResult(work_unit, result)

class VariationInfo:
    meta_info = ['data_set_name', 'variation_name', 'exp_name', 'data_file_name', 'raw_results_file', 'total_folds' ]
    
    def __init__(self, data_set_name, variation_name, exp_name, data_file_name, raw_results_file, total_folds):
        self.data_set_name = data_set_name
        self.variation_name = variation_name
        self.exp_name = exp_name
        self.data_file_name = data_file_name
        self.raw_results_file = raw_results_file
        self.total_folds = total_folds
    
    def __eq__(self, other):
        return (isinstance(other, VariationInfo) 
                and all((getattr(self, f) == getattr(other, f) for f in self.meta_info)))
    
    def __hash__(self):
        return reduce(xor, map(hash, (getattr(self, f) for f in self.meta_info)))
    
    def __str__(self):
        return ", ".join((str(getattr(self, mi)) for mi in self.meta_info))

class WorkUnit:
    def __init__(self, variation_info, fold_num):
        self.variation_info = variation_info
        self.fold_num = fold_num
        
    def __str__(self):
        return "%s, %d" % (self.variation_info, self.fold_num)
    
class WorkUnitResult:
    def __init__(self, work_unit, result):
        self.work_unit = work_unit
        self.result = result

def gen_work_units_iterable(experiment, named_data_sets, experiment_directory):
    experiment_obj = get_experiment_obj(experiment)
    named_data_sets_obj = get_named_data_sets_obj(named_data_sets)
    
    for (data_set_name, data_set_generator) in named_data_sets_obj:
        logging.info("Beginning loading ds info *only* on %s" % data_set_name)
        l_experiment, _ = get_exp_ds_pair(experiment_obj, data_set_generator)
        named_experiment_variations = l_experiment.named_experiment_variations
        
        _, raw_results_dir = get_frp_rrp(experiment_directory, data_set_name)
        name_to_file_stream_getter_pairs = get_existing_variation_results_nfs_pairs(raw_results_dir)
        existing_variations_computed = set((name for (name, fsg) in name_to_file_stream_getter_pairs))
        
        data_info = data_set_generator()
        
        training_test_tuples = None
        
        for (variation_name, _) in named_experiment_variations:
            if variation_name in existing_variations_computed:
                logging.info("Already have results for %s" %variation_name)
                continue
            
            if training_test_tuples is None:
                # Only want to execute this if there are variations to run. No point loading the data otherwise.
                training_test_tuples = list(l_experiment.training_test_sets_extractor(data_info.data, data_info.oracle))
                num_folds = len(training_test_tuples)
            
            fn = tgz_filename_getter(variation_name, raw_results_dir)
            variation_info = VariationInfo(data_set_name, variation_name, experiment, 
                                           named_data_sets, fn, num_folds)
            for fold_num in xrange(num_folds):
                yield WorkUnit(variation_info, fold_num)
    
