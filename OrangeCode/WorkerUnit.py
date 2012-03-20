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
    meta_info = ['data_set_name', 'variation_name', 'exp_name', 'data_file_name' ]
    
    def __init__(self, data_set_name, variation_name, exp_name, data_file_name):
        self.data_set_name = data_set_name
        self.variation_name = variation_name
        self.exp_name = exp_name
        self.data_file_name = data_file_name
    
    def __eq__(self, other):
        return (isinstance(other, VariationInfo) 
                and all((getattr(self, f) == getattr(other, f) for f in self.meta_info)))
    
    def data_set_name(self):
        return reduce(xor, map(hash, (getattr(self, f) for f in self.meta_info)))

class WorkUnit:
    def __init__(self, variation_info, fold_num):
        self.variation_info = variation_info
        self.fold_num = fold_num
    
class WorkUnitResult:
    def __init__(self, work_unit, result):
        self.work_unit = work_unit
        self.result = result

def gen_work_units_iterable(experiment, named_data_sets, experiment_directory):
    experiment_obj = get_experiment_obj(experiment)
    named_data_sets_obj = get_named_data_sets_obj(named_data_sets)
    
    for (data_set_name, data_set_generator) in named_data_sets_obj:
        logging.info("Beginning loading ds info *only* on %s" % data_set_name)
        l_experiment, data_info_generator = get_exp_ds_pair(experiment_obj, data_set_generator)
        named_experiment_variations = l_experiment.named_experiment_variations
        
        full_result_path, raw_results_dir = get_frp_rrp(experiment_directory, data_set_name)
        name_to_file_stream_getter_pairs = get_existing_variation_results_nfs_pairs(raw_results_dir)
        existing_variations_computed = set((name for (name, fsg) in name_to_file_stream_getter_pairs))
        
        data_info = data_set_generator()
        training_test_tuples = list(l_experiment.training_test_sets_extractor(data_info.data, data_info.oracle))
        num_folds = len(training_test_tuples)
        
        for (variation_name, variation) in named_experiment_variations:
            if variation_name in existing_variations_computed:
                logging.info("Already have results for %s" %variation_name)
                continue
            
            variation_info = VariationInfo(data_set_name, variation_name, experiment, named_data_sets)
            for fold_num in xrange(num_folds):
                yield WorkUnit(variation_info, fold_num)

class Worker:
    def __init__(self, experiment_directory):
        self.experiment_directory = experiment_directory
    
    def work_reducer(self, variation_info, work_unit_results):
        work_unit_results = sorted(work_unit_results, key=lambda wur: wur.work_unit.fold_num)
        all_results = [wur.result for wur in work_unit_results]
        variation_result = MultiResultSet(all_results)
        
        full_result_path, raw_results_dir = get_frp_rrp(self.experiment_directory, 
                                                        variation_info.data_set_name)
        stream_from_name_getter = get_stream_from_name_getter_for(raw_results_dir)
    
        with stream_from_name_getter(variation_info.variation_name) as stream:
            variation_result.serialize(stream)

def mapfn(k, work_unit):
    return (work_unit.variation_info, main_gen_work_unit_result(work_unit))

def main_gen_raw_results(experiment, named_data_sets, experiment_directory, do_multi):        
    logging.info("Beginning gen raw results")
    worker = Worker(experiment_directory)
    
    logging.info("Generating work units")
    work_units = list(gen_work_units_iterable(experiment, named_data_sets, experiment_directory))
    
    work_units.sort(key=lambda wu: wu.variation_info)
    if do_multi:
        import mincemeat
        logging.info("Beginning mince meat server")
        s = mincemeat.Server()
        s.datasource = dict(enumerate(work_units))
        s.mapfn = mapfn
        s.reducefn = worker.work_reducer
        s.run_server(password="changeme")
        logging.info("Ending mince meat server")
    else:
        work_unit_results = (main_gen_work_unit_result(work_unit) for work_unit in work_units)
    
        for key, group in groupby(work_unit_results, lambda wur: wur.work_unit.variation_info):
            group = list(group)
            worker.work_reducer(key, group)