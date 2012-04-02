'''
Created on Mar 19, 2012

@author: charles
'''
import os
import glob
from utils import stream_getter, my_import
from os.path import basename, splitext
from functools import partial
from SelectionStrategyEvaluator import BudgetBasedStoppingCriteria

def get_stream_from_name_getter_for(raw_results_dir):
    def internal(variation_name):
        return stream_getter(tgz_filename_getter(variation_name, raw_results_dir))
    return internal

def tgz_filename_getter(variation_name, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    return os.path.join(path, variation_name + '.tar.gz')

def get_experiment_obj(experiment):
    if isinstance(experiment, str):
        experiment_obj = my_import(experiment).experiment
    else:
        experiment_obj = experiment
    
    return experiment_obj

def get_named_data_sets_obj(named_data_sets):
    if isinstance(named_data_sets, str):
        named_data_sets_obj = my_import(named_data_sets).named_data_sets
    else:
        named_data_sets_obj = named_data_sets
    return named_data_sets_obj

def get_frp_rrp(experiment_directory, data_set_name):
    full_result_path = os.path.join(experiment_directory, data_set_name)
    raw_results_dir = os.path.join(full_result_path, "raw_results")
    return (full_result_path, raw_results_dir)
    
def get_existing_variation_results_nfs_pairs(raw_results_dir):
    files = glob.glob(os.path.join(raw_results_dir, "*.tar.gz"))
    return  [(splitext(splitext(basename(f))[0])[0], partial(open, f, "rb")) for f in files]

def get_exp_ds_pair(experiment, data_set_generator):
    l_experiment = experiment.copy()
    
    if hasattr(data_set_generator, 'sc'):
        sc = getattr(data_set_generator, 'sc')
        if isinstance(sc, int):
            i = sc # Can't go using sc in the lambda, and assigning to it. Will be passing a function then.
            sc = lambda *args, **kwargs: BudgetBasedStoppingCriteria(i)
            
        l_experiment.stopping_condition_generator = sc
    
    return (l_experiment, data_set_generator)