'''
Created on Mar 20, 2012

@author: charles
'''
import logging
from main_runner_utils import *
import multiprocessing as mp

def main_gen_raw_results_only(experiment, named_data_sets, 
                              experiment_directory, data_set_name, 
                              variation_name):
    experiment_obj = get_experiment_obj(experiment)
    named_data_sets_obj = get_named_data_sets_obj(named_data_sets)
    full_result_path, raw_results_dir = get_frp_rrp(experiment_directory, data_set_name)
    
    named_data_sets_dict = dict(named_data_sets_obj)
    data_set_generator = named_data_sets_dict[data_set_name]
    
    l_experiment, data_info_generator = get_exp_ds_pair(experiment_obj, data_set_generator)
    named_experiment_variations = l_experiment.named_experiment_variations
    
    named_experiment_variations_dict = dict(named_experiment_variations)
    variation = named_experiment_variations_dict[variation_name]
    
    variation_result = l_experiment.execute_on_only(data_info_generator(), variation)
    stream_from_name_getter = get_stream_from_name_getter_for(raw_results_dir)
    
    with stream_from_name_getter(variation_name) as stream:
        variation_result.serialize(stream)

def main_gen_raw_results(experiment, named_data_sets, experiment_directory, do_multi, *margs, **mkwargs):
    if do_multi:
        mp.log_to_stderr()
        logger = mp.get_logger()
        logger.setLevel(logging.INFO)
        pool = mp.Pool()
        my_apply = pool.apply_async
    else:
        logger = logging.info
        my_apply = lambda ex, args: ex(*args)
    
    experiment_obj = get_experiment_obj(experiment)
    named_data_sets_obj = get_named_data_sets_obj(named_data_sets)
    
    for (data_set_name, data_set_generator) in named_data_sets_obj:
        logging.info("Beginning processing on %s" % data_set_name)
        l_experiment, data_info_generator = get_exp_ds_pair(experiment_obj, data_set_generator)
        named_experiment_variations = l_experiment.named_experiment_variations
        
        full_result_path, raw_results_dir = get_frp_rrp(experiment_directory, data_set_name)
        name_to_file_stream_getter_pairs = get_existing_variation_results_nfs_pairs(raw_results_dir)
        existing_variations_computed = set((name for (name, fsg) in name_to_file_stream_getter_pairs))
        
        for (variation_name, variation) in named_experiment_variations:
            if variation_name in existing_variations_computed:
                logging.info("Already have results for %s. Skipping evaluation." % variation_name)
                continue
            
            logging.info("Starting evaluation on variation %s" % variation_name)
            
            my_apply(main_gen_raw_results_only, 
                         (experiment, named_data_sets, experiment_directory, 
                          data_set_name, variation_name))
    if do_multi:
        pool.close()
        pool.join()