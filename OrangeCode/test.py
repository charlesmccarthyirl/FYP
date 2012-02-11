from SelectionStrategyEvaluator import *
import logging
import cProfile
import itertools
import os, sys, glob
from os.path import basename, splitext
from functools import partial
from utils import stream_getter, uniqueify
import csv
import functools
from functools import partial
import logging
import latexcodec
import multiprocessing as mp

def tgz_filename_getter(variation_name, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    return os.path.join(path, variation_name + '.tar.gz')

def get_experiment_obj(experiment):
    if isinstance(experiment, str):
        experiment_obj = __import__(experiment).experiment
    else:
        experiment_obj = experiment
    
    return experiment_obj

def get_named_data_sets_obj(named_data_sets):
    if isinstance(named_data_sets, str):
        named_data_sets_obj = __import__(named_data_sets).named_data_sets
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
    
    variation_result = l_experiment.execute_on_only(data_info_generator, variation)
    stream_from_name_getter = get_stream_from_name_getter_for(raw_results_dir)
    
    with stream_from_name_getter(variation_name) as stream:
        variation_result.serialize(stream)

def log_throwaway(to_log):
    def inner(*args):
        mp.get_logger().info(to_log)

def main_gen_raw_results(experiment, named_data_sets, experiment_directory, do_multi=False):
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
        
def get_stream_from_name_getter_for(raw_results_dir):
    def internal(variation_name):
        return stream_getter(tgz_filename_getter(variation_name, raw_results_dir))
    return internal

def main(experiment, named_data_sets, experiment_directory,
        do_create_summary=True, latex_encode=True, do_multi=True):
    logging.info("Beginning generating raw results")
    if do_multi:
        main_gen_raw_results(experiment, named_data_sets, experiment_directory, do_multi)
        logging.info("Ending generating raw results.")
    
    logging.info("Beginning Nicity Processing.")
    experiment = get_experiment_obj(experiment)
    named_data_sets = get_named_data_sets_obj(named_data_sets)
    
    summary_results = OrderedDict()
    
    for (data_set_name, data_set_generator) in named_data_sets:
        logging.info("Beginning processing on %s" % data_set_name)
        
        l_experiment, data_info_generator = get_exp_ds_pair(experiment, data_set_generator)
        
        full_result_path, raw_results_dir = get_frp_rrp(experiment_directory, data_set_name)
        
        name_to_file_stream_getter_pairs = get_existing_variation_results_nfs_pairs(raw_results_dir)
        
        existing_results = ExperimentResult()
        existing_results.load_from_csvs(name_to_file_stream_getter_pairs)
        
        stream_from_name_getter = get_stream_from_name_getter_for(raw_results_dir)
        results = l_experiment.execute_on(data_info_generator, existing_results, 
                                          stream_from_name_getter=stream_from_name_getter)

        if do_create_summary:
            summary_results[data_set_name] = OrderedDict([(var_name, var_result.AULC()) 
                                                   for (var_name, var_result) 
                                                   in results.items()])
            
        try:
            g = results.generate_graph(data_set_name)
            g.writePDFfile(os.path.abspath(full_result_path)) # Yes this is intentional, want it in the experiment directory, but with the same name as the folder.
        except ImportError, ex:
            logging.info("Unable to generate graph for %s data set. Graphing module unavailable in system: %s" %(data_set_name, ex)) 

    if do_create_summary:
        logging.info("Beginning summary csv generation")
        
        with open(os.path.join(experiment_directory, "summary.csv"), 'wb') as summary_stream:
            writer = csv.writer(summary_stream)
            writerow = writer.writerow
            format_num = lambda n: n if n is None else "%.3f" % n
            
            if latex_encode:
                str_encoder = lambda s: s.encode('latex')
                highlight = lambda x: "\\textbf{%s}" % x
            else:
                str_encoder = lambda s: s
                highlight = lambda x: x
            
            # Get all the union of all the variations names. 
            variations = uniqueify(itertools.chain(*[r.keys() for r in summary_results.values()]))
            data_names = summary_results.keys()
            
            top_results = [max(r.values()) for r in (summary_results[dn] for dn in data_names)]
            
            #None at start to leave column for variation names
            writerow([None] + map(str_encoder, data_names))
            
            for variation in variations:
                variation_results = [r.get(variation, None) for r in (summary_results[dn] for dn in data_names)]
                variation_results_highlighted = [highlight(format_num(v)) if v == t else format_num(v) for (v, t) in zip(variation_results, top_results)]
                row = [str_encoder(variation)] + variation_results_highlighted
                writerow(row)
            
        logging.info("Ending summary csv generation")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
    experiment = sys.argv[1]
    named_data_sets = sys.argv[2]
    experiment_directory = os.path.expanduser(sys.argv[3])
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)
    main(experiment, named_data_sets, experiment_directory)
#    cProfile.run("main(experiment, named_data_sets, experiment_directory)", "mainProfile")
