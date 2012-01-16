from SelectionStrategyEvaluator import *
import logging
import cProfile
import itertools
import os, sys, glob
from os.path import basename, splitext
from functools import partial
from utils import stream_getter
import functools
import logging

def tgz_filename_getter(variation_name, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    return os.path.join(path, variation_name + '.tar.gz')

def main(experiment, named_data_sets, experiment_directory, do_create_graphs=True, do_create_summary=True):
    # data_set_name -> example_table (pre-shuffled)
    
    summary_results = {}
    
    for (data_set_name, data_set_generator) in named_data_sets:
        l_experiment = experiment.copy()
        
        logging.info("Beginning processing on %s" % data_set_name)
        data_set = data_set_generator()
        
        if hasattr(data_set, 'sc'):
            sc = getattr(data_set, 'sc')
            if isinstance(sc, int):
                i = sc # Can't go using sc in the lambda, and assigning to it. Will be passing a function then.
                sc = lambda *args, **kwargs: BudgetBasedStoppingCriteria(i)
                
            l_experiment.stopping_condition_generator = sc
        
        full_result_path = os.path.join(experiment_directory, data_set_name)
        raw_results_dir = os.path.join(full_result_path, "raw_results")
        
        
        files = glob.glob(os.path.join(raw_results_dir, "*.tar.gz"))
        name_to_file_stream_getter_pairs = [(splitext(splitext(basename(f))[0])[0], partial(open, f, "rb")) for f in files]
        
        existing_results = ExperimentResult()
        existing_results.load_from_csvs(name_to_file_stream_getter_pairs)
        
        results = l_experiment.execute_on(data_set, existing_results)
        
        results.write_to_csvs(lambda variation_name: 
                              stream_getter(tgz_filename_getter(variation_name, raw_results_dir)))
        
        if do_create_summary:
            summary_results[data_set_name] = dict([(var_name, var_result.AULC()) 
                                                   for (var_name, var_result) 
                                                   in results.items()])
        
        if do_create_graphs:
            try:
                results.write_to_selection_graphs(lambda variation_name: stream_getter(tgz_filename_getter(variation_name, os.path.join(full_result_path, 'selection_graphs')), True), data_set)
            except ImportError, ex:
                logging.info("Unable to generate selection graphs for %s data set. Graphing module unavailable in system: %s" %(data_set_name, ex)) 
            
        try:
            g = results.generate_graph(data_set_name)
            g.writePDFfile(os.path.abspath(full_result_path)) # Yes this is intentional, want it in the experiment directory, but with the same name as the folder.
        except ImportError, ex:
            logging.info("Unable to generate graph for %s data set. Graphing module unavailable in system: %s" %(data_set_name, ex)) 

    if do_create_summary:
        logging.info("Beginning summary csv generation")
        with open(os.path.join(experiment_directory, "summary.csv"), 'wb') as summary_stream:
            writer = csv.writer(summary_stream)
            # Get all the union of all the variations names. 
            variations = list(set(itertools.chain(*[r.keys() for r in summary_results.values()])))
            
            #None at start to leave column for data set names
            writer.writerow([None] + variations)
            for (data_name, data_results) in summary_results.items():
                row = [data_name] + [data_results.get(var_name, None) for var_name in variations]
                writer.writerow(row)
            
        logging.info("Ending summary csv generation")
                
    

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
    experiment = __import__(sys.argv[1]).experiment
    named_data_sets = __import__(sys.argv[2]).named_data_sets
    experiment_directory = os.path.expanduser(sys.argv[3])
    do_create_graph = sys.argv[4] != "0"
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)
    main(experiment, named_data_sets, experiment_directory, do_create_graph)
#    cProfile.run("main(experiment, named_data_sets, experiment_directory)", "mainProfile")
