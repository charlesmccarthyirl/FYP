import orange, orngStat, orngTest
from Eval import *
import logging
import cProfile
import itertools
import os, sys, glob
from os.path import basename, splitext
from functools import partial

def csv_filename_getter(variation_name, path):
    return os.path.join(path, variation_name + '.csv')

def stream_getter(filename):
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    return open(filename, 'wb')

def main(experiment, named_data_sets, experiment_directory):
    # data_set_name -> example_table (pre-shuffled)
    for (data_set_name, data_set_generator) in named_data_sets:
        logging.info("Beginning processing on %s" % data_set_name)
        data_set = data_set_generator()
        
        csv_path = os.path.join(experiment_directory, data_set_name)
        
        files = glob.glob(os.path.join(csv_path, "*.csv"))
        name_to_file_stream_getter_pairs = [(splitext(basename(f))[0], partial(open, f, "rb")) for f in files]
        
        existing_results = ExperimentResult()
        existing_results.load_from_csvs(name_to_file_stream_getter_pairs)
        
        results = experiment.execute_on(data_set, existing_results)
        g = results.generate_graph(data_set_name)

        g.writePDFfile(os.path.abspath(csv_path)) # Yes this is intentional, want it in the experiment directory, but with the same name as the folder.
    
        results.write_to_csvs(lambda variation_name: 
                              stream_getter(csv_filename_getter(variation_name, csv_path)))
    
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
#    cProfile.run("main()", "mainProfile")
    experiment = __import__(sys.argv[1]).experiment
    named_data_sets = __import__(sys.argv[2]).named_data_sets
    experiment_directory = sys.argv[3]
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)
    main(experiment, named_data_sets, experiment_directory)
