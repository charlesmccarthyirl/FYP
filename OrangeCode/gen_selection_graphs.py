#! /usr/bin/python
'''
Created on Feb 5, 2012

@author: charles
'''
import optparse
import os
from SelectionStrategyEvaluator import ExperimentResult
from functools import partial
from Datasets.non_textual import named_data_sets as non_textual_data_sets
from Datasets.textual import named_data_sets as textual_data_sets
from itertools import chain
from glob import glob
from utils import my_import

if __name__ == '__main__':
    parser = optparse.OptionParser("usage: %prog [options] data_info_name input_results_file output_file")
    parser.add_option('--cv', help='Cross Validation to include', dest='cv',
                      default=0, action='store', type='int')
    parser.add_option('--generateall', help='boolean option', dest='generate_all',
                      default=False, action='store_true')
    parser.add_option('--nocolour', help='boolean option forces greyscale graphs', dest='colour',
                      default=True, action='store_false')
    parser.add_option('--experiment', help='Use the experiment to figure out what files to get', dest='experiment',
                      default=None, action='store')
    (options, args) = parser.parse_args()
    
    data_info_name, input_results_file, output_file = args
    cv_no = options.cv
    input_results_file = os.path.expanduser(input_results_file)
    output_file = os.path.expanduser(output_file)
    
    data_sets_dict = dict(chain(non_textual_data_sets, textual_data_sets))
    
    data_info = data_sets_dict[data_info_name]()
    
    if options.experiment:
        experiment_obj = my_import(options.experiment).experiment
        var_names = [name for (name, variation) 
                     in experiment_obj.named_experiment_variations]
        variation_files = [os.path.join(input_results_file, var_name+".tar.gz")
                           for var_name in var_names]
    elif os.path.isdir(input_results_file):
        variation_files = glob(os.path.join(input_results_file, ".tar.gz"))
    else:
        variation_files = [input_results_file]
    
    get_var_name = lambda f: os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
    vn_to_fsg_list = [(get_var_name(f), partial(open, f, 'r')) 
                      for f in variation_files]
    
    exp_result = ExperimentResult()
    exp_result.load_from_csvs(vn_to_fsg_list)
    
    def my_stream_getter(var_name, cv):
        # There's only going to be one variation, so only concerned with cv
        if cv == cv_no:
            if output_file.endswith('.pdf'):
                f = output_file
            else:
                if not os.path.exists(output_file):
                    os.makedirs(output_file)
                f = os.path.join(output_file, var_name + '.pdf')
            return open(f, 'wb')
        else:
            return None
    
    exp_result.write_to_selection_graphs(my_stream_getter, 
                                         data_info, 
                                         options.generate_all, 
                                         colour=options.colour)