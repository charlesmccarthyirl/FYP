'''
Created on Feb 5, 2012

@author: charles
'''
import optparse
import os
from SelectionStrategyEvaluator import ExperimentResult
from functools import partial
from DataSets import named_data_sets as non_textual_data_sets
from TextualDataSets import named_data_sets as textual_data_sets
from itertools import chain

if __name__ == '__main__':
    parser = optparse.OptionParser("usage: %prog [options] data_info_name input_results_file output_file")
    parser.add_option('--cv', help='Cross Validation to include', dest='cv',
                      default=0, action='store', type='int')
    parser.add_option('--generateall', help='boolean option', dest='generate_all',
                      default=False, action='store_true')
    (options, args) = parser.parse_args()
    
    data_info_name, input_results_file, output_file = args
    cv_no = options.cv
    
    data_sets_dict = dict(chain(non_textual_data_sets, textual_data_sets))
    
    data_info = data_sets_dict[data_info_name]()
    
    variation_name = os.path.splitext(os.path.splitext(os.path.basename(input_results_file))[0])[0]
    vn_to_fsg = (variation_name, partial(open, input_results_file, 'r'))
    
    exp_result = ExperimentResult()
    exp_result.load_from_csvs([vn_to_fsg])
    
    def my_stream_getter(var_name, cv):
        # There's only going to be one variation, so only concerned with cv
        if cv == cv_no:
            return open(output_file, 'wb')
        else:
            return None
    
    exp_result.write_to_selection_graphs(my_stream_getter, data_info, options.generate_all)
    