'''
Created on Jan 28, 2012

@author: charles
'''
import csv, sys, os
from CaseProfilingTests import testIncrementalRcdl
from Datasets.textual import named_data_sets as textual_data_sets
from Datasets.non_textual import named_data_sets as non_textual_data_sets
from itertools import chain, islice
import pyx
import optparse

if __name__ == '__main__':
    parser = optparse.OptionParser("usage: %prog data_set_name output_file")
    (options, args) = parser.parse_args()
    
    ds_name, output_filename = args
    
    textual_dict = dict(chain(textual_data_sets, non_textual_data_sets))
    
    data_info_loader = textual_dict[ds_name]
    
    timings = testIncrementalRcdl(data_info_loader, do_cumulative_incremental=True)
    
    with open(output_filename, 'wb') as f:
        w = csv.writer(f)
        w.writerow(('Case Base Size', 'Incremental', 'Brute Force', 'Cumulative Incremental'))
        w.writerows(timings)