'''
Created on Feb 5, 2012

@author: charles
'''
from __future__ import division
import optparse
import csv
from PrecomputedDistance import DataInfo
from itertools import groupby
from utils import count_iterable
from operator import itemgetter

if __name__ == '__main__':
    parser = optparse.OptionParser("usage: %prog [options] datasets_module output_file")
    parser.add_option('-c', '--cite', help='boolean option', dest='do_cite',
                      default=False, action='store_true')
    (options, args) = parser.parse_args()
    named_data_sets_module, output_filename = args
    named_data_sets = __import__(named_data_sets_module).named_data_sets
    
    with open(output_filename, 'wb') as summary_stream:
        writer = csv.writer(summary_stream)
        writer.writerow(('Dataset', '# Instances', '# Labels', 'Label Distribution'))
        for (data_set_name, data_set_generator) in named_data_sets:
            data_info = data_set_generator()
            assert(isinstance(data_info, DataInfo))
            
            labels = data_info.possible_classes
            
            data = sorted(data_info.data, key=data_info.oracle)
            groups = groupby(data, data_info.oracle)
            
            label_to_decimals = [(label, count_iterable(group) / len(data_info.data)) 
                                 for (label, group) in groups]
            
            label_to_decimals.sort(key=itemgetter(1), reverse=True)
            
            if options.do_cite:
                data_set_name = "%s\\citep{data:%s}" % (data_set_name, data_set_name)
                
            writer.writerow((data_set_name, 
                             len(data_info.data), 
                             len(labels), 
                             ' / '.join(('%.02f' % dec for (l, dec) in label_to_decimals))))
            
            
            