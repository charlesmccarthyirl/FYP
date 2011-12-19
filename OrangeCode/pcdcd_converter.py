'''
Created on Dec 19, 2011

@author: charles
'''

from DataInfoLoaders import get_data_info
import sys, os
import orange
from PrecomputedDistance import DataInfo
import gzip

def clean_path(path):
    return os.path.abspath(os.path.expanduser(path))

if __name__ == '__main__':
    in_file = clean_path(sys.argv[1])
    out_file = clean_path(sys.argv[2])
    dist_constructor_str = sys.argv[3]
    dist_constructor_generator = getattr(orange, "ExamplesDistanceConstructor_%s" % dist_constructor_str)
    dist_constructor = lambda data: dist_constructor_generator()(list(data))
    data_info = get_data_info(in_file, dist_constructor)
    
    do_zip = sys.argv[4] == '1'
    open_args = (out_file, "wb")
    open_func = gzip.open if do_zip else open
    
    with open_func(*open_args) as ofs:
        data_info.serialize(ofs, DataInfo.get_numeric_str_repr_getter())