'''
Created on Mar 29, 2012

@author: charles
'''
import all_datasets
from utils import sub_pairs

names = ['Comp', 'Vehicle', 'breathalyzer', 'haberman', 'dermatology']
named_data_sets = sub_pairs(all_datasets.named_data_sets, names)