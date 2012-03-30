'''
Created on Mar 29, 2012

@author: charles
'''
import non_textual
from utils import sub_pairs

names = ['glass', 'tae', 'heartdisease', 'lungcancer']
named_data_sets = sub_pairs(non_textual.named_data_sets, names)