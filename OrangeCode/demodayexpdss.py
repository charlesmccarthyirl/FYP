'''
Created on Mar 13, 2012

@author: charles
'''
from experiment1 import experiment as expbase
from BreadAndButter import get_sub_experiment
from DataSets import named_data_sets as basedss

experiment = get_sub_experiment(expbase, 
                                (
                                    "Random Selection",
                                    "Maximum Diversity Sampling",
                                    "CompStrat 2 - New Case - Coverage"
                                 ))

basedssdict = dict(basedss)
named_data_sets = [(dsn, basedssdict[dsn])
                   for dsn in ('zoo', 'iris')]