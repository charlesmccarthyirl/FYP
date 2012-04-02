'''
Created on Mar 13, 2012

@author: charles
'''
from Experiments.all import experiment as expbase
from Datasets.non_textual import named_data_sets as basedss

experiment = expbase.create_sub_experiment((
                                    "Random Selection",
                                    "Maximum Diversity Sampling",
                                    "CompStrat 2 - New Case - Coverage"
                                 ))

basedssdict = dict(basedss)
named_data_sets = [(dsn, basedssdict[dsn])
                   for dsn in ('zoo', 'iris')]