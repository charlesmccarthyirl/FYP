import orange, orngStat, orngTest
from Eval import *
from BreadAndButter import *

named_data_sets = create_named_data_set_generators(("iris", 
                                                    "zoo", 
                                                    "alcohol-limits", 
                                                    "breast-cancer-wisconsin-cont", 
                                                    "wine", 
                                                    "adult_sample", 
                                                    "car", 
                                                    "diabetes"))