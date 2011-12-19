'''
Created on Dec 19, 2011

@author: charles
'''
from SelectionStrategyEvaluator import Oracle
import os
from PrecomputedDistance import DataInfo
import gzip

def orange_load(filename, distance_constructor):
    import orange
    true_oracle = Oracle(lambda ex: ex.get_class().value)
    
    if distance_constructor is None:
        distance_constructor=lambda data: orange.ExamplesDistanceConstructor_Euclidean()(list(data))
    
    data = orange.ExampleTable(filename)
    
    dist_meas = distance_constructor(data) # Want global distances always.
    dist_constructor = lambda data: dist_meas
    
    return DataInfo(data, true_oracle, distance_constructor).get_instance_based()


def get_data_info(filename, distance_constructor):
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".csv", ".tab", ".arff"]:
        return orange_load(filename, distance_constructor)
    elif ext == ".pcdcd":
        with open(filename) as f:
            return DataInfo.deserialize(f)
    elif filename.endswith(".pcdcd.gzip"):
        with gzip.open(filename, "rb") as f:
            return DataInfo.deserialize(f)
    else:
        raise NotImplementedError("No implementation for ext %s" % ext)
            
        
         
    
    
    

        