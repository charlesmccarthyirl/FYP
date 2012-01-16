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
    distance_constructor = lambda data: dist_meas
    
    return DataInfo(data, true_oracle, distance_constructor).get_instance_based()

def get_data_info(filename, distance_constructor, do_normalize_distances=True):
    ext = os.path.splitext(filename)[1].lower()
    di = None
    if ext in [".csv", ".tab", ".arff"]:
        di = orange_load(filename, distance_constructor)
    else:
        open_op = open
        if ext == ".gz":
            open_op = gzip.open
            ext = os.path.splitext(os.path.splitext(filename)[0])[1].lower()
                
        if ext == ".pcdt":
            method = DataInfo.DeserializationMethod.csv
        elif ext == ".pcdb":
            method = DataInfo.DeserializationMethod.proto
        else:
            raise NotImplementedError("No implementation for ext %s" % ext)
        
        with open_op(filename) as f:
            di = DataInfo.deserialize(f, method)
            
    if do_normalize_distances:
        di = di.get_distance_normalized()
    
    return di
            
        
         
    
    
    

        