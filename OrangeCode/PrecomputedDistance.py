import itertools
import logging

class PrecomputedDistances:
    def id_from_ex(self, ex):
        return int(ex.get_meta(self.__meta_id_name))
    
    def __init__(self, data, distance_measurer):
        logging.info("Starting pre-computing distance matrix for %s" % data)
        self.instance_to_id_lookup = dict(((b, a) for (a, b) in enumerate(data)))
        
        sym_matrix = []
        
        for row_num in xrange(len(data)):
            row_comp_els = itertools.islice(data, 0, row_num+1)
            row = [distance_measurer(data[row_num], other) for other in row_comp_els]
            sym_matrix.append(row)

        logging.info("Finishing pre-computing distance matrix for %s" % data)
        
        self.__sym_matrix = sym_matrix
    
    def get_distance(self, ex1, ex2):
        id1 = self.instance_to_id_lookup[ex1]
        id2 = self.instance_to_id_lookup[ex2]
        
        if id1 > id2:
            id2, id1 = id1, id2
            
        return self.__sym_matrix[id2][id1]

def generate_precomputed_example_distance_constructor(data, distance_measurer):
    precomputed_distances = PrecomputedDistances(data, distance_measurer)
    def inner(data):
        return precomputed_distances.get_distance
    return inner