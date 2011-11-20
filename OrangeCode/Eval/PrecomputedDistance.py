import orange
import logging

class ExamplesDistance_LookupBased(orange.ExamplesDistance):
    def __init__(self, distance_lookup_func):
        self.distance_lookup_func = distance_lookup_func
    
    def __call__(self, ex1, ex2):
        return self.distance_lookup_func(ex1, ex2)

class PrecomputedDistances:
    def id_from_ex(self, ex):
        return int(ex.get_meta(self.__meta_id_name))
    
    def __init__(self, data, distance_constructor, meta_id_name="ex_id"):
        self.__meta_id_name = meta_id_name
        
        logging.info("Starting pre-computing distance matrix for %s" % data)
        
        pairs_to_compute = ((ex1, ex2) for ex1 in data 
                                       for ex2 in data 
                            if self.id_from_ex(ex1) <= self.id_from_ex(ex2))
        
        sym_matrix = orange.SymMatrix(len(data))
        
        distance_measurer = distance_constructor()(data)
        
        for (ex1, ex2) in pairs_to_compute:
            sym_matrix[self.id_from_ex(ex1), self.id_from_ex(ex2)] = distance_measurer(ex1, ex2)
        
        logging.info("Finishing pre-computing distance matrix for %s" % data)
        
        self.__sym_matrix = sym_matrix
    
    def get_distance(self, ex1, ex2):
        if isinstance(ex1, orange.Example):
            ex1 = self.id_from_ex(ex1)
            
        if isinstance(ex2, orange.Example):
            ex2= self.id_from_ex(ex2)
            
        return self.__sym_matrix[ex1, ex2]
            

def generate_example_distance_constructor_generator(
        distance_constructor=orange.ExamplesDistanceConstructor_Euclidean):
    def inner(data):
        def other_thing(*args, **kwargs):
            #logging.info(args)
            #logging.info(kwargs)
            distance_measurer = distance_constructor()(data, 0)
            return ExamplesDistance_LookupBased(distance_measurer)
        
        def other_thing_2(*args, **kwargs):
            return other_thing
        
        return other_thing_2 # Outer lambda is for the future call of distance_constructor()
                                                   # which returns a callable (passed data domain etc.)), which
                                                   # returns an ExamplesDistance instance.
    return inner

def generate_precomputed_example_distance_constructor_generator(
        distance_constructor=orange.ExamplesDistanceConstructor_Euclidean):
    def inner(data):
        precomputed_distances = PrecomputedDistances(data, distance_constructor)
        
        def other_thing(*args, **kwargs):
            return ExamplesDistance_LookupBased(precomputed_distances.get_distance)
        
        return lambda *args, **kwargs: other_thing # Outer lambda is for the future call of distance_constructor()
                                                   # which returns a callable (passed data domain etc.)), which
                                                   # returns an ExamplesDistance instance.
    return inner