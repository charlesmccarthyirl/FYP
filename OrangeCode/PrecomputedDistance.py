from itertools import islice, imap, chain, izip
import logging
import csv
import re

class Instance(object):
    
    @staticmethod
    def multiple_from(selections):
        if isinstance(selections, str):
            ms = Instance.repr_re.finditer(selections)
            return [Instance(int(m.groups()[0]), m.groups()[1]) for m in ms]
        elif isinstance(selections, Instance):
            return [selections]
        else:
            return selections
    
    @staticmethod
    def single_from(thing):
        i = Instance.multiple_from(thing)[0]
        assert isinstance(Instance, i)
        return i
         
    
    repr_re = re.compile(r"(\d+) \((.+)\)")
    
    def __init__(self, id_no=None, label=None, payload=None, instance_to_copy=None):
        self._label = None 
        self._payload = None
        self._id_no = None
        
        if instance_to_copy:
            self._label = instance_to_copy.label
            self._payload = instance_to_copy.payload
            self._id_no = instance_to_copy.id_no
        
        if self._label is None:
            self._label = label 
        if self._payload is None:
            self._payload = payload
        if self._id_no is None:
            self._id_no = id_no
    
    def __eq__(self, other):
        assert(isinstance(other, Instance))
        return self.id_no == other.id_no
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __lt__(self, other):
        return self.id_no < other.id_no
    
    def __hash__(self, *args, **kwargs):
        return hash(self.id_no)
    
    def __repr__(self, *args, **kwargs):
        return "%s (%s)" % (str(self.id_no), self.label)
    
    def __str__(self, *args, **kwargs):
        return repr(self)
        
    @property
    def label(self):
        return self._label
    
    @property
    def payload(self):
        return self._payload
    
    @property
    def id_no(self):
        return self._id_no

class DataInfo:    
    def __init__(self, data, oracle, distance_constructor, possible_classes=None):
        self.data = data
        self.oracle = oracle
        self.distance_constructor = distance_constructor
        self._is_precached = False
        if possible_classes is None:
            possible_classes = list(set((self.oracle(d) for d in self.data)))
        self.possible_classes = possible_classes
    
    def get_instance_based(self):
        data = [Instance(index, self.oracle(ex), ex) for (index, ex) in enumerate(self.data)]
        true_oracle = lambda ex: ex.label
        def distance_constructor(data):
            distance_measurer = self.distance_constructor((ex.payload for ex in data))
            return lambda ex1, ex2: distance_measurer(ex1.payload, ex2.payload)
        return DataInfo(data, true_oracle, distance_constructor, possible_classes=self.possible_classes)
    
    def get_precached(self):
        if self._is_precached:
            return self
        
        precomputed_distances = PrecomputedDistances(self.data, self.distance_constructor(self.data))
        
        di = DataInfo(self.data, self.oracle, lambda data: precomputed_distances.get_distance, 
                      possible_classes=self.possible_classes)
        di._is_precached = True
        return di   
    
    def serialize(self, stream, str_repr_getter = None):
        str_repr_getter = str_repr_getter or str
        precomputed_distances = PrecomputedDistances(self.data, self.distance_constructor(self.data))
        dist_matrix = precomputed_distances.dist_matrix
        
        writer = csv.writer(stream)
        writer.writerow(("Instance", "Classification", "Computed Distances"))
        
        data_with_classes = ((str_repr_getter(d), self.oracle(d)) for d in self.data) 
        data_with_classes_to_distances = izip(data_with_classes, dist_matrix)
        rows = imap(list, imap(lambda its: chain(*its), data_with_classes_to_distances))
        writer.writerows(rows)
    
    @staticmethod
    def get_numeric_str_repr_getter():
        id_no = [0]
        def str_repr_getter(ex):
            id_to_return = id_no[0]
            id_no[0] += 1
            return id_to_return
        
        return str_repr_getter
    
    @staticmethod
    def deserialize(stream):
        logging.info("Beginning deserializing Data Info from %s" % stream)
        reader = csv.reader(stream)
        rows = islice(reader, 1, None)
        row_splits = [(r[0], r[1], map(float, r[2:])) for r in rows]
        
        data = [Instance(id_no=en[0], label=en[1][1], payload=en[1][0]) for en in enumerate(row_splits)]
        dist_matrix = [r[2] for r in row_splits]
        instance_to_id_lookup = dict(((b, a) for (a, b) in enumerate(data)))
        
        precomputed_distances = PrecomputedDistances([], None)
        precomputed_distances.data = list(data) # Copy, in case people go a-screwing
        precomputed_distances.dist_matrix = dist_matrix
        precomputed_distances.instance_to_id_lookup = instance_to_id_lookup
        
        di = DataInfo(data, lambda e: e.label, lambda data: precomputed_distances.get_distance)
        di._is_precached = True
        
        logging.info("Finishing deserializing Data Info from %s" % stream)
        return di

class PrecomputedDistances:
    def __init__(self, data, distance_measurer):
        logging.info("Starting pre-computing distance matrix for %s" % data[:3])
        
        # In case it's shuffled, or whatever. Want the original data to be able to 
        # lookup based on id.
        self.data = list(data)
        
        self.instance_to_id_lookup = dict(((b, a) for (a, b) in enumerate(self.data)))
        
        dist_matrix = []
        
        for row_num in xrange(len(data)):
            row_comp_els = islice(data, 0, row_num+1)
            row = [distance_measurer(data[row_num], other) for other in row_comp_els]
            dist_matrix.append(row)

        logging.info("Finishing pre-computing distance matrix for %s" % data[:3])
        
        self.dist_matrix = dist_matrix
    
    def get_distance(self, ex1, ex2):
        id1 = self.instance_to_id_lookup[ex1]
        id2 = self.instance_to_id_lookup[ex2]
        
        if id1 > id2:
            id2, id1 = id1, id2
            
        return self.dist_matrix[id2][id1]

def generate_precomputed_example_distance_constructor(data, distance_measurer):
    precomputed_distances = PrecomputedDistances(data, distance_measurer)
    def inner(data):
        return precomputed_distances.get_distance
    return inner