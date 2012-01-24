from __future__ import division
from itertools import islice, imap, chain, izip, permutations
import logging
import csv
import re
import pcdcb_pb2

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
        assert isinstance(i, Instance)
        return i
         
    
    repr_re = re.compile(r"(\d+) \((.+)\)")
    
    def __init__(self, id_no=None, label=None, payload=None, instance_to_copy=None):
        self.label = None 
        self.payload = None
        self.id_no = None
        
        if instance_to_copy:
            self.label = instance_to_copy.label
            self.payload = instance_to_copy.payload
            self.id_no = instance_to_copy.id_no
        
        if self.label is None:
            self.label = label 
        if self.payload is None:
            self.payload = payload
        if self.id_no is None:
            self.id_no = id_no
    
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

class DataInfo: 
    def copy(self):
        di = DataInfo(self.data, self.oracle, self.distance_constructor, self.possible_classes)
        di._is_precached = self._is_precached
        return di
       
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
        
        dm = self.distance_constructor(self.data)
        precomputed_distances = PrecomputedDistances(self.data, dm)
        
        di = DataInfo(self.data, self.oracle, lambda data: precomputed_distances.get_distance, 
                      possible_classes=self.possible_classes)
        di._is_precached = True
        return di
    
    def get_distance_normalized(self):
        dm = self.distance_constructor(self.data)
        
        max_dist = max(dm(*p) for p in permutations(self.data, 2)) #dist should be the same both ways, but just in case
        
        if max_dist <= 1.0:
            return self
        
        di = self.copy()
        old_dc = di.distance_constructor
        
        def new_distance_constructor(data):
            old_dm = old_dc(data)
            return lambda ex1, ex2: old_dm(ex1, ex2)/max_dist
            
        di.distance_constructor = new_distance_constructor
        di._is_precached = False # It might almost be precached, but technically at the mo, it's not.
        return di
    
    class SerializationMethod:
        @staticmethod
        def csv(stream, pcd_tuples):
            writer = csv.writer(stream)
            writer.writerow(("Instance", "Classification", "Computed Distances"))
            
            rows = imap(list, imap(lambda d: chain((d[0], d[1]), d[2]), pcd_tuples))
            
            writer.writerows(rows)
        
        @staticmethod
        def proto(stream, pcd_tuples):
            results = pcdcb_pb2.PrecomputedDistanceData()
            for (payload, label, distances) in pcd_tuples:
                entry = results.entry.add()
                entry.instance.payload = str(payload)
                entry.instance.label = str(label)
                entry.distances.extend(imap(float, distances))
            
            stream.write(results.SerializeToString())

    
    def get_pcd_tuples(self, str_repr_getter):
        str_repr_getter = str_repr_getter or str
        precomputed_distances = PrecomputedDistances(self.data, self.distance_constructor(self.data))
        dist_matrix = precomputed_distances.dist_matrix
        data_with_classes = ((str_repr_getter(d), self.oracle(d)) for d in self.data) 
        data_with_classes_to_distances = izip(data_with_classes, dist_matrix)
        return ((p, c, ds) for ((p, c), ds) in data_with_classes_to_distances)
    
    def serialize(self, stream, serialization_method, str_repr_getter = None):
        return serialization_method(stream, self.get_pcd_tuples(str_repr_getter))
    
    @staticmethod
    def get_numeric_str_repr_getter():
        id_no = [0]
        def str_repr_getter(ex):
            id_to_return = id_no[0]
            id_no[0] += 1
            return id_to_return
        
        return str_repr_getter
    
    
    class DeserializationMethod:
        @staticmethod
        def csv(stream):
            
            reader = csv.reader(stream)
            rows = islice(reader, 1, None)
            
            pcd_tuples = [(r[0], r[1], map(float, r[2:])) for r in rows]
            return pcd_tuples
        
        @staticmethod
        def proto(stream):
            results = pcdcb_pb2.PrecomputedDistanceData()
            results.ParseFromString(stream.read())
            pcd_tupes = [ (e.instance.payload, e.instance.label, list(e.distances)) for e in results.entry]
            return pcd_tupes
    
    @staticmethod
    def deserialize_pcd_tuples(pcd_tuples):
        data = [Instance(id_no=en[0], label=en[1][1], payload=en[1][0]) for en in enumerate(pcd_tuples)]
        dist_matrix = [r[2] for r in pcd_tuples]
        instance_to_id_lookup = dict(((b, a) for (a, b) in enumerate(data)))
        
        precomputed_distances = PrecomputedDistances([], None)
        precomputed_distances.data = list(data) # Copy, in case people go a-screwing
        precomputed_distances.dist_matrix = dist_matrix
        precomputed_distances.instance_to_id_lookup = instance_to_id_lookup
        
        di = DataInfo(data, lambda e: e.label, lambda data: precomputed_distances.get_distance)
        di._is_precached = True
        
        return di
    
    @staticmethod
    def deserialize(stream, deserialization_method):
        logging.info("Beginning deserializing Data Info from %s" % stream)
        return DataInfo.deserialize_pcd_tuples(deserialization_method(stream))
        logging.info("Finishing deserializing Data Info from %s" % stream)


def generate_precomputed_example_distance_constructor(data, distance_measurer):
    precomputed_distances = PrecomputedDistances(data, distance_measurer)
    def inner(data):
        return precomputed_distances.get_distance
    return inner