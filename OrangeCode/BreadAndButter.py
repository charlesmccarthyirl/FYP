from __future__ import division
import orange, orngStat, orngTest
from Eval import *
import glob, os
from functools import partial
import collections
import logging
import itertools
import random
import functools
from collections import defaultdict

RANDOM_SEED = 42
DATASETS_DIR = "../Datasets/"
DATASET_EXTENSIONS = [".csv", ".tab", ".arff"]

true_oracle = lambda ex: ex.get_class().value
oracle_generator = lambda *args, **kwargs: Oracle(true_oracle)

k=5

def classifier_generator(training_data, distance_constructor, *args, **kwargs):
    if len(training_data) == 0:
        return None
    #logging.debug("Beginning get_classifier") 
    #logging.debug("on %s" % list(training_data))
    classifier = orange.kNNLearner(training_data, k=k, rankWeight=False, distanceConstructor=distance_constructor())
    #logging.debug("Ending get_classifier") 
    return classifier

def get_training_test_sets_extractor(rand_seed=None):
    def split_data(data):   
        splits = n_fold_cross_validation(data, 10, true_oracle, rand_seed=rand_seed)
        return splits
    return split_data

def create_named_experiment_variations_generator(named_selection_strategy_generators):
    return lambda *args, **kwargs: dict([(name, 
                                          ExperimentVariation(classifier_generator, 
                                                              selection_strategy_generator))
                                         for (name, selection_strategy_generator) 
                                         in named_selection_strategy_generators.items()])
    
def create_experiment(stopping_condition_generator, named_experiment_variations, rand_seed=RANDOM_SEED):
    return Experiment(oracle_generator,
                      true_oracle,
                      stopping_condition_generator, 
                      get_training_test_sets_extractor(rand_seed), 
                      named_experiment_variations)
    
data_files = [os.path.join(DATASETS_DIR, filename)
              for filename in os.listdir(DATASETS_DIR) 
              if os.path.splitext(filename)[1].lower() in DATASET_EXTENSIONS]

data_files_dict = dict([(os.path.splitext(os.path.basename(df))[0], df) for df in data_files])

def euclidean_distance_constructor_generator(data):
    return orange.ExamplesDistanceConstructor_Euclidean

def load_data_distance_constructor_pair(
        base_filename, 
        random_seed=RANDOM_SEED, 
        #distance_constructor_generator=generate_example_distance_constructor_generator()):
        distance_constructor_generator=generate_precomputed_example_distance_constructor_generator()):
        #distance_constructor_generator=euclidean_distance_constructor_generator):
    d = orange.ExampleTable(data_files_dict[base_filename], randomGenerator=orange.RandomGenerator(random_seed))
    
    prev_len = len(d)
    d.remove_duplicates()
    new_len = len(d)
    
    if (prev_len != new_len):
        logging.info("Removed %d duplicates contained in %s" % (prev_len - new_len, base_filename))
    
    # Have to add these after . . .
    if not d.domain.has_meta("ex_id"):
        var = orange.FloatVariable("ex_id")
        varId = orange.newmetaid()
        d.domain.add_meta(varId, var)
        i = 0
        for ex in d:
            ex.set_meta(varId, i)
            i += 1
    
    distance_constructor = distance_constructor_generator(d)
    
    d.shuffle() # Could all be clustered together in the file. Some of my operations might 
                # (and do . . .) go in order - so can skew the results *a lot*.
    return (d, distance_constructor)


def create_named_data_set_generators(base_data_set_infos):
    base_data_set_infos = [info if isinstance(info, dict) 
                                else {"base_filename": info} 
                           for info in base_data_set_infos]
    return [(data_set_info["base_filename"], partial(load_data_distance_constructor_pair, **data_set_info)) 
            for data_set_info in base_data_set_infos]
    
class Instance(object):
    next_instance_no = 1
    
    def __init__(self, label, payload):
        self._label = label
        self._payload = payload
        self._instance_no = Instance.next_instance_no
        Instance.next_instance_no += 1
    
    def __eq__(self, other):
        assert(isinstance(other, Instance))
        return self._payload == other._payload
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self, *args, **kwargs):
        return hash(self._payload)
    
    def __lt__(self, other):
        assert(isinstance(other, Instance))  
        return self._instance_no < other._instance_no  
        
    @property
    def label(self):
        return self._label
    
    @property
    def payload(self):
        return self._payload
    
    @staticmethod
    def first_instantiated(inst1, inst2):
        assert(isinstance(inst1, Instance))
        assert(isinstance(inst2, Instance))
        return min((inst1, inst2))

def max_multiple(the_list, key=None):
    '''
    Finds the list of maximum elements in a given list.
    @param the_list: The list to find the maximum in.
    @param key: The key function to compare on.
    
    >>> m_list = [('a', 3), ('b', 2), ('c', 3)]
    >>> max_multiple(m_list, key=lambda x: x[1])
    [('a', 3), ('c', 3)]
    >>> m_list
    [('a', 3), ('b', 2), ('c', 3)]
    '''
    
    ms = []
    m_key = None
    
    for thing in the_list:
        thing_key = thing if key is None else key(thing)
        if m_key is None or thing_key > m_key:
            m_key = thing_key
            ms = [thing]
        elif thing_key == m_key:
            ms.append(thing)
    return ms

def standard_inverse_distance_weighting(distance):
    return 1/(distance+1)

def existing_takes_precedence_tie_breaker(ties):
    return iter(ties).next()

class KNN:    
    def __init__(self, data, dist_meas, true_oracle, possible_classes, 
                 distance_weighter=standard_inverse_distance_weighting, 
                 instance_tie_breaker=existing_takes_precedence_tie_breaker,
                 classification_tie_breaker=min):
        self.data = data
        self.dist_meas = dist_meas
        self.true_oracle = true_oracle
        self.distance_weighter = distance_weighter
        self.instance_tie_breaker = instance_tie_breaker
        self.classification_tie_breaker = classification_tie_breaker
        self.possible_classes = possible_classes
    
    def __key_generator(self):
        return functools.cmp_to_key(self.__comparer)
    
    def __comparer(self, inst_dist1, inst_dist2):
        inst1, dist1 = inst_dist1
        inst2, dist2 = inst_dist2
        
        return dist1 - dist2
    
    @staticmethod
    def sum_instances(the_list):
        '''
        Creates a (el -> sum) pair for the input list of (el->val) pairs
        @param the_list: List of el->val pairs.
        
        >>> m_list = [('a', 1), ('a', 2), ('b', 3), ('a', 4)]
        >>> KNN.sum_instances(m_list)
        [('a', 7), ('b', 3)]
        >>> m_list
        [('a', 1), ('a', 2), ('b', 3), ('a', 4)]
        '''
        output = defaultdict(float)
        for e in the_list: 
            if output.has_key(e[0]):
                output[e[0]] += e[1]
            else:
                output[e[0]] = e[1]
        return output.items()
    
    def find_nearest(self, instance, k, exclude_self=True):
        current_nearest_sorted = []
        cmp_key = self.__key_generator()
        
        for other in self.data:
            if other is instance and exclude_self:
                continue
            other_dist = self.dist_meas(instance, other)
            
            if len(current_nearest_sorted) < k \
               or other_dist <= current_nearest_sorted[-1][1]:
                current_nearest_sorted.append((other, self.dist_meas(instance, other)))
                current_nearest_sorted.sort(key=cmp_key)
                
                if len(current_nearest_sorted) <= k:
                    continue
                
                maxes = max_multiple(current_nearest_sorted, key=cmp_key)
                
                current_nearest_sorted = current_nearest_sorted[:-(len(maxes))]
                while len(current_nearest_sorted) < k:
                    max_instances = [m[0] for m in maxes]
                    best = self.instance_tie_breaker(max_instances)
                    best_index = max_instances.index(best)
                    current_nearest_sorted.append(maxes[best_index])
                    del(maxes[best_index])

        return [nnd[0] for nnd in current_nearest_sorted]
    
    def get_probabilities(self, instance, neighbours=None, k=None, exclude_self=True):
        if neighbours is None:
            neighbours = self.find_nearest(instance, k, exclude_self)
        classes = map(self.true_oracle, neighbours)
        weights = [self.dist_meas(instance, n) for n in neighbours]
        class_weights = KNN.sum_instances(itertools.izip(classes, weights))
        weights_sum = sum(weights) or 1 # to avoid div by 0
        return [(_class, s/weights_sum) for (_class, s) in class_weights]
    
    def classifiy(self, instance, k):
        return self.classify_from_neighbours(instance, self.find_nearest(instance, k, exclude_self=True))
    
    def classify_from_neighbours(self, instance, neighbours):
        class_probabilities = self.get_probabilities(instance, neighbours)
        max_class_probabilities = max_multiple(class_probabilities, key=lambda cp: cp[1]) or self.possible_classes
        return self.classification_tie_breaker((m[0] for m in max_class_probabilities))
        
        