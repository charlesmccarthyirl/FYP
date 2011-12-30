from __future__ import division
import itertools
import functools
from utils import max_multiple

def standard_inverse_distance_weighting(distance):
    return 1/(distance+1)

def squared_inverse_distance_weighting(distance):
    return 1/(pow(distance, 2) + 1)

class KNN:    
    def __init__(self, data, k, dist_meas, oracle, possible_classes, 
                 distance_weighter=squared_inverse_distance_weighting, 
                 instance_tie_breaker=min,
                 classification_tie_breaker=min):
        self.data = data
        self.k = k
        self.dist_meas = dist_meas
        self.oracle = oracle
        self.distance_weighter = distance_weighter
        self.instance_tie_breaker = instance_tie_breaker
        self.classification_tie_breaker = classification_tie_breaker
        self.possible_classes = possible_classes
    
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
        output = {}
        for e in the_list: 
            if output.has_key(e[0]):
                output[e[0]] += e[1]
            else:
                output[e[0]] = e[1]
        return output.items()
    
    @staticmethod
    def _comparer(inst_dist1, inst_dist2):
        inst1, dist1 = inst_dist1
        inst2, dist2 = inst_dist2
        
        return dist1 - dist2
    
    @staticmethod
    def s_find_nearest(instance, data, k, dist_meas, instance_tie_breaker=min, exclude_self=True):
        cmp_key = functools.cmp_to_key(KNN._comparer)
        
        current_nearest_sorted = []
        
        for other in data:
            if other is instance and exclude_self:
                continue
            other_dist = dist_meas(instance, other)
            
            if len(current_nearest_sorted) < k \
               or other_dist <= current_nearest_sorted[-1][1]:
                current_nearest_sorted.append((other, dist_meas(instance, other)))
                current_nearest_sorted.sort(key=cmp_key)
                
                if len(current_nearest_sorted) <= k:
                    continue
                
                maxes = max_multiple(current_nearest_sorted, key=cmp_key)
                
                current_nearest_sorted = current_nearest_sorted[:-(len(maxes))]
                while len(current_nearest_sorted) < k:
                    max_instances = [m[0] for m in maxes]
                    best = instance_tie_breaker(max_instances)
                    best_index = max_instances.index(best)
                    current_nearest_sorted.append(maxes[best_index])
                    del(maxes[best_index])

        return [nnd[0] for nnd in current_nearest_sorted]
    
    def find_nearest(self, instance, exclude_self=True):
        return KNN.s_find_nearest(instance, self.data, self.k, self.dist_meas, 
                                  self.instance_tie_breaker, exclude_self)
        
        
    
    def get_probabilities(self, instance, neighbours=None, exclude_self=True):
        if neighbours is None:
            neighbours = self.find_nearest(instance, exclude_self)
        classes = map(self.oracle, neighbours)
        weights = [self.distance_weighter(self.dist_meas(instance, n)) for n in neighbours]
        class_weights = KNN.sum_instances(itertools.izip(classes, weights))
        weights_sum = sum(weights)
        if weights_sum:
            class_probs = [(_class, s/weights_sum) for (_class, s) in class_weights]
            non_mentioned = [c for c in self.possible_classes if c not in (cp[0] for cp in class_probs)]
            class_probs.extend(((c, 0) for c in non_mentioned))
        else:
            equal_prob = 1 / len(self.possible_classes)
            class_probs = [(c, equal_prob) for c in self.possible_classes]
        return class_probs
    
    def classify(self, instance):
        return self.classify_from_neighbours(instance, self.find_nearest(instance, exclude_self=True))
    
    def classify_from_neighbours(self, instance, neighbours):
        class_probabilities = self.get_probabilities(instance, neighbours)
        max_class_probabilities = max_multiple(class_probabilities, key=lambda cp: cp[1]) or self.possible_classes
        return self.classification_tie_breaker((m[0] for m in max_class_probabilities))
        
