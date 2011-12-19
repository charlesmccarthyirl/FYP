from __future__ import division
import itertools
import functools

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

def squared_inverse_distance_weighting(distance):
    return pow(standard_inverse_distance_weighting(distance), 2)

def existing_takes_precedence_tie_breaker(ties):
    return iter(ties).next()

class KNN:    
    def __init__(self, data, k, dist_meas, oracle, possible_classes, 
                 distance_weighter=squared_inverse_distance_weighting, 
                 instance_tie_breaker=existing_takes_precedence_tie_breaker,
                 classification_tie_breaker=min):
        self.data = data
        self.k = k
        self.dist_meas = dist_meas
        self.oracle = oracle
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
        output = {}
        for e in the_list: 
            if output.has_key(e[0]):
                output[e[0]] += e[1]
            else:
                output[e[0]] = e[1]
        return output.items()
    
    def find_nearest(self, instance, exclude_self=True):
        current_nearest_sorted = []
        cmp_key = self.__key_generator()
        
        for other in self.data:
            if other is instance and exclude_self:
                continue
            other_dist = self.dist_meas(instance, other)
            
            if len(current_nearest_sorted) < self.k \
               or other_dist <= current_nearest_sorted[-1][1]:
                current_nearest_sorted.append((other, self.dist_meas(instance, other)))
                current_nearest_sorted.sort(key=cmp_key)
                
                if len(current_nearest_sorted) <= self.k:
                    continue
                
                maxes = max_multiple(current_nearest_sorted, key=cmp_key)
                
                current_nearest_sorted = current_nearest_sorted[:-(len(maxes))]
                while len(current_nearest_sorted) < self.k:
                    max_instances = [m[0] for m in maxes]
                    best = self.instance_tie_breaker(max_instances)
                    best_index = max_instances.index(best)
                    current_nearest_sorted.append(maxes[best_index])
                    del(maxes[best_index])

        return [nnd[0] for nnd in current_nearest_sorted]
    
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
        