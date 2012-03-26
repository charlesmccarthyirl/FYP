'''
Created on 23 Oct 2011

@author: Charles McCarthy
'''
from __future__ import division
import random
from utils import average, meanstdv, lazyproperty
from itertools import permutations, imap
from CaseProfiling import RcdlCaseProfile, AddRemovalStore
from functools import partial
from math import ceil



def index_of(collection, needle):
    i = 0
    for element in collection:
        if element == needle:
            return i
        i += i
    
    return -1

class Selection:
    def __init__(self, selection, index_of):
        self.selection = selection
        self.index = index_of

class SelectionStrategy:
    def __init__(self, *args, **kwargs):
        pass
    
    def select(self, collection):
        '''
        Given a collection of items to select from, selects a list of items and returns them (along with their index_of).
        @param collection: The indexable collection to select from.
        '''
        pass

class RandomSelectionStrategy(SelectionStrategy):
    def __init__(self, random_seed=None, *args, **kwargs):
        SelectionStrategy.__init__(self, *args, **kwargs)
        
        # Other stuff might have screwed with the state of random.random - so I'm taking my own.
        self._random = random.Random()
        self._random.seed(random_seed)
        
    
    def select(self, collection):   
        '''
        Given a collection of items to select from, selects a list of items and returns them (along with their index_of).
        @param collection: The indexable collection to select from.
        '''
        upTo = len(collection) - 1
        r = self._random.randint(0, upTo)
        return [Selection(collection[r], r)]

class Measure:
    def measure(self, example):
        pass
    
    @staticmethod
    def create_measure_constructor(reducer, measure_contructors):
        def init(*args, **kwargs):
            m = Measure()
            measures = [f(*args, **kwargs) for f in measure_contructors]
            m.measure = lambda example: reducer([f.measure(example) for f in measures])
            return m
        return init

class ClassifierBasedCompetenceMeasure(Measure):
    def __init__(self, probability_generator, case_base, *args, **kwargs): 
        self._probability_getter = probability_generator(case_base, *args, **kwargs)

    def measure(self, example):
        probabilities = (c_p[1] for c_p in self._probability_getter(example))
        return max(probabilities)

class ClassifierBasedMarginSamplingMeasure(Measure):
    def __init__(self, probability_generator, case_base, *args, **kwargs):
        self._probability_getter = probability_generator(case_base, *args, **kwargs) 

    def measure(self, example):
        probabilities = [c_p[1] for c_p in self._probability_getter(example)]
        top_2 = sorted(probabilities, reverse=True)[:2]
        return abs(top_2[0] - top_2[1])

class DiversityMeasure(Measure):
    def __init__(self, case_base, distance_constructor, *args, **kwargs):
        self._case_base = case_base
        self._distance_constructor = distance_constructor

    def measure(self, example):
        if len(self._case_base) == 0:
            return 1
        
        dm = self._distance_constructor(self._case_base)
        
        diversity =  min((dm(example, e) for e in self._case_base if e is not example))
        return diversity
    
class DensityMeasure(Measure):
    def __init__(self, data, distance_constructor, stretch=True, *args, **kwargs):
        dm = distance_constructor(data)
        sm = lambda *args, **kwargs: 1 - dm(*args, **kwargs)
        get_density = lambda ex: average((sm(ex, o) for o in data))
        self.density_lookup = dict(((ex, get_density(ex)) for ex in data))
        if stretch:
            min_density = min(self.density_lookup.itervalues())
            max_density = max(self.density_lookup.itervalues())
            assert(0 <= min_density <= 1)
            assert(0 <= max_density <= 1)
            
            max_density_after = max_density - min_density
            
            scale_prod = 1.0/max_density_after
            
            for k in self.density_lookup.iterkeys():
                stretched_d = (self.density_lookup[k] - min_density) * scale_prod
                assert(0 <= stretched_d <= 1)
                self.density_lookup[k] = stretched_d

    def measure(self, example):
        return self.density_lookup[example]
    
    
class SparsityMeasure(DensityMeasure):
    def __init__(self, *args, **kwargs):
        DensityMeasure.__init__(self, *args, stretch=True, **kwargs)
        
    def measure(self, example):
        return 1 - DensityMeasure.measure(self, example)

class DensityTimesDiversityMeasure(Measure):
    def __init__(self, *args, **kwargs):
        self.density_measure = DensityMeasure(*args, **kwargs)
        self.diversity_measure = DiversityMeasure(*args, **kwargs)

    def measure(self, example):
        return self.density_measure.measure(example) * self.diversity_measure.measure(example)
    
class SingleCompetenceSelectionStrategy(SelectionStrategy):
    @staticmethod
    def take_minimum(measure1, measure2):
        return measure1 < measure2
    
    @staticmethod
    def take_maximum(measure1, measure2):
        return measure1 > measure2
    
    def __init__(self, 
                 competence_measure_generator, 
                 do_take_measure1_over_measure2, 
                 on_selection_action=None, 
                 *args, 
                 **kwargs):
        SelectionStrategy.__init__(self, *args, **kwargs)
        
        self._competence_measure_generator = lambda: competence_measure_generator(*args, **kwargs)
        self._do_take_measure1_over_measure2 = do_take_measure1_over_measure2
        self.on_selection_action = on_selection_action
        
    def select(self, collection):   
        '''
        Given a collection of items to select from, selects a list of items and returns them (along with their index_of).
        @param collection: The indexable collection to select from.
        '''
        if (len(collection) == 0):
            return None
        
        i = 0
        selected_i = 0
        selected_i_measure = None
        
        m = self._competence_measure_generator()
        assert isinstance(m, Measure)
        
        for example in collection:
            example_measure = m.measure(example)
            if selected_i_measure is None \
               or example_measure is not None \
                  and self._do_take_measure1_over_measure2(example_measure, selected_i_measure):
                selected_i = i
                selected_i_measure = example_measure
            i += 1
        
        selected_example = collection[selected_i]
        if (self.on_selection_action):
            self.on_selection_action(selected_example)
        
        return [Selection(selected_example, selected_i)]

