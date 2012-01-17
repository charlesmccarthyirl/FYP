'''
Created on 23 Oct 2011

@author: Charles McCarthy
'''

import random
from utils import average, meanstdv
from itertools import permutations, imap
from CaseProfiling import RcdlCaseProfile, AddRemovalStore

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

class CompetenceMeasure:
    def measure(self, example):
        pass

class ClassifierBasedCompetenceMeasure(CompetenceMeasure):
    def __init__(self, probability_generator, case_base, *args, **kwargs): 
        self._probability_getter = probability_generator(case_base, *args, **kwargs)

    def measure(self, example):
        probabilities = (c_p[1] for c_p in self._probability_getter(example))
        return max(probabilities)

class ClassifierBasedMarginSamplingMeasure(CompetenceMeasure):
    def __init__(self, probability_generator, case_base, *args, **kwargs):
        self._probability_getter = probability_generator(case_base, *args, **kwargs) 

    def measure(self, example):
        probabilities = [c_p[1] for c_p in self._probability_getter(example)]
        top_2 = sorted(probabilities, reverse=True)[:2]
        return abs(top_2[0] - top_2[1])

class DiversityMeasure(CompetenceMeasure):
    def __init__(self, case_base, distance_constructor, *args, **kwargs):
        self._case_base = case_base
        self._distance_constructor = distance_constructor

    def measure(self, example):
        if len(self._case_base) == 0:
            return 1
        
        dm = self._distance_constructor(self._case_base)
        
        diversity =  min((dm(example, e) for e in self._case_base if e is not example))
        return diversity
    
class DensityMeasure(CompetenceMeasure):
    def __init__(self, data, distance_constructor, *args, **kwargs):
        dm = distance_constructor(data)
        get_density = lambda ex: average((dm(ex, o) for o in data))
        self.density_lookup = dict(((ex, get_density(ex)) for ex in data))

    def measure(self, example):
        return self.density_lookup[example]
    
class DensityTimesDiversityMeasure(CompetenceMeasure):
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
        assert isinstance(m, CompetenceMeasure)
        
        for example in collection:
            example_measure = m.measure(example)
            if selected_i_measure is None or self._do_take_measure1_over_measure2(example_measure, selected_i_measure):
                selected_i = i
                selected_i_measure = example_measure
            i += 1
        
        selected_example = collection[selected_i]
        if (self.on_selection_action):
            self.on_selection_action(selected_example)
        
        return [Selection(selected_example, selected_i)]

class CaseProfileBasedCompetenceMeasure(CompetenceMeasure):
    def __init__(self, probability_generator, case_base, case_profile_builder=None, possible_classes=None, oracle=None, *args, **kwargs):
        self.case_profile_builder = case_profile_builder
        self._probability_getter = probability_generator(case_base, *args, possible_classes=possible_classes, oracle=oracle, **kwargs)
        self.case_base = case_base
        self.possible_classes = possible_classes
        self.oracle = oracle

    def measure(self, example):
        pass

class ExampleCoverageOnlyCompetenceMeasure(CaseProfileBasedCompetenceMeasure):
    def measure(self, example):
        probabilities = self._probability_getter(example)
        example_possibilities = [(_class,
                                  probability, 
                                  len(self.case_profile_builder.suppose(example, _class).get(example, AddRemovalStore()).added.coverage_set)) 
                                 for (_class, probability) in probabilities]
        mean, std = meanstdv([cov_len for (c, p, cov_len) in example_possibilities])
        
        return std
            
        