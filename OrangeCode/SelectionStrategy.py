'''
Created on 23 Oct 2011

@author: Charles McCarthy
'''

import random
import logging
import CaseProfiling
from functools import partial

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
        self._distance_measure = distance_constructor(case_base) if len(case_base) > 0 else None

    def measure(self, example):
        if not self._distance_measure:
            return 0
        
        return min((self._distance_measure(example, e) for e in self._case_base))
    
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
    def __init__(self, probability_generator, case_base, case_profile_builder=None, *args, **kwargs):
        self.case_profile_builder = case_profile_builder
        self._probability_getter = probability_generator(case_base, *args, **kwargs)
        self.case_base = case_base

    def measure(self, example):
        probabilities = self._probability_getter(example)

        example_possibilities = [(probability, 
                                  self.case_profile_builder.suppose(example, _class)) 
                                 for (_class, probability) in probabilities if probability > 0]
        
        def compute_rcdl_score(rcdl_profile):
            return len(rcdl_profile.coverage_set) - len(rcdl_profile.liability_set)
        
        def compute_add_removal_score(add_removal):
            return compute_rcdl_score(add_removal.added) - compute_rcdl_score(add_removal.removed)
        
        def compute_add_removals_dict_score(add_removals_dict):
            return sum((compute_add_removal_score(add_removal) for add_removal in add_removals_dict.values()))
        
        return sum((probability*compute_add_removals_dict_score(add_removals_dict) 
                    for (probability, add_removals_dict) in example_possibilities))
        