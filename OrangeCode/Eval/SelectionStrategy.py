'''
Created on 23 Oct 2011

@author: Charles McCarthy
'''

import random
import orange
import logging

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
        
    def delete_from(self, example_table):
        index_of = self.index
        if index_of is None:
            index_of = index_of(self.selection)
        
        del(example_table[index_of])

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
    def __init__(self, *args, **kwargs):
        SelectionStrategy.__init__(self, *args, **kwargs)
        
        # Other stuff might have screwed with the state of random.random - so I'm taking my own.
        self._random = random.Random()
        self._random.seed()
        
    
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
    def __init__(self, classifier_generator, *args, **kwargs):
        self._classifier = classifier_generator(kwargs["case_base"], *args, **kwargs) 

    def measure(self, example):
        try:
            probabilities = self._classifier(example, orange.GetProbabilities)
            return max(probabilities)
        except:
            return 0; # Slight hack - if the classifier for some reason can't give me its best probability, technically its best is 0. 
                      # Needed for if 0 training examples.

class SingleCompetenceSelectionStrategy(SelectionStrategy):
    def __init__(self, competence_measure_generator, do_take_measure1_over_measure2, *args, **kwargs):
        SelectionStrategy.__init__(self, *args, **kwargs)
        
        self._competence_measure_generator = lambda: competence_measure_generator(*args, **kwargs)
        self._do_take_measure1_over_measure2 = do_take_measure1_over_measure2
        
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
        
        return [Selection(collection[selected_i], selected_i)]