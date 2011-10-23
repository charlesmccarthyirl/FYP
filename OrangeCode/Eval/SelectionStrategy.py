'''
Created on 23 Oct 2011

@author: Charles Mc
'''

import random

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
        Given a collection of items to select from, selects an item and returns it (along with its index_of).
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
        Given a collection of items to select from, selects an item and returns it (along with its index_of).
        @param collection: The indexable collection to select from.
        '''
        upTo = len(collection) - 1
        r = self._random.randint(0, upTo)
        return Selection(collection[r], r)