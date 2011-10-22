import orange, orngStat, orngTest
import random
import matplotlib.pyplot as plt
import orange_extensions

class Oracle:
    def __init__(self, classifyLambda):
        self.classify = classifyLambda

class Selection:
    def __init__(self, selection, index):
        self.selection = selection
        self.index = index
        
    def delete_from(self, example_table):
        index = self.index
        if index is None:
            index = orange_extensions.index(self.selection)
        
        del(example_table[index])

class StoppingCriteria:
    def is_criteria_met(self, case_base, unlabelled_set):
        pass

class BudgetBasedStoppingCriteria(StoppingCriteria):
    def __init__(self, budget, initial_case_base_size=0):
        '''
        
        @param budget: The number of cases that can be labelled before the criteria is met.
        @param initial_case_base_size: The size of the initial case base (so that a calculation 
            can be done to see if it's expended
        '''
        self._budget = budget
        self._initial_case_base_size = initial_case_base_size
    
    def is_criteria_met(self, case_base, unlabelled_set):
        return len(case_base) - self._initial_case_base_size == self._budget

class SelectionStrategy:
    def __init__(self, *args, **kwargs):
        pass
    
    def select(self, collection):
        '''
        Given a collection of items to select from, selects an item and returns it (along with its index).
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
        Given a collection of items to select from, selects an item and returns it (along with its index).
        @param collection: The indexable collection to select from.
        '''
        upTo = len(collection) - 1
        r = self._random.randint(0, upTo)
        return Selection(collection[r], r)

def main():
    data = orange.ExampleTable(r"iris.arff")
    rndind = orange.MakeRandomIndices2(
                data, 
                p0=0.5)
    
    unlabelled = data.select(rndind, 0)
    test = data.select(rndind, 1)
    
    case_base = orange.ExampleTable(unlabelled.domain)

    oracle = Oracle(orange.Example.get_class)
    stopping_condition = BudgetBasedStoppingCriteria(10)
    selection_strategy = RandomSelectionStrategy()
    
    points = [] # (x,y) elements

    while not stopping_condition.is_criteria_met(case_base, unlabelled):
        selection = selection_strategy.select(unlabelled)
        
        selectedExample = orange.Example(selection.selection)
        selection.delete_from(unlabelled)
        selectedExample.set_class(oracle.classify(selectedExample))
        
        case_base.append(selectedExample)
        
        classifier = orange.kNNLearner(case_base, k=1)
        
        testResults = orngTest.testOnData([classifier], test)
        CAs = orngStat.CA(testResults)
        
        number_in_case_base = len(case_base)
        classification_accuracy = CAs[0]
        
        points.append((number_in_case_base, classification_accuracy))
        print("Accuracy for %3d = %f" % (number_in_case_base, classification_accuracy))
    
    xs = [number_in_case_base for (number_in_case_base, classification_accuracy) in points]
    ys = [classification_accuracy for (number_in_case_base, classification_accuracy) in points]
    plt.plot(xs, ys)
    plt.savefig("test.png")
    
if __name__ == "__main__":
    main()

