import orange, orngStat, orngTest
import random
import matplotlib.pyplot as plt
import orange_extensions

# TODO: Put in assert isinstance around the place for pydev.

class Result:
    def __init__(self, case_base_size, classification_accuracy, area_under_roc_curve):
        self.case_base_size = case_base_size
        self.classification_accuracy = classification_accuracy
        self.area_under_roc_curve = area_under_roc_curve

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

class SelectionStrategyEvaluator:
    def __init__(self, 
                 oracle_generator, 
                 stopping_condition_generator, 
                 selection_strategy_generator,
                 classifier_generator,
                 **kwargs):
        self.oracle_generator = oracle_generator
        self.stopping_condition_generator = stopping_condition_generator
        self.selection_strategy_generator = selection_strategy_generator
        self.classifier_generator = classifier_generator
    
    def generate_results(self, test_set, unlabelled_set):
        results = [] 
        
        # Order of assignment here important so that **locals has the right info (e.g. the stopping_criteria may care about the oracle)
        case_base = orange.ExampleTable(unlabelled_set.domain)
        selection_strategy = self.selection_strategy_generator(**locals())
        oracle = self.oracle_generator(**locals())
        stopping_condition = self.stopping_condition_generator(**locals())
        
        while not stopping_condition.is_criteria_met(case_base, unlabelled_set):
            selection = selection_strategy.select(unlabelled_set) #TODO: Change to being a multi-selection thing
            
            selectedExample = orange.Example(selection.selection)
            selection.delete_from(unlabelled_set)
            selectedExample.set_class(oracle.classify(selectedExample))
            
            case_base.append(selectedExample)
            
            classifier = self.classifier_generator(case_base)
            
            testResults = orngTest.testOnData([classifier], test_set)
            
            case_base_size = len(case_base)
            classification_accuracy = orngStat.CA(testResults)[0]
            area_under_roc_curve = orngStat.AUC(testResults)[0]
            
            results.append(Result(case_base_size, classification_accuracy, area_under_roc_curve))
            
        return results

def main():
    oracle_generator = lambda *args, **kwargs: Oracle(orange.Example.get_class)
    stopping_condition_generator = lambda *args, **kwargs: BudgetBasedStoppingCriteria(10)
    selection_strategy_generator = lambda *args, **kwargs: RandomSelectionStrategy()
    classifier_generator = lambda training_data, *args, **kwargs: orange.kNNLearner(training_data, k=5)
    
    data = orange.ExampleTable(r"iris.arff")
    rndind = orange.MakeRandomIndices2(data, p0=.8)
    
    test = data.select(rndind, 0)
    unlabelled = data.select(rndind, 0, negate=1)
    
    evaluator = SelectionStrategyEvaluator(**locals())
    points = evaluator.generate_results(test, unlabelled)
    #print("Accuracy for %3d = %f" % (case_base_size, classification_accuracy))
    
    xs = [result.case_base_size for result in points]
    ys = [result.classification_accuracy for result in points]
    plt.plot(xs, ys)
    plt.savefig("test.png")
    
if __name__ == "__main__":
    main()

