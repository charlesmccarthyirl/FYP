'''
Created on 23 Oct 2011

@author: Charles Mc
'''

import orange, orngStat, orngTest

class Result:
    def __init__(self, case_base_size, classification_accuracy, area_under_roc_curve):
        self.case_base_size = case_base_size
        self.classification_accuracy = classification_accuracy
        self.area_under_roc_curve = area_under_roc_curve

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
            
            result = Result(case_base_size, classification_accuracy, area_under_roc_curve)
            
            results.append(result)
            
        return results