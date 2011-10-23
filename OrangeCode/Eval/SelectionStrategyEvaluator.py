'''
Created on 23 Oct 2011

@author: Charles McCarthy
'''

import orange, orngStat, orngTest
from SelectionStrategy import SelectionStrategy

class Result:
    def __init__(self, case_base_size, classification_accuracy, area_under_roc_curve):
        self.case_base_size = case_base_size
        self.classification_accuracy = classification_accuracy
        self.area_under_roc_curve = area_under_roc_curve

class ResultSet(list):
    def AULC(self):
        '''
        Calculates the area under the learning curve, based on simple (rectangle + top triangle)
        area for each couple of Results.
        
        >>> r = ResultSet()
        >>> r.append(Result(4, 5, 0))
        >>> r.append(Result(0, 1, 0))
        >>> r.append(Result(1, 1, 0))
        >>> r.AULC()
        10.0
        '''
        # Should be sorted already - but just in case . . .
        orderedResults = sorted(self, key=lambda x: x.case_base_size)
        previous_result = Result(0, 0, 0)
        total_area = 0
        for result in orderedResults:
            assert isinstance(result, Result)
            width = result.case_base_size - previous_result.case_base_size
            triangle_height = abs(result.classification_accuracy - previous_result.classification_accuracy)
            rectangle_height = min((result.classification_accuracy, previous_result.classification_accuracy))
            
            rectangle_area = width*rectangle_height
            triangle_area = 0.5*width*triangle_height
            
            total_area += rectangle_area + triangle_area
            previous_result = result
            
        return total_area
        

class StoppingCondition:
    def is_criteria_met(self, case_base, unlabelled_set):
        pass

class BudgetBasedStoppingCriteria(StoppingCondition):
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

class Oracle:
    def __init__(self, classifyLambda):
        self.classify = classifyLambda
    
    def classify(self, instance):
        pass

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
        results = ResultSet()
        
        # Order of assignment here important so that **locals has the right info (e.g. the stopping_criteria may care about the oracle)
        case_base = orange.ExampleTable(unlabelled_set.domain)
        selection_strategy = self.selection_strategy_generator(**locals())
        oracle = self.oracle_generator(**locals())
        stopping_condition = self.stopping_condition_generator(**locals())
        
        assert isinstance(selection_strategy, SelectionStrategy)
        assert isinstance(oracle, Oracle)
        assert isinstance(stopping_condition, StoppingCondition)
        
        while not stopping_condition.is_criteria_met(case_base, unlabelled_set):
            selection = selection_strategy.select(unlabelled_set)
            
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