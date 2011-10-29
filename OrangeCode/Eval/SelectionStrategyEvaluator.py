'''
Created on 23 Oct 2011

@author: Charles McCarthy
'''

import collections
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

def average(iterable):
    sum = 0
    length = 0
    for i in iterable:
        sum += i
        length += 1
        
    return sum / length

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
    
    
    
    def generate_results_from_many(self, data_test_iterable):
        all_results = [list(self.generate_results(test_set, unlabelled_set))
                       for (test_set, unlabelled_set) 
                       in data_test_iterable]
        
        aligned_results = zip(*all_results)
        
        if not all((all((result is not None and result.case_base_size == aligned_result[0].case_base_size
                         for result
                         in aligned_result))
                    for aligned_result
                    in aligned_results)):
            raise Exception("case_base_sizes don't seem aligned")
        
        averaged_result_instances = (Result(aligned_result[0].case_base_size, 
                                            average((result.classification_accuracy for result in aligned_result)), 
                                            average((result.area_under_roc_curve for result in aligned_result))) 
                                     for aligned_result 
                                     in aligned_results)
        
        return ResultSet(averaged_result_instances)
    
    def __generate_result(self, case_base, test_set):
        classifier = self.classifier_generator(case_base)
                
        testResults = orngTest.testOnData([classifier], test_set)
        
        case_base_size = len(case_base)
        classification_accuracy = orngStat.CA(testResults)[0]
        area_under_roc_curve = orngStat.AUC(testResults)[0]
        
        return Result(case_base_size, classification_accuracy, area_under_roc_curve)
    
    def generate_results(self, unlabelled_set, test_set):
        results = ResultSet()
        
        # hacky Laziness. Just don't want to have to do **locals() myself, but I can't pass self.
        selection_strategy_evaluator = self
        del(self)
        
        # Order of assignment here important so that **locals has the right info (e.g. the stopping_criteria may care about the oracle)
        case_base = orange.ExampleTable(unlabelled_set.domain)
        selection_strategy = selection_strategy_evaluator.selection_strategy_generator(**locals())
        oracle = selection_strategy_evaluator.oracle_generator(**locals())
        stopping_condition = selection_strategy_evaluator.stopping_condition_generator(**locals())
        
        assert isinstance(selection_strategy, SelectionStrategy)
        assert isinstance(oracle, Oracle)
        assert isinstance(stopping_condition, StoppingCondition)
        
        results.append(selection_strategy_evaluator.__generate_result(case_base, test_set))
        
        while not stopping_condition.is_criteria_met(case_base, unlabelled_set):
            selections = selection_strategy.select(unlabelled_set)
            
            if not isinstance(selections, collections.Iterable):
                selections = [selections]
            
            for selection in selections:
                selectedExample = orange.Example(selection.selection)
                selection.delete_from(unlabelled_set)
                selectedExample.set_class(oracle.classify(selectedExample))
                
                case_base.append(selectedExample)
                
                result = selection_strategy_evaluator.__generate_result(case_base, test_set)
                
                results.append(result)
            
        return results