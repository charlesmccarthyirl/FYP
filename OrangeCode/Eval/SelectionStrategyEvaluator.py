'''
Created on 23 Oct 2011

@author: Charles McCarthy
'''

import collections
import orange, orngStat, orngTest
from pyx import *
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

class PercentageBasedStoppingCriteria(BudgetBasedStoppingCriteria):
    def __init__ (self, fraction, data, initial_case_base_size):
        BudgetBasedStoppingCriteria.__init__(self, len(data)*fraction, initial_case_base_size)

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
    
class ExperimentVariation:
    def __init__(self, classifier_generator, selection_strategy):
        self.classifier_generator = classifier_generator
        self.selection_strategy = selection_strategy
    

class Experiment:
    def __init__(self, 
                 oracle_generator, 
                 stopping_condition_generator,
                 training_test_sets_extractor,
                 named_experiment_variations):
        self.oracle_generator = oracle_generator
        self.stopping_condition_generator = stopping_condition_generator
        self.training_test_sets_extractor = training_test_sets_extractor
        self.named_experiment_variations = named_experiment_variations
        
    def execute_on(self, data): 
        named_variation_results = ExperimentResult()  
        
        stopping_condition_generator = lambda *args, **kwargs: self.stopping_condition_generator(*args, data=data, **kwargs)
        
        for (variation_name, variation) in self.named_experiment_variations.items():
            assert isinstance(variation, ExperimentVariation)
            
            evaluator = SelectionStrategyEvaluator(self.oracle_generator, 
                                                   stopping_condition_generator,
                                                   variation.selection_strategy,
                                                   variation.classifier_generator)
            
            named_variation_results[variation_name] = evaluator.generate_results_from_many(self.training_test_sets_extractor(data))

        return named_variation_results

class ExperimentResult(dict):
    def generate_graph(self, title=None):
        max_x=max((max((result.case_base_size 
                       for result in result_set)) 
                  for result_set in self.values()))
        max_y=1.0
        
        g = graph.graphxy(width=10,
                          height=10, # Want a square graph . . .
                          x=graph.axis.linear(title="Case Base Size", min=0, max=max_x), #This might seem redundant - but pyx doesn't handle non-varying y well. So specifying the min and max avoids that piece of pyx code.
                          y=graph.axis.linear(title="Classification Accuracy", min=0, max=max_y),
                          key=graph.key.key(pos="br", dist=0.1))
        
        # either provide lists of the individual coordinates
        points = [graph.data.values(x=[result.case_base_size for result in result_set], 
                                    y=[result.classification_accuracy for result in result_set], 
                                    title="%s (AULC: %.3f)" % (name, result_set.AULC())) 
                  for (name, result_set) in self.items()]
        
        g.plot(points, [graph.style.line([color.gradient.Rainbow])])
        
        if (title):
            g.text(g.width/2, 
                   g.height + 0.2, 
                   title,
                   [text.halign.center, text.valign.bottom, text.size.Large])
        
        return g