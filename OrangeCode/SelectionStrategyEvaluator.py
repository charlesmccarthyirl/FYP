'''
Created on 23 Oct 2011

@author: Charles McCarthy
'''

from __future__ import division
import collections
from cross_validation import KFold
from pyx import graph, text, color
import logging
import csv
from itertools import compress, imap, izip, islice
import random

def to_numerically_indexed(sequence):
    index_dict = {}
    next_index = 0
    for e in sequence:
        index = index_dict.get(e)
        if index is None:
            index = next_index
            next_index += 1
            index_dict[e] = index
        yield index

def n_fold_cross_validation(data, n, true_oracle, rand_seed=None):
    if rand_seed is not None:
        data = list(data)
        random_function = random.Random(rand_seed).random
        random.shuffle(data, random_function)
#    classes = imap(true_oracle, data)
#    classes_as_numbers = list(to_numerically_indexed(classes))
#    train_test_bit_maps = StratifiedKFold(classes_as_numbers, n)
    train_test_bit_maps = KFold(len(data), n)
    return ((list(compress(data, train)), list(compress(data, test)))  
           for train, test in train_test_bit_maps)

class Result:
    def __init__(self, case_base_size=0, classification_accuracy=0):
        self.case_base_size = case_base_size
        self.classification_accuracy = classification_accuracy

class ResultSet(list):
    def read_csv(self, stream):
        logging.debug("Starting CSV reading from stream %s" % stream)
        
        reader = csv.reader(stream)
        rows = ((float(cell) if '.' in cell else int(cell)for cell in row) 
                for row in islice(reader, 1, None) if len(row) > 1)
        self.extend((Result(*row) for row in rows))
        
        logging.debug("Ending CSV reading from stream %s" % stream)
    
    def write_csv(self, stream):
        logging.debug("Starting CSV generation on stream %s" % stream)
        
        writer = csv.writer(stream)
        writer.writerow(("Case base size", "Classification Accuracy"))
        orderedResults = sorted(self, key=lambda x: x.case_base_size)
        writer.writerows(((result.case_base_size, 
                           result.classification_accuracy) 
                          for result in orderedResults))
        
        logging.debug("Ending CSV generation on stream %s" % stream)
    
    def AULC(self):
        '''
        Calculates the area under the learning curve, based on simple (rectangle + top triangle)
        area for each couple of Results.
        
        >>> r = ResultSet()
        >>> r.append(Result(4, 5))
        >>> r.append(Result(0, 1))
        >>> r.append(Result(1, 1))
        >>> r.AULC()
        10.0
        '''
        logging.debug("Starting calculating AULC")
        
        # Should be sorted already - but just in case . . .
        orderedResults = sorted(self, key=lambda x: x.case_base_size)
        previous_result = Result()
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
        
        logging.debug("Finishing calculating AULC")
        
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
        return len(case_base) - self._initial_case_base_size >= self._budget

class PercentageBasedStoppingCriteria(BudgetBasedStoppingCriteria):
    def __init__ (self, fraction, data, initial_case_base_size):
        BudgetBasedStoppingCriteria.__init__(self, round(len(data)*fraction), initial_case_base_size)

class Oracle:
    def __init__(self, classifyLambda):
        self.classify = classifyLambda
    
    def __call__(self, instance):
        self.classify(instance)

def average(iterable):
    total = 0
    length = 0
    for i in iterable:
        total += i
        length += 1
        
    return total / length

def add_dicts(dict1, dict2):
    result = dict1.copy()
    result.update(dict2)
    return result

class SelectionStrategyEvaluator:
    def __init__(self, 
                 oracle_generator, 
                 true_oracle,
                 stopping_condition_generator, 
                 selection_strategy_generator,
                 classifier_generator,
                 **kwargs):
        self.oracle_generator = oracle_generator
        self.true_oracle = true_oracle
        self.stopping_condition_generator = stopping_condition_generator
        self.selection_strategy_generator = selection_strategy_generator
        self.classifier_generator = classifier_generator
        self.kwargs = kwargs
    
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
                                            average((result.classification_accuracy 
                                                     for result in aligned_result))) 
                                     for aligned_result 
                                     in aligned_results)
        
        return ResultSet(averaged_result_instances)
    
    def generate_ca(self, t, p):
        count = 0
        correct = 0
        for (ti, pi) in izip(t, p):
            count += 1
            if ti == pi:
                correct += 1
        
        if count == 0:
            return 0
        
        return correct / count
    
    def generate_ca_of_classifier(self, classifier, test_set):
        return self.generate_ca(imap(self.true_oracle, test_set), imap(classifier, test_set))
    
    def __generate_result(self, case_base, test_set):
        case_base_size = len(case_base)
        try:
            classifier = self.classifier_generator(case_base, **self.kwargs)
            classification_accuracy = self.generate_ca_of_classifier(classifier, test_set)
        except:
            if case_base_size != 0:
                raise
            # Some classifiers have issues with 0 examples in the training set.
            classification_accuracy = 0
        
        return Result(case_base_size, classification_accuracy)
    
    def generate_results(self, unlabelled_set, test_set):
        results = ResultSet()
        
        if not isinstance(unlabelled_set, list):
            unlabelled_set = list(unlabelled_set)
        
        # hacky Laziness. Just don't want to have to do **locals() myself, but I can't pass self.
        selection_strategy_evaluator = self
        del(self)
        
        # Order of assignment here important so that **locals has the right info (e.g. the stopping_criteria may care about the oracle)
        case_base = [] 
        selection_strategy = selection_strategy_evaluator.selection_strategy_generator(**add_dicts(locals(), selection_strategy_evaluator.kwargs))
        oracle = selection_strategy_evaluator.oracle_generator(**add_dicts(locals(), selection_strategy_evaluator.kwargs))
        stopping_condition = selection_strategy_evaluator.stopping_condition_generator(**add_dicts(locals(), selection_strategy_evaluator.kwargs))
        
        results.append(selection_strategy_evaluator.__generate_result(case_base, test_set))
        
        while not stopping_condition.is_criteria_met(case_base, unlabelled_set):
            selections = selection_strategy.select(unlabelled_set)
            
            if not isinstance(selections, collections.Iterable):
                selections = [selections]
            
            for selection in selections:
                del(unlabelled_set[selection.index])
                case_base.append(selection.selection)

            logging.debug("Starting testing with case base size of %d and test set size of %d" % (len(case_base), len(test_set)))
            result = selection_strategy_evaluator.__generate_result(case_base, test_set)
            results.append(result)
            logging.debug("Finishing testing with case base size of %d and test set size of %d" % (len(case_base), len(test_set))) 
        
        return results
    
class ExperimentVariation:
    def __init__(self, classifier_generator, selection_strategy):
        self.classifier_generator = classifier_generator
        self.selection_strategy = selection_strategy
    
class Experiment:
    def __init__(self, 
                 oracle_generator,
                 true_oracle,
                 stopping_condition_generator,
                 training_test_sets_extractor,
                 named_experiment_variations_generator):
        self.oracle_generator = oracle_generator
        self.true_oracle = true_oracle
        self.stopping_condition_generator = stopping_condition_generator
        self.training_test_sets_extractor = training_test_sets_extractor
        self.named_experiment_variations_generator = named_experiment_variations_generator
        
    def execute_on(self, data_distance_constructor_pair, existing_named_variation_results=None): 
        named_variation_results = ExperimentResult()
        
        data, distance_constructor = data_distance_constructor_pair
        
        named_experiment_variations = self.named_experiment_variations_generator(data)
        
        for (variation_name, variation) in named_experiment_variations.items():
            assert isinstance(variation, ExperimentVariation)
            if existing_named_variation_results.has_key(variation_name):
                logging.info("Already have results for %s. Skipping evaluation." % variation_name)
                named_variation_results[variation_name] = existing_named_variation_results[variation_name]
                continue
            
            evaluator = SelectionStrategyEvaluator(self.oracle_generator, 
                                                   self.true_oracle,
                                                   self.stopping_condition_generator,
                                                   variation.selection_strategy,
                                                   variation.classifier_generator,
                                                   data=data,
                                                   distance_constructor=distance_constructor)
            
            logging.info("Starting evaluation on variation %s" % variation_name)
            named_variation_results[variation_name] = evaluator.generate_results_from_many(self.training_test_sets_extractor(data))
            logging.info("Finishing evaluation on variation %s" % variation_name)

        return named_variation_results

class ExperimentResult(dict):
    def load_from_csvs(self, name_to_stream_generator_pairs):
        for (variation_name, stream_generator) in name_to_stream_generator_pairs:
            with stream_generator() as stream:                
                result_set = ResultSet()
                result_set.read_csv(stream)
                self[variation_name] = result_set
                
    
    def write_to_csvs(self, stream_from_name_getter):
        for (variation_name, result_set) in self.items():
            with stream_from_name_getter(variation_name) as stream:
                result_set.write_csv(stream)
    
    def generate_graph(self, title=None):
        logging.debug("Starting graph generation")
        
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
            title = title.replace("_", r"\_")
            g.text(g.width/2, 
                   g.height + 0.2, 
                   title,
                   [text.halign.center, text.valign.bottom, text.size.Large])
        
        logging.debug("Finishing graph generation")
        
        return g