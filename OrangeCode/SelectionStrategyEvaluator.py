'''
Created on 23 Oct 2011

@author: Charles McCarthy
'''

from __future__ import division
import collections
from cross_validation import KFold
from PrecomputedDistance import Instance
from SelectionStrategy import Selection
from Knn import KNN
import math
try:
    import pyx
except:
    pass

try:
    import pygraphviz
    from pyPdf import PdfFileWriter, PdfFileReader
except:
    pass

import sys
import logging
import csv
from itertools import compress, imap, izip, islice, izip_longest, chain, combinations
import random
import tarfile, StringIO
from os.path import splitext
from utils import average


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

def n_fold_cross_validation(data, n, true_oracle, random_seed=None):
    if random_seed is not None:
        data = list(data)
        random_function = random.Random(random_seed).random
        random.shuffle(data, random_function)
    train_test_bit_maps = KFold(len(data), n)
    return ((list(compress(data, train)), list(compress(data, test)))  
           for train, test in train_test_bit_maps)

class Result:
    def __init__(self, case_base_size=0, classification_accuracy=0, selections=None):
        self.case_base_size = case_base_size
        self.classification_accuracy = classification_accuracy
        self.selections = Instance.multiple_from(selections) if selections is not None else selections        

class ResultSet(list):
    def __init__(self, training_set=None, test_set=None):
        self.training_set = list(training_set or [])
        self.test_set = list(test_set or [])
        list.__init__(self)
    
    def deserialize(self, stream):
        logging.debug("Starting CSV reading from stream %s" % stream)
        
        reader = csv.reader(stream)
        rows = [[float(cell) if '.' in cell 
                             else int(cell) if cell.isdigit() 
                                            else cell
                for cell in row]
                for row in islice(reader, 1, None) if len(row) > 1]
        self.extend((Result(*row[:3]) for row in rows if row[0] is not None and row[0] != ""))
        
        self.training_set.extend((Instance.single_from(row[3]) for row in rows if row[3]))
        self.test_set.extend((Instance.single_from(row[4]) for row in rows if row[4]))
        
        logging.debug("Ending CSV reading from stream %s" % stream)
    
    def serialize(self, stream):
        logging.debug("Starting CSV generation on stream %s" % stream)
        
        writer = csv.writer(stream)
        writer.writerow(("Case base size", "Classification Accuracy", "Selected", "Training Set", "Test Set"))
        orderedResults = sorted(self, key=lambda x: x.case_base_size)
        
        main_results = list(((result.case_base_size, 
                           result.classification_accuracy,
                           repr(result.selections) if result.selections else None) 
                          for result in orderedResults))
        
        results_zipped = list(izip_longest(main_results, self.training_set or [], self.test_set or []))
        results_expanded = (list(chain((res or [None]*3), (train, ), (test, ))) for (res, train, test) in results_zipped)
        
        writer.writerows(results_expanded)
        
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
        
class MultiResultSet(ResultSet):
    def __init__(self, all_results=None):
        all_results = list(all_results or [])
        self.all_results = all_results
        
        aligned_results = zip(*all_results)
        
        ResultSet.__init__(self)
        
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
        
        self.extend(averaged_result_instances)
        
    def serialize(self, stream):
        with tarfile.open(mode="w:gz", fileobj=stream) as tar:
            for (i, resultset) in chain(enumerate(self.all_results), (("summary", self),)):
                mem_stream = StringIO.StringIO()
                ResultSet.serialize(resultset, mem_stream)
                mem_stream.seek(0)
                
                info = tar.tarinfo("%s.csv" % i)
                info.size = len(mem_stream.buf)
                
                tar.addfile(info, fileobj=mem_stream)
        
    def deserialize(self, stream):
        with tarfile.open(mode="r", fileobj=stream) as tar:
            new_all_results = [None] * (len(tar.getnames()) - 1)
            for info in tar:
                name = splitext(info.name)[0]
                if name == "summary":
                    result_set = self
                else:
                    i = int(name)
                    result_set = ResultSet()
                    new_all_results[i] = result_set
                
                ResultSet.deserialize(result_set, tar.extractfile(info))
                
            assert(all(new_all_results))
            self.all_results.extend(new_all_results)

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
        return self.classify(instance)



def add_dicts(dict1, dict2):
    result = dict1.copy()
    result.update(dict2)
    return result

class SelectionStrategyEvaluator:
    def __init__(self, 
                 oracle_generator, 
                 oracle,
                 stopping_condition_generator, 
                 selection_strategy_generator,
                 classifier_generator,
                 **kwargs):
        self.oracle_generator = oracle_generator
        self.oracle = oracle
        self.stopping_condition_generator = stopping_condition_generator
        self.selection_strategy_generator = selection_strategy_generator
        self.classifier_generator = classifier_generator
        self.kwargs = kwargs
    
    def generate_results_from_many(self, data_test_iterable):
        all_results = [self.generate_results(test_set, unlabelled_set)
                       for (test_set, unlabelled_set) 
                       in data_test_iterable]

        return MultiResultSet(all_results)
    
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
        return self.generate_ca(imap(self.oracle, test_set), imap(classifier, test_set))
    
    def __generate_result(self, case_base, test_set, selections):
        if selections is not None:
            selections = [sel.selection for sel in selections] # selections are a bunch of Selections - need to just have the underlying objects, not the associated index of the selection aswell
        
        case_base_size = len(case_base)
        try:
            classifier = self.classifier_generator(case_base, oracle=self.oracle, **self.kwargs)
            classification_accuracy = self.generate_ca_of_classifier(classifier, test_set)
        except:
            if case_base_size != 0:
                raise
            # Some classifiers have issues with 0 examples in the training set.
            classification_accuracy = 0
        
        return Result(case_base_size, classification_accuracy, selections)
    
    def generate_results(self, unlabelled_set, test_set):
        results = ResultSet(unlabelled_set, test_set)
        
        data = list(unlabelled_set)
        unlabelled_set = list(unlabelled_set)
        
        classifier_generator = self.classifier_generator # just so it's in locals()
        # hacky Laziness. Just don't want to have to do **locals() myself, but I can't pass self.
        selection_strategy_evaluator = self
        del(self)
        
        # Order of assignment here important so that **locals has the right info (e.g. the stopping_criteria may care about the oracle)
        case_base = [] 

        oracle = selection_strategy_evaluator.oracle_generator(**add_dicts(locals(), selection_strategy_evaluator.kwargs))
        selection_strategy = selection_strategy_evaluator.selection_strategy_generator(**add_dicts(locals(), selection_strategy_evaluator.kwargs))
        stopping_condition = selection_strategy_evaluator.stopping_condition_generator(**add_dicts(locals(), selection_strategy_evaluator.kwargs))
        
        results.append(selection_strategy_evaluator.__generate_result(case_base, test_set, None))
        
        while not stopping_condition.is_criteria_met(case_base, unlabelled_set):
            selections = selection_strategy.select(unlabelled_set)
            
            if not isinstance(selections, collections.Iterable):
                selections = [selections]
            
            for selection in selections:
                if isinstance(selection, Selection):
                    index = selection.index
                    selection = selection.selection
                else:
                    index = unlabelled_set,index(selection)
                
                del(unlabelled_set[index])
                case_base.append(selection)

            logging.debug("Starting testing with case base size of %d and test set size of %d" % (len(case_base), len(test_set)))
            result = selection_strategy_evaluator.__generate_result(case_base, test_set, selections)
            results.append(result)
            logging.debug("Finishing testing with case base size of %d and test set size of %d" % (len(case_base), len(test_set))) 
        
        return results
    
# TODO: Put classifier stuff in its own class

class ExperimentVariation:
    def __init__(self, classifier_generator, probability_generator, nns_getter_generator, selection_strategy):
        self.classifier_generator = classifier_generator
        self.probability_generator = probability_generator
        self.nns_getter_generator = nns_getter_generator
        self.selection_strategy = selection_strategy
    
class Experiment:
    def copy(self):
        return Experiment(self.oracle_generator_generator, 
                          self.stopping_condition_generator, 
                          self.training_test_sets_extractor, 
                          self.named_experiment_variations_generator)
    
    def __init__(self, 
                 oracle_generator_generator,
                 stopping_condition_generator,
                 training_test_sets_extractor,
                 named_experiment_variations_generator):
        self.oracle_generator_generator = oracle_generator_generator
        self.stopping_condition_generator = stopping_condition_generator
        self.training_test_sets_extractor = training_test_sets_extractor
        self.named_experiment_variations_generator = named_experiment_variations_generator
        
    def execute_on(self, data_info, existing_named_variation_results=None): 
        named_variation_results = ExperimentResult()

        named_experiment_variations = self.named_experiment_variations_generator(data_info.data, 
                                                                                 data_info.oracle)
        
        for (variation_name, variation) in named_experiment_variations.items():
            assert isinstance(variation, ExperimentVariation)
            if existing_named_variation_results.has_key(variation_name):
                logging.info("Already have results for %s. Skipping evaluation." % variation_name)
                variation_result = existing_named_variation_results[variation_name]
            else:
                evaluator = SelectionStrategyEvaluator(self.oracle_generator_generator(data_info.oracle), 
                                                       data_info.oracle,
                                                       self.stopping_condition_generator,
                                                       variation.selection_strategy,
                                                       variation.classifier_generator,
                                                       probability_generator=variation.probability_generator,
                                                       nns_getter_generator=variation.nns_getter_generator,
                                                       distance_constructor=data_info.distance_constructor,
                                                       possible_classes=data_info.possible_classes)
                logging.info("Starting evaluation on variation %s" % variation_name)
                variation_result = evaluator.generate_results_from_many(self.training_test_sets_extractor(data_info.data, data_info.oracle))
                logging.info("Finishing evaluation on variation %s" % variation_name)
                
            named_variation_results[variation_name] = variation_result
            

        return named_variation_results

class ExperimentResult(dict):
    def load_from_csvs(self, name_to_stream_generator_pairs):
        for (variation_name, stream_generator) in name_to_stream_generator_pairs:
            with stream_generator() as stream:                
                result_set = MultiResultSet()
                result_set.deserialize(stream)
                self[variation_name] = result_set
                
    
    def write_to_csvs(self, stream_from_name_getter):
        for (variation_name, result_set) in self.items():
            with stream_from_name_getter(variation_name) as stream:
                result_set.serialize(stream)
                
    def write_to_selection_graphs(self, stream_from_name_getter, 
                                  data_info, write_all_selections=True):
        if not (sys.modules.has_key('pygraphviz') and sys.modules.has_key('pyPdf')):
            raise ImportError('pygraphviz/pypdf not available on this system.')
        
        logging.info("Beginning Graph Generation")
        
        def add_edge(g, a, b, length):
            g.add_edge(a, b, len=length)
            
        def set_node_as_test(g, n):
            g.get_node(n).attr['color'] = 'white'
        
        def set_node_as_train(g, n):
            g.get_node(n).attr['color'] = 'black'
        
        def set_node_selected(g, n):
            g.get_node(n).attr['color'] = 'green'
        
        logging.debug("Beginning Add Graph Data")
        
        dm = data_info.distance_constructor(data_info.data)
        G = pygraphviz.AGraph(overlap='scalexy',  splines='false', aspect='1.333')
        
        G.edge_attr['color'] = 'gray'
        G.node_attr['style'] = 'filled'
        
        def m_key(d):
            if d == 0:
                return 1000000
            return d
        
        logging.debug("Calculating min")
        min_dist = min((dm(*p) for p in combinations(data_info.data, 2)), key=m_key)
        logging.debug("Finished Calculating min")
        
        for a in data_info.data:
            nearest = KNN.s_find_nearest(a, data_info.data, int(math.sqrt(len(data_info.data))), dm)
            for n in nearest:
                if G.has_edge(a, n):
                    continue
                dist = dm(a, n)
                if (dist == 0):
                    dist = min_dist
                add_edge(G, a, n, dist)
        
        for n in G.nodes_iter():
            set_node_as_train(G, n)
        
        logging.debug("Ending Add Graph Data")
        
        logging.debug("Beginning Graph Layout")
        G.layout(prog="neato")
        logging.debug("Ending Graph Layout")

        for (variation_name, multi_result_set) in self.items():
            logging.debug("Generating graph for variation %s" % variation_name)
            stream = stream_from_name_getter(variation_name)
            if stream is None:
                continue
            
            with stream as stream, \
                 tarfile.open(mode="w:gz", fileobj=stream) as tar:
    
                    for (cv_no, resultset) in enumerate(multi_result_set.all_results):
                        logging.debug("Generating graph for cv %d" % cv_no)
                        output = PdfFileWriter()
                        
                        for e in resultset.test_set:
                            set_node_as_test(G, e)
    
                        for (j, result) in enumerate(sorted(resultset, 
                                                            key=lambda r: r.case_base_size)):
                            _format="pdf"
                            
                            if result.selections is not None:
                                for sel in result.selections:
                                    set_node_selected(G, sel)
                            
                            if write_all_selections or j == len(resultset) - 1: # Changing to just gen-ing last.
                                mem_stream = StringIO.StringIO()
                                G.draw(mem_stream, format=_format)
                                mem_stream.seek(0)
                                
                                input1 = PdfFileReader(mem_stream)
                                output.addPage(input1.getPage(0))
                        
                        mem_stream = StringIO.StringIO()
                        output.write(mem_stream)
                        mem_stream.seek(0)    
                        info = tar.tarinfo("CV%02d.%s" % (cv_no, _format))
                        info.size = len(mem_stream.buf)
    
                        tar.addfile(info, fileobj=mem_stream)
                        
                        for e in resultset.test_set:
                            set_node_as_train(G, e)
                        
        logging.info("Ending Graph Generation")
    
    def generate_graph(self, title=None):
        if not sys.modules.has_key('pyx'):
            raise ImportError('pyx not available on this system.')
        logging.debug("Starting graph generation")
        
        max_x=max((max((result.case_base_size 
                       for result in result_set)) 
                  for result_set in self.values()))
        max_y=1.0
        g = pyx.graph.graphxy(width=10,
                          height=10, # Want a square graph . . .
                          x=pyx.graph.axis.linear(title="Case Base Size", min=0, max=max_x), #This might seem redundant - but pyx doesn't handle non-varying y well. So specifying the min and max avoids that piece of pyx code.
                          y=pyx.graph.axis.linear(title="Classification Accuracy", min=0, max=max_y),
                          key=pyx.graph.key.key(pos="mr",hinside=0)) #http://www.physik.tu-dresden.de/~baecker/python/pyxgraph/examples.ps.gz
        
        # either provide lists of the individual coordinates
        points = [pyx.graph.data.values(x=[result.case_base_size for result in result_set], 
                                    y=[result.classification_accuracy for result in result_set], 
                                    title="%s (AULC: %.3f)" % (name, result_set.AULC())) 
                  for (name, result_set) in self.items()]
        
        g.plot(points, [pyx.graph.style.line([pyx.color.gradient.ReverseRainbow])])
        
        if (title):
            title = title.replace("_", r"\_")
            g.text(g.width/2, 
                   g.height + 0.2, 
                   title,
                   [pyx.text.halign.center, pyx.text.valign.bottom, pyx.text.size.Large])
        
        logging.debug("Finishing graph generation")
        
        return g