'''
Created on Mar 3, 2012

@author: charles
'''
from __future__ import division
import random
from utils import average, meanstdv, lazyproperty, first
from itertools import permutations, imap
from CaseProfiling import RcdlCaseProfile, AddRemovalStore, SuppositionResults
from functools import partial
from operator import add, sub
from math import ceil
from SelectionStrategy import CompetenceMeasure, SelectionStrategy
import itertools
from copy import copy

class CaseProfileBasedCompetenceMeasure(CompetenceMeasure):
    def __init__(self, probability_generator, case_base, case_profile_builder=None, possible_classes=None, oracle=None, *args, **kwargs):
        self.case_profile_builder = case_profile_builder
        self._probability_getter = probability_generator(case_base, *args, possible_classes=possible_classes, oracle=oracle, **kwargs)
        self.case_base = case_base
        self.possible_classes = possible_classes
        self.oracle = oracle

    def measure(self, example):
        pass

class ExampleCoverageOnlyCompetenceMeasure(CaseProfileBasedCompetenceMeasure):
    def measure(self, example):
        class_to_supposition_results = self.case_profile_builder.suppose_multiple(example, self.possible_classes)
        class_to_scores = [(c, len(r.get(example, AddRemovalStore()).added.coverage_set)) for (c, r) in class_to_supposition_results]
        total = sum((el[1] for el in class_to_scores))
        class_to_scores_normalized = [(c, score/total if total > 0 else score) for (c, score) in class_to_scores]
        mean, std = meanstdv([cov_len for (c, cov_len) in class_to_scores_normalized])
        
        return std
            
class CompetenceBasedSelectionStrategy(SelectionStrategy):
    def __init__(self, probability_generator, case_base, case_profile_builder=None, possible_classes=None, oracle=None, *args, **kwargs):
        self.case_profile_builder = case_profile_builder
        self._probability_getter = probability_generator(case_base, *args, possible_classes=possible_classes, oracle=oracle, **kwargs)
        self.case_base = case_base
        self.possible_classes = possible_classes
        self.oracle = oracle
        SelectionStrategy.__init__(self, *args, **kwargs)
        
    def select(self, collection):
        collection_supposes = [(e, self.case_profile_builder.suppose_multiple(e, self.possible_classes))
                                for e in collection]  
        selected = self._select_from_supposes(collection_supposes)
        self.case_profile_builder.put(selected)
        return selected
    
    def _select_from_supposes(self, collection_supposes):
        pass

class SizeDeviationCombo(CompetenceBasedSelectionStrategy):
    def __calculate_size_deviation_pair(self, example, class_to_supposition_results):
        class_to_scores = [(c, len(r.get(example, AddRemovalStore()).added.coverage_set)) for (c, r) in class_to_supposition_results]
        total = sum((el[1] for el in class_to_scores))
        mean, std = meanstdv([cov_len for (c, cov_len) in class_to_scores])
        return (total, std)
        
    def _select_from_supposes(self, collection_supposes):
        temp = ((e, self.__calculate_size_deviation_pair(e, r)) for (e, r) in collection_supposes)
        temp = [(e, (total, std/total if total > 0 else std)) for (e, (total, std)) in temp]
        temp.sort(key=lambda el: el[1][0], reverse=True) # Size first
        top_bit_size = int(ceil(len(temp) * 0.25))
        temp = temp[:top_bit_size]
        el = min(temp, key=lambda el: el[1][1]) # Now deviation
        return el[0]

class Splitter:
    def __init__(self, supposition_results):
        '''
        Hides supposition results behind a function
        
        @param supposition_results:
        
            'a' - all (combined) supposition_results
            'd' - direct (new + other direct) supposition results
            's' - shunt supposition results
            'f' - flip supposition results
            'dn' - supposition results of just the new Case's directs
            'do' - supposition results of just the other Cases directs.
        '''
        self.supposition_results = supposition_results
    
    @lazyproperty
    def dsf(self):
        return self.supposition_results.extract_directs_shunts_flips()
    
    @property
    def a(self):
        return self.supposition_results
    
    @property
    def d(self):
        return self.dsf[0]
    
    @property
    def s(self):
        return self.dsf[1]
    
    @property
    def f(self):
        return self.dsf[2]
    
    @property
    def dn(self):
        a = copy(self.d)
        temp = a[a.supposed_case]
        a.clear()
        a[a.supposed_case] = temp
        return a    
    
    @property
    def do(self):
        a = copy(self.d)
        del(a[a.supposed_case])
        return a
    
    def __call__(self, char):
        return getattr(self, char)

def comp_sum(supposition_results, ar, rcdl, set_totaller=len):
    if isinstance(ar, str):
        add_remove_names = ('added', 'removed')
        first_drill_down = first(lambda el: el[0] == ar[0], add_remove_names)
        ar = lambda sr: getattr(sr, first_drill_down)
    if isinstance(rcdl, str):
        sets = ('reachability_set', 'coverage_set', 'dissimilarity_set', 'liability_set')
        set_names = [first(lambda el: el[0] == letter, sets) for letter in rcdl]
        rcdl = lambda profile: [getattr(profile, sn) for sn in set_names]
    
    assert isinstance(supposition_results, SuppositionResults)
    return sum(sum(imap(set_totaller, rcdl(ar(change)))) 
        for change in supposition_results.itervalues())

class SplitterHider:
    def __init__(self, score_getter):
        '''
        
        @param score_getter: Given a func which it may call to get
            
        '''
        self.score_getter = score_getter
            
    def __call__(self, supposition_results, *args, **kwargs):
        splitter = Splitter(supposition_results)
        return self.score_getter(splitter, *args, **kwargs)

class TotalOp:
    def __init__(self, reducer, score_getter):
        '''
        
        @param score_getter: Given supposition results, returns a double of the score
        '''
        self.reducer = reducer
        self.score_getter = score_getter
        
    def __call__(self, class_to_supposition_results, *args, **kwargs):
        scores = (self.score_getter(supposition_results, *args, **kwargs) 
                    for (label, supposition_results) in class_to_supposition_results)
        return self.reducer(scores)

class Total(TotalOp):
    def __init__(self, *args, **kwargs):
        TotalOp.__init__(self, sum, *args, **kwargs)

class Any(TotalOp):
    def __init__(self, *args, **kwargs):
        TotalOp.__init__(self, lambda seq: iter(seq).next(), *args, **kwargs)

class Oper:
    def __init__(self, bi_operator, *operands):
        self.bi_operator = bi_operator
        self.operands = operands
        
    def __call__(self, *args, **kwargs):
        return reduce(self.bi_operator, (f(*args, **kwargs) for f in self.operands))

class Plus(Oper):
    def __init__(self, *operands):
        Oper.__init__(self, add, *operands)

class Minus(Oper):
    def __init__(self, *operands):
        Oper.__init__(self, sub, *operands)

class GenericCompetenceMeasure(CaseProfileBasedCompetenceMeasure):
    def __init__(self, measurer, *args, **kwargs):
        '''
        
        @param measurer: given (class_to_supposition_results, probability_getter, *args, **kwargs)
        '''
        self.measurer = measurer
        self.args = args
        self.kwargs = kwargs
        
        CaseProfileBasedCompetenceMeasure.__init__(self,*args, **kwargs)
    
    def measure(self, example):
        class_to_supposition_results = self.case_profile_builder.suppose_multiple(example, self.possible_classes)
        return self.measurer(example=example, class_to_supposition_results=class_to_supposition_results, 
                             probability_getter=self._probability_getter, 
                             *self.args, **self.kwargs)
        