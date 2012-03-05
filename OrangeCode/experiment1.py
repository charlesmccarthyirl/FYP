from SelectionStrategyEvaluator import *
from BreadAndButter import *
from CaseProfiling import *
from SelectionStrategy import *
from CompetenceSelectionStrategies import *
from functools import partial
from collections import OrderedDict
from operator import pos, neg

stopping_condition_generator = lambda data, *args, **kwargs: PercentageBasedStoppingCriteria(0.1, data, 0)
random_selection_strategy_generator = lambda *args, **kwargs: RandomSelectionStrategy(random_seed=RANDOM_SEED)

competence_measure_generator = ClassifierBasedCompetenceMeasure
margin_sampling_measure_generator = ClassifierBasedMarginSamplingMeasure

classifier_output_selection_strategy_generator = partial(SingleCompetenceSelectionStrategy,  
                                                         competence_measure_generator, 
                                                         SingleCompetenceSelectionStrategy.take_minimum)

margin_sampling_selection_strategy_generator = partial(SingleCompetenceSelectionStrategy,
                                                        margin_sampling_measure_generator, 
                                                        SingleCompetenceSelectionStrategy.take_minimum)

maximum_diversity_selection_strategy_generator = partial(SingleCompetenceSelectionStrategy,
                                                            DiversityMeasure, 
                                                            SingleCompetenceSelectionStrategy.take_maximum)

maximum_dtd_selection_strategy_generator = partial(SingleCompetenceSelectionStrategy,
                                                            DensityTimesDiversityMeasure, 
                                                            SingleCompetenceSelectionStrategy.take_maximum)

def g(measurer, op):
    return gen_case_profile_ss_generator(partial(GenericCompetenceMeasure, measurer), op=op)

def f(func):
    def inner( first_arg, *args, **kwargs):
        return func(first_arg)
    return inner

take_minimum = SingleCompetenceSelectionStrategy.take_minimum
take_maximum = SingleCompetenceSelectionStrategy.take_maximum

def create_other_case_generic(rcdl, direct_other,  shunt, flip_added, flip_removed, op):
    big_op = Plus if shunt is pos else Minus
    return g(big_op(Total(SplitterHider(f(lambda s: sum((direct_other(comp_sum(s.do, 'added', rcdl)),
                                                         flip_added(comp_sum(s.f, 'added', rcdl)),
                                                         flip_removed(comp_sum(s.f, 'removed', rcdl))))))),
                    Any(SplitterHider(f(lambda s: comp_sum(s.s, 'removed', rcdl))))), 
             op=op)

named_selection_strategy_generators = [
    ("Random Selection", random_selection_strategy_generator),
    ("Uncertainty Sampling", classifier_output_selection_strategy_generator),
    ("Margin Sampling", margin_sampling_selection_strategy_generator),
    ("Maximum Diversity Sampling", maximum_diversity_selection_strategy_generator),
    ("Maximum Density Times Diversity", maximum_dtd_selection_strategy_generator),
    ("CompStrat 1 - New Case - Reachability", 
      g(Total(SplitterHider(f(lambda s: comp_sum(s.dn, 'added', 'r')))), 
        op=take_minimum)),
    
    ("CompStrat 2 - New Case - Coverage", 
     gen_case_profile_ss_generator(ExampleCoverageOnlyCompetenceMeasure, 
                                  op=take_minimum)),
    
    ("CompStrat 3 - NCATMaxDMin", 
     gen_case_profile_ss_generator2(SizeDeviationCombo)),

    ("CompStrat 4 - New Case - Dissimilarity", 
      g(Total(SplitterHider(f(lambda s: comp_sum(s.dn, 'added', 'd')))), 
        op=take_minimum)),
    
    ("CompStrat 5 - New Case - Liability", 
      g(Total(SplitterHider(f(lambda s: comp_sum(s.dn, 'added', 'l')))), 
        op=take_minimum)),
                         
    ("CompStrat 6 - Other Cases - Coverage", 
     create_other_case_generic('c', pos, pos, neg, pos, take_minimum)),
                                       
    ("CompStrat 7 - Other Cases - Reachability", 
     create_other_case_generic('r', pos, pos, neg, pos, take_minimum)),
                                       
    ("CompStrat 8 - Other Cases - Liability", 
     create_other_case_generic('l', lambda e: 0, pos, neg, pos, take_maximum)),
                                       
    ("CompStrat 9 - Other Cases - Dissimilarity", 
     create_other_case_generic('d', pos, pos, pos, neg, take_minimum)),
]

named_experiment_variations = create_named_experiment_variations(named_selection_strategy_generators)

experiment = create_experiment(stopping_condition_generator, named_experiment_variations)