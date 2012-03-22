from SelectionStrategyEvaluator import *
from BreadAndButter import *
from CaseProfiling import *
from SelectionStrategy import *
from CompetenceSelectionStrategies import *
from functools import partial
from collections import OrderedDict
from operator import pos, neg
from itertools import combinations

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

wrapped_len = f(len)

from utils import average

def all_pairs_similarity(_set, source, distance_constructor, case_base, include_source=True, 
                         op=average, **kwargs):
    distance_measurer = distance_constructor(case_base)
    similarity_measurer = lambda *args, **kwargs: 1 - distance_measurer(*args, **kwargs)
    my_set = set(_set)
    if include_source:
        my_set.add(source)
    score = op((similarity_measurer(*p) for p in combinations(my_set, 2)))
    return score

def direct_similarity(_set, source, distance_constructor, case_base, include_source=True, 
                         op=sum, **kwargs):
    distance_measurer = distance_constructor(case_base)
    similarity_measurer = lambda *args, **kwargs: 1 - distance_measurer(*args, **kwargs)
    score = op((similarity_measurer(source, o) for o in _set))
    return score

def create_other_case_generic(rcdl, direct_other,  shunt, flip_added, flip_removed, op, 
                              all_set_totaller=wrapped_len, direct_other_totaller=None,
                              shunt_totaller=None, flip_added_totaller=None,
                              flip_removed_totaller=None):
    direct_other_totaller = direct_other_totaller or all_set_totaller
    shunt_totaller = shunt_totaller or all_set_totaller
    flip_added_totaller = flip_added_totaller or all_set_totaller
    flip_removed_totaller = flip_removed_totaller or all_set_totaller
    
    if not (shunt is pos or shunt is neg):
        raise Exception("shunt was neither negative or positive")
    big_op = Plus if shunt is pos else Minus
    return g(big_op(Total(SplitterHider(lambda s, *args, **kwargs: 
                                            sum((direct_other(comp_sum(s.do, 'added', rcdl, direct_other_totaller, **kwargs)),
                                                 flip_added(comp_sum(s.f, 'added', rcdl, flip_added_totaller, **kwargs)),
                                                 flip_removed(comp_sum(s.f, 'removed', rcdl, flip_removed_totaller, **kwargs)))))),
                    Any(SplitterHider(lambda s, *args, **kwargs: comp_sum(s.s, 'removed', rcdl, shunt_totaller, **kwargs)))), 
             op=op)

named_selection_strategy_generators = [
    ("Random Selection", random_selection_strategy_generator),
    ("Uncertainty Sampling", classifier_output_selection_strategy_generator),
    #("Margin Sampling", margin_sampling_selection_strategy_generator),
    #("Maximum Diversity Sampling", maximum_diversity_selection_strategy_generator),
    #("Maximum Density Times Diversity", maximum_dtd_selection_strategy_generator),
    #("CompStrat 1 - New Case - Reachability", 
      #g(Total(SplitterHider(lambda s, *args, **kwargs: comp_sum(s.dn, 'added', 'r', wrapped_len, **kwargs))), 
        #op=take_minimum)),
    
    #("CompStrat 2 - New Case - Coverage", 
     #gen_case_profile_ss_generator(ExampleCoverageOnlyCompetenceMeasure, 
                                  #op=take_minimum)),
    
    #("CompStrat 3 - NCATMaxDMin", 
     #gen_case_profile_ss_generator2(SizeDeviationCombo)),

    #("CompStrat 4 - New Case - Dissimilarity", 
      #g(Total(SplitterHider(lambda s, *args, **kwargs: comp_sum(s.dn, 'added', 'd', wrapped_len, **kwargs))), 
        #op=take_minimum)),
    
    #("CompStrat 5 - New Case - Liability", 
      #g(Total(SplitterHider(lambda s, *args, **kwargs: comp_sum(s.dn, 'added', 'l', wrapped_len, **kwargs))), 
        #op=take_minimum)),
                         
    #("CompStrat 6 - Other Cases - Coverage", 
     #create_other_case_generic('c', pos, pos, neg, pos, take_minimum)),
                                       
    #("CompStrat 7 - Other Cases - Reachability", 
     #create_other_case_generic('r', pos, pos, neg, pos, take_minimum)),
                                       
    #("CompStrat 8 - Other Cases - Liability", 
     #create_other_case_generic('l', lambda e: 0, pos, neg, pos, take_maximum)),
                                       
    #("CompStrat 9 - Other Cases - Dissimilarity", 
     #create_other_case_generic('d', pos, pos, pos, neg, take_minimum)),
                                       
    ("CompStrat: Local Reachability (Similarity)", 
      g(Total(SplitterHider(lambda s, *args, **kwargs: comp_sum(s.dn, 'added', 'r', 
                                                                all_pairs_similarity, 
                                                                **kwargs))), 
        op=take_minimum)),
                                       
    ("CompStrat: Local Coverage (Similarity)", 
      g(Total(SplitterHider(lambda s, *args, **kwargs: comp_sum(s.dn, 'added', 'c', 
                                                                direct_similarity, 
                                                                **kwargs))), 
        op=take_minimum)),
                                       
    ("CompStrat: Local Liability (Similarity)", 
      g(Total(SplitterHider(lambda s, *args, **kwargs: comp_sum(s.dn, 'added', 'l', direct_similarity, **kwargs)),
              class_to_suppositions_preprocessor=combiner_preprocessor), 
        op=take_minimum)),
                                       
    ("CompStrat: Global Coverage (Similarity)", 
     create_other_case_generic('c', pos, neg, neg, pos, take_minimum, all_set_totaller=direct_similarity)),
                                       
    ("CompStrat: Global Reachability (Similarity)", 
     create_other_case_generic('r', pos, neg, neg, pos, take_minimum, all_set_totaller=direct_similarity)),
                                       
    ("CompStrat: Global Liability (Similarity)", 
     create_other_case_generic('l', lambda e: 0, pos, neg, pos, take_maximum, all_set_totaller=direct_similarity)),
                                       
    ("CompStrat: Global Dissimilarity (Similarity)", 
     create_other_case_generic('d', pos, pos, pos, neg, take_minimum, all_set_totaller=direct_similarity)),

]

named_experiment_variations = create_named_experiment_variations(named_selection_strategy_generators)

experiment = create_experiment(stopping_condition_generator, named_experiment_variations)