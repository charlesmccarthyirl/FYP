from SelectionStrategyEvaluator import *
from BreadAndButter import *
from CaseProfiling import *
from SelectionStrategy import *
from CompetenceSelectionStrategies import *
from functools import partial
from collections import OrderedDict
from operator import pos, neg, itemgetter
from itertools import combinations

stopping_condition_generator = lambda data, *args, **kwargs: PercentageBasedStoppingCriteria(0.1, data, 0)
random_selection_strategy_generator = lambda *args, **kwargs: RandomSelectionStrategy(random_seed=RANDOM_SEED)

competence_measure_generator = ClassifierCertaintyMeasure
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
    
    measures = (similarity_measurer(*p) for p in combinations(my_set, 2))
    score = op(measures)
    return score

def direct_similarity(_set, source, distance_constructor, case_base, op=sum, **kwargs):
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
    ("Margin Sampling", margin_sampling_selection_strategy_generator),
    ("Maximum Diversity Sampling", maximum_diversity_selection_strategy_generator),
    ("Sparsity Minimization",  partial(SingleCompetenceSelectionStrategy,  
                                       SparsityMeasure, 
                                       SingleCompetenceSelectionStrategy.take_minimum)),
    ("Diversity + Density Maximization",  partial(SingleCompetenceSelectionStrategy,  
                                                   Measure.create_measure_constructor(average, [DiversityMeasure, DensityMeasure]), 
                                                   SingleCompetenceSelectionStrategy.take_maximum)),
    ("Certainty + Sparsity Minimization",  partial(SingleCompetenceSelectionStrategy,  
                                                   Measure.create_measure_constructor(average, [ClassifierCertaintyMeasure, SparsityMeasure]), 
                                                   SingleCompetenceSelectionStrategy.take_minimum)),
                         
    ("CompStrat - Global Coverage (Counting)", 
     create_other_case_generic('c', pos, pos, neg, pos, take_minimum)),
                                       
    ("CompStrat - Global Reachability (Counting)", 
     create_other_case_generic('r', pos, pos, neg, pos, take_minimum)),
                                       
    ("CompStrat - Global Liability (Counting)", 
     create_other_case_generic('l', lambda e: 0, pos, neg, pos, take_maximum)),
                                       
    ("CompStrat - Global Dissimilarity (Counting)", 
     create_other_case_generic('d', pos, pos, pos, neg, take_minimum)),
                                       
    ("CompStrat - Global Coverage (Similarity)", 
     create_other_case_generic('c', pos, neg, neg, pos, take_minimum, all_set_totaller=direct_similarity)),
                                       
    ("CompStrat - Global Reachability (Similarity)", 
     create_other_case_generic('r', pos, neg, neg, pos, take_minimum, all_set_totaller=direct_similarity)),
                                       
    ("CompStrat - Global Liability (Similarity)", 
     create_other_case_generic('l', lambda e: 0, pos, neg, pos, take_maximum, all_set_totaller=direct_similarity)),
                                       
    ("CompStrat - Global Dissimilarity (Similarity)", 
     create_other_case_generic('d', pos, pos, pos, neg, take_minimum, all_set_totaller=direct_similarity)),

]

cross_label_aggregators = {"Total": Total, "Deviation": Deviation}
set_item_score_seq_generators = {"All-Pairs Similarity (incl source)": partial(all_pairs_similarity, include_self=True),
                                "All-Pairs Similarity (excl source)": partial(all_pairs_similarity, include_self=False),
                                "Direct Similarity": direct_similarity,
                                "Counting": wrapped_len}
set_item_score_combiners = {"Average": average, "Total": sum}
rcdl_sets = {"Reachability": "r", "Coverage": "c", "Dissimilarity": "d", "Liability": "l"}
all_rcdl_set_combinations = (  list(itertools.combinations(rcdl_sets.items(), 1)) 
                             + list(itertools.combinations(rcdl_sets.items(), 2)) 
                             + list(itertools.combinations(rcdl_sets.items(), 3)) 
                             + [rcdl_sets.items()])
density_inclusions = {"(With Density)": DensityMeasure, "(With Sparsity)": SparsityMeasure, "": None}
localities = {"Local": None, "Global": None }
quantity_goals = {"Minimization": take_minimum, "Maximization": take_maximum}
measure_reducers = {"+": sum}

class SelectionStrategySpecification:
    def __init__(self, locality_kv, rcdl_sets_kvs, set_item_score_combiner_kv, 
                           set_item_score_seq_generator_kv, density_inclusion_kv, 
                           cross_label_aggregator_kv, quantity_goal_kv,
                           measure_reducer_kv):
        self.locality_kv = locality_kv # done xxx
        self.rcdl_sets_kvs = rcdl_sets_kvs # done
        self.set_item_score_combiner_kv = set_item_score_combiner_kv # done
        self.set_item_score_seq_generator_kv = set_item_score_seq_generator_kv # done
        self.density_inclusion_kv = density_inclusion_kv # done
        self.cross_label_aggregator_kv = cross_label_aggregator_kv # done
        self.quantity_goal_kv = quantity_goal_kv # done
        self.measure_reducer_kv = measure_reducer_kv
    
    def get_name(self):
        name_list = [self.locality_kv[0], 
                     (" " + self.measure_reducer_kv[0] + " ").join(map(itemgetter(0), self.rcdl_sets_kvs)), 
                     self.set_item_score_combiner_kv[0], 
                     self.set_item_score_seq_generator_kv[0], self.density_inclusion_kv[0], 
                     "Cross-Label", self.cross_label_aggregator_kv[0], self.quantity_goal_kv[0]]
        name_list = filter(None, name_list)
        experiment_name = " ".join(name_list)
        return experiment_name
    
    def generate_measure_constructor(self):
        # Note, won't contain self.quantity_goal, since this isn't really in the measure
        if self.locality_kv[0] == 'Global':
            raise NotImplementedError()
        
        all_comp_measure_constructors = []
        
        for rcdl_set_kv in self.rcdl_sets_kvs:
            comp_measurer = self.cross_label_aggregator_kv[1](SplitterHider(
                                lambda s, *args, **kwargs: comp_sum(s.dn, 'added', rcdl_set_kv[1], 
                                                                    partial(self.set_item_score_seq_generator_kv[1], 
                                                                            op=self.set_item_score_combiner_kv[1]), 
                                                                    **kwargs)))
            measure_constructor = partial(GenericCompetenceMeasure, comp_measurer) 
            
            if self.density_inclusion_kv[1]:
                measure_constructor = Measure.create_measure_constructor(average, [measure_constructor, self.density_inclusion_kv[1]])
            
            all_comp_measure_constructors.append(measure_constructor)
        
        if len(all_comp_measure_constructors) > 1:
            measure_constructor = Measure.create_measure_constructor(self.measure_reducer_kv[1], all_comp_measure_constructors)

        return measure_constructor
    
    def create_experiment(self):
        measure_constructor = self.generate_measure_constructor()
        return gen_case_profile_ss_generator(measure_constructor, op=self.quantity_goal_kv[1])
    
    def create_experiment_pair(self):
        return (self.get_name(), self.create_experiment())

def accept_multi_rcdl_strat_spec(sel_strat_spec):
    assert(isinstance(sel_strat_spec, SelectionStrategySpecification))
    strat_name = sel_strat_spec.get_name()
    if "Reachability" in strat_name and "Dissimilarity" in strat_name:
        return False
    interesting_exps = [
                        "Total Direct Similarity (With Sparsity) Cross-Label Total Minimization",
                        "Total All-Pairs Similarity (incl source) (With Sparsity) Cross-Label Deviation Minimization",
                        "Total Counting (With Sparsity) Cross-Label Total Minimization",
                        "Total Counting Cross-Label Total Minimization"
                        ]
    return any((e in strat_name for e in interesting_exps))

def sel_strat_filter(sel_strat_spec):
    assert(isinstance(sel_strat_spec, SelectionStrategySpecification))
    return not (   (sel_strat_spec.locality_kv[0] == "Global")
                or (sel_strat_spec.quantity_goal_kv[0] == "Maximization")
                or (sel_strat_spec.set_item_score_seq_generator_kv[0] == "Counting" and sel_strat_spec.set_item_score_combiner_kv[0] == "Average")
                or (sel_strat_spec.density_inclusion_kv[1] == DensityMeasure)
                or ('Dissimilarity' in map(itemgetter(0), sel_strat_spec.rcdl_sets_kvs))
                ) and (len(sel_strat_spec.rcdl_sets_kvs) == 1 or accept_multi_rcdl_strat_spec(sel_strat_spec))
    
all_possible_exp_specs = [SelectionStrategySpecification(*p) 
                          for p in itertools.product(localities.iteritems(), 
                                                     all_rcdl_set_combinations, 
                                                     set_item_score_combiners.iteritems(), 
                                                     set_item_score_seq_generators.iteritems(), 
                                                     density_inclusions.iteritems(), 
                                                     cross_label_aggregators.iteritems(), 
                                                     quantity_goals.iteritems(),
                                                     measure_reducers.iteritems())]

wanted_sel_strat_specs = filter(sel_strat_filter, all_possible_exp_specs)

wanted_sel_strategy_generators = [e.create_experiment_pair() for e in wanted_sel_strat_specs]

named_selection_strategy_generators += wanted_sel_strategy_generators

named_experiment_variations = create_named_experiment_variations(named_selection_strategy_generators)

experiment = create_experiment(stopping_condition_generator, named_experiment_variations)