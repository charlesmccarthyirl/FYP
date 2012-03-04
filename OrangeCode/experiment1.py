from SelectionStrategyEvaluator import *
from BreadAndButter import *
from CaseProfiling import *
from SelectionStrategy import *
from CompetenceSelectionStrategies import *
from functools import partial
from collections import OrderedDict

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


named_selection_strategy_generators = [
    ("Random Selection", random_selection_strategy_generator),
    ("Uncertainty Sampling", classifier_output_selection_strategy_generator),
    ("Margin Sampling", margin_sampling_selection_strategy_generator),
    ("Maximum Diversity Sampling", maximum_diversity_selection_strategy_generator),
    ("Maximum Density Times Diversity", maximum_dtd_selection_strategy_generator),
    ("CompStrat 1 - NRATMin", 
     gen_case_profile_ss_generator(
        partial(GenericCompetenceMeasure, 
                Total(SplitterHider(lambda s, *args, **kwargs: comp_sum(s.dn, 'a', 'r')))), 
        op=SingleCompetenceSelectionStrategy.take_minimum)),
        
    ("CompStrat 2 - NCADMin", 
     gen_case_profile_ss_generator(ExampleCoverageOnlyCompetenceMeasure, 
                                  op=SingleCompetenceSelectionStrategy.take_minimum)),
    
    ("CompStrat 3 - NCATMaxDMin", 
     gen_case_profile_ss_generator2(SizeDeviationCombo)),
                              
                                       ]

named_experiment_variations = create_named_experiment_variations(named_selection_strategy_generators)

experiment = create_experiment(stopping_condition_generator, named_experiment_variations)