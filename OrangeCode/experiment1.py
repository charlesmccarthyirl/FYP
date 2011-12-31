from SelectionStrategyEvaluator import *
from BreadAndButter import *
from CaseProfiling import *
from SelectionStrategy import *
from functools import partial

stopping_condition_generator = lambda data, *args, **kwargs: PercentageBasedStoppingCriteria(0.1, data, 0)
random_selection_strategy_generator = lambda *args, **kwargs: RandomSelectionStrategy(random_seed=RANDOM_SEED)

competence_measure_generator = ClassifierBasedCompetenceMeasure
margin_sampling_measure_generator = ClassifierBasedMarginSamplingMeasure
case_profile_measure_generator = CaseProfileBasedCompetenceMeasure

def case_profile_selection_strategy_generator(*args, **kwargs): 
    case_profile_builder=CaseProfileBuilder(k, *args, **kwargs)
    return SingleCompetenceSelectionStrategy(
        case_profile_measure_generator, 
        SingleCompetenceSelectionStrategy.take_maximum, 
        *args,
        case_profile_builder=case_profile_builder,
        on_selection_action=lambda example: case_profile_builder.put(example),
        **kwargs)


classifier_output_selection_strategy_generator = partial(SingleCompetenceSelectionStrategy,  
                                                         competence_measure_generator, 
                                                         SingleCompetenceSelectionStrategy.take_minimum)

margin_sampling_selection_strategy_generator = partial(SingleCompetenceSelectionStrategy,
                                                        margin_sampling_measure_generator, 
                                                        SingleCompetenceSelectionStrategy.take_minimum)

maximum_diversity_selection_strategy_generator = partial(SingleCompetenceSelectionStrategy,
                                                            DiversityMeasure, 
                                                            SingleCompetenceSelectionStrategy.take_maximum)

maximum_density_selection_strategy_generator = partial(SingleCompetenceSelectionStrategy,
                                                            DensityMeasure, 
                                                            SingleCompetenceSelectionStrategy.take_maximum)

maximum_dtd_selection_strategy_generator = partial(SingleCompetenceSelectionStrategy,
                                                            DensityTimesDiversityMeasure, 
                                                            SingleCompetenceSelectionStrategy.take_maximum)

maximum_dpd_selection_strategy_generator = partial(SingleCompetenceSelectionStrategy,
                                                            DensityPlusDiversityMeasure, 
                                                            SingleCompetenceSelectionStrategy.take_maximum)


named_selection_strategy_generators = {
                                       "Random Selection": random_selection_strategy_generator,
                                       "Least Confident Selection": classifier_output_selection_strategy_generator,
                                       "Margin Sampling": margin_sampling_selection_strategy_generator,
                                       "Competence Based Selection": case_profile_selection_strategy_generator,
                                       "Maximum Density Sampling": maximum_density_selection_strategy_generator,
                                       "Maximum Diversity Sampling": maximum_diversity_selection_strategy_generator,
                                       "Maximum Density*Diversity": maximum_dtd_selection_strategy_generator,
                                       "Maximum Density+Diversity": maximum_dpd_selection_strategy_generator,
                                       }

named_experiment_variations_generator = create_named_experiment_variations_generator(named_selection_strategy_generators)

experiment = create_experiment(stopping_condition_generator, named_experiment_variations_generator)