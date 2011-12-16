import orange, orngStat, orngTest
from Eval import *
from BreadAndButter import *
from CaseProfiling import *
from SelectionStrategy import *

stopping_condition_generator = lambda data, *args, **kwargs: PercentageBasedStoppingCriteria(0.1, data, 0)
random_selection_strategy_generator = lambda *args, **kwargs: RandomSelectionStrategy(random_seed=RANDOM_SEED)

competence_measure_generator = lambda *args, **kwargs: ClassifierBasedCompetenceMeasure(probability_generator, 
                                                                                        *args, 
                                                                                        **kwargs)

margin_sampling_measure_generator = lambda *args, **kwargs: ClassifierBasedMarginSamplingMeasure(probability_generator, 
                                                                                                 *args, 
                                                                                                 **kwargs)

case_profile_measure_generator = lambda *args, **kwargs: CaseProfileBasedCompetenceMeasure(probability_generator, 
                                                                                                 *args, 
                                                                                                 **kwargs)

def case_profile_selection_strategy_generator(distance_constructor, *args, **kwargs): 
    case_profile_builder=CaseProfileBuilder(classifier_generator, distance_constructor, k)
    return SingleCompetenceSelectionStrategy(
        case_profile_measure_generator, 
        SingleCompetenceSelectionStrategy.take_maximum, 
        *args,
        case_profile_builder=case_profile_builder,
        on_selection_action=lambda example: case_profile_builder.put(example),
        distance_constructor=distance_constructor,
        **kwargs)


classifier_output_selection_strategy_generator = lambda *args, **kwargs: SingleCompetenceSelectionStrategy(
                                                                                        competence_measure_generator, 
                                                                                        SingleCompetenceSelectionStrategy.take_minimum, 
                                                                                        *args,
                                                                                        **kwargs)

margin_sampling_selection_strategy_generator = lambda *args, **kwargs: SingleCompetenceSelectionStrategy(
                                                                                        margin_sampling_measure_generator, 
                                                                                        SingleCompetenceSelectionStrategy.take_minimum, 
                                                                                        *args,
                                                                                        **kwargs)

maximum_diversity_selection_strategy_generator = lambda *args, **kwargs: SingleCompetenceSelectionStrategy(
                                                                                        DiversityMeasure, 
                                                                                        SingleCompetenceSelectionStrategy.take_maximum, 
                                                                                        *args,
                                                                                        **kwargs)

named_selection_strategy_generators = {
                                       "Random Selection": random_selection_strategy_generator,
                                       "Least Confident Selection": classifier_output_selection_strategy_generator,
                                       "Margin Sampling": margin_sampling_selection_strategy_generator,
#                                       "Maximum Diversity Sampling": maximum_diversity_selection_strategy_generator,
#                                       "Competence Based Selection": case_profile_selection_strategy_generator
                                       }

named_experiment_variations_generator = create_named_experiment_variations_generator(named_selection_strategy_generators)

experiment = create_experiment(stopping_condition_generator, named_experiment_variations_generator)