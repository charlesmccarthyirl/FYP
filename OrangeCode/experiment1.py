import orange, orngStat, orngTest
from Eval import *
from BreadAndButter import *

stopping_condition_generator = lambda data, *args, **kwargs: PercentageBasedStoppingCriteria(0.1, data, 0)
random_selection_strategy_generator = lambda *args, **kwargs: RandomSelectionStrategy(random_seed=RANDOM_SEED)
competence_measure_generator = lambda *args, **kwargs: ClassifierBasedCompetenceMeasure(classifier_generator, 
                                                                                        *args, 
                                                                                        **kwargs)

margin_sampling_measure_generator = lambda *args, **kwargs: ClassifierBasedMarginSamplingMeasure(classifier_generator, 
                                                                                                 *args, 
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

named_selection_strategy_generators = {
                                       "Random Selection": random_selection_strategy_generator,
                                       "Uncertainty Sampling": classifier_output_selection_strategy_generator,
                                       "Margin Sampling": margin_sampling_selection_strategy_generator
                                       }

named_experiment_variations = create_named_experiment_variations(named_selection_strategy_generators)

experiment = create_experiment(stopping_condition_generator, named_experiment_variations)