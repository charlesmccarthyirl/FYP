import orange, orngStat, orngTest
from Eval import *
from BreadAndButter import *

stopping_condition_generator = lambda data, *args, **kwargs: PercentageBasedStoppingCriteria(0.1, data, 0)
random_selection_strategy_generator = lambda *args, **kwargs: RandomSelectionStrategy(random_seed=RANDOM_SEED)
competence_measure_generator = lambda *args, **kwargs: ClassifierBasedCompetenceMeasure(classifier_generator, 
                                                                                        *args, 
                                                                                        **kwargs)

competence_selector = lambda measure1, measure2: measure1 < measure2 
classifier_output_selection_strategy_generator = lambda *args, **kwargs: SingleCompetenceSelectionStrategy(
                                                                                        competence_measure_generator, 
                                                                                        competence_selector, 
                                                                                        *args,
                                                                                        **kwargs)
named_selection_strategy_generators = {
                                       "Random Selection": random_selection_strategy_generator,
                                       "Uncertainty Sampling": classifier_output_selection_strategy_generator
                                       }

named_experiment_variations = create_named_experiment_variations(named_selection_strategy_generators)

experiment = create_experiment(stopping_condition_generator, named_experiment_variations)