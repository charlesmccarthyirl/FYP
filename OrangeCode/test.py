import orange, orngStat, orngTest
from Eval import *

def n_fold_cross_validation(data, n):
    rndind = orange.MakeRandomIndicesCV(data, folds=n)
    return ((data.select(rndind, fold, negate=1), 
             data.select(rndind, fold)) 
            for fold in xrange(n))
    
def main():
    oracle_generator = lambda *args, **kwargs: Oracle(orange.Example.get_class)
    stopping_condition_generator = lambda data, *args, **kwargs: PercentageBasedStoppingCriteria(0.1, data, 0)
    classifier_generator = lambda training_data, *args, **kwargs: orange.kNNLearner(training_data, k=5)
    
    random_selection_strategy_generator = lambda *args, **kwargs: RandomSelectionStrategy()
    
    competence_measure_generator = lambda *args, **kwargs: ClassifierBasedCompetenceMeasure(classifier_generator, 
                                                                                            *args, 
                                                                                            **kwargs)
    competence_selector = lambda measure1, measure2: measure1 < measure2 
    classifier_output_selection_strategy_generator = lambda *args, **kwargs: SingleCompetenceSelectionStrategy(competence_measure_generator, 
                                                                                             competence_selector, 
                                                                                             *args,
                                                                                             **kwargs)
    named_experiment_variations = {"Random Selection": ExperimentVariation(classifier_generator, random_selection_strategy_generator),
                                   "Uncertainty Sampling": ExperimentVariation(classifier_generator, classifier_output_selection_strategy_generator)}
    
    training_test_sets_extractor = lambda data: n_fold_cross_validation(data, 10)
    
    experiment = Experiment(oracle_generator, 
                            stopping_condition_generator, 
                            training_test_sets_extractor, 
                            named_experiment_variations)
    
    d = orange.ExampleTable(r"iris.arff")
    d.shuffle() # Could all be clustered together in the file. Some of my operations might 
                   # (and do . . .) go in order - so can skew the results *a lot*.

    results = experiment.execute_on(d)
    g = results.generate_graph("iris")
    #g.writeEPSfile("points")
    g.writePDFfile("points")
    
if __name__ == "__main__":
    main()
