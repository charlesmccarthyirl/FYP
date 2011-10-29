import orange, orngStat, orngTest
from Eval import *
from pyx import *

def main():
    oracle_generator = lambda *args, **kwargs: Oracle(orange.Example.get_class)
    stopping_condition_generator = lambda *args, **kwargs: BudgetBasedStoppingCriteria(10)
    classifier_generator = lambda training_data, *args, **kwargs: orange.kNNLearner(training_data, k=5)
    #selection_strategy_generator = lambda *args, **kwargs: RandomSelectionStrategy()
    
    competence_measure_generator = lambda *args, **kwargs: ClassifierBasedCompetenceMeasure(classifier_generator, 
                                                                                            *args, 
                                                                                            **kwargs)
    competence_selector = lambda measure1, measure2: measure1 < measure2 
    selection_strategy_generator = lambda *args, **kwargs: SingleCompetenceSelectionStrategy(competence_measure_generator, 
                                                                                             competence_selector, 
                                                                                             *args,
                                                                                             **kwargs)
    
    num_folds = 10
    
    data = orange.ExampleTable(r"iris.arff")
    data.shuffle() # Could all be clustered together in the file. Some of my operations might 
                   # (and do . . .) go in order - so can skew the results *a lot*.
    rndind = orange.MakeRandomIndicesCV(data, folds=num_folds)
    
    data_test_iterable = ((data.select(rndind, fold, negate=1), 
                                     data.select(rndind, fold)) for fold in xrange(num_folds))

    evaluator = SelectionStrategyEvaluator(**locals())
    points = evaluator.generate_results_from_many(data_test_iterable)
    print points.AULC()
    #print("Accuracy for %3d = %f" % (case_base_size, classification_accuracy))
    
    xs = [result.case_base_size for result in points]
    ys = [result.classification_accuracy for result in points]

    max_x=max(xs)
    max_y=1.0
    g = graph.graphxy(width=10,
                      height=10, # Want a square graph . . .
                      x=graph.axis.linear(title="Case Base Size", min=0, max=max_x), #This might seem redundant - but pyx doesn't handle non-varying y well. So specifying the min and max avoids that piece of pyx code.
                      y=graph.axis.linear(title="Classification Accuracy", min=0, max=max_y),
                      key=graph.key.key(pos="br", dist=0.1))
    
    # either provide lists of the individual coordinates
    g.plot([graph.data.values(x=xs, y=ys, title="Uncertainty Sampling")], [graph.style.line([color.gradient.Rainbow])])
    # or provide one list containing the whole points
    #g.plot(graph.data.points(zip(range(10), range(10)), x=1, y=2))
    #g.writeEPSfile("points")
    g.writePDFfile("points")
    
if __name__ == "__main__":
    main()
