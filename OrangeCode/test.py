import orange, orngStat, orngTest
from Eval import *
from pyx import *

def main():
    oracle_generator = lambda *args, **kwargs: Oracle(orange.Example.get_class)
    stopping_condition_generator = lambda *args, **kwargs: BudgetBasedStoppingCriteria(10)
    selection_strategy_generator = lambda *args, **kwargs: RandomSelectionStrategy()
    classifier_generator = lambda training_data, *args, **kwargs: orange.kNNLearner(training_data, k=5)
    
    data = orange.ExampleTable(r"iris.arff")
    rndind = orange.MakeRandomIndices2(data, p0=.8)
    
    test = data.select(rndind, 0)
    unlabelled = data.select(rndind, 0, negate=1)
    
    evaluator = SelectionStrategyEvaluator(**locals())
    points = evaluator.generate_results(test, unlabelled)
    print points.AULC()
    #print("Accuracy for %3d = %f" % (case_base_size, classification_accuracy))
    
    xs = [result.case_base_size for result in points]
    ys = [result.classification_accuracy for result in points]


    g = graph.graphxy(width=max(xs),
                      x=graph.axis.linear(title="Case Base Size"),
                      y=graph.axis.linear(title="Classification Accuracy"),
                      key=graph.key.key(pos="br", dist=0.1))
    
    # either provide lists of the individual coordinates
    g.plot([graph.data.values(x=xs, y=ys, title="Random Sampling")], [graph.style.line([color.gradient.Rainbow])])
    # or provide one list containing the whole points
    #g.plot(graph.data.points(zip(range(10), range(10)), x=1, y=2))
    #g.writeEPSfile("points")
    g.writePDFfile("points")
    
if __name__ == "__main__":
    main()

