import orange, orngStat, orngTest
import matplotlib.pyplot as plt
from Eval import *

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
    plt.plot(xs, ys)
    plt.savefig("test.png")
    
if __name__ == "__main__":
    main()

