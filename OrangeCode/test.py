import orange, orngStat, orngTest
import random
import matplotlib.pyplot as plt
import orange_extensions

class Oracle:
    def __init__(self, classifyLambda):
        self.classify = classifyLambda

class Selection:
    def __init__(self, selection, index):
        self.selection = selection
        self.index = index
        
    def delete_from(self, list):
        index = self.index
        if index is None:
            index = orange_extensions.index(self.selection)
        
        del(list[index])
            

class SelectionStrategy:
    def __init__(self, *args, **kwargs):
        pass
    
    def select(self, collection):
        '''
        Given a collection of items to select from, selects an item and returns it (along with its index).
        @param collection: The indexable collection to select from.
        '''
        pass

class RandomSelectionStrategy(SelectionStrategy):
    def __init__(self, *args, **kwargs):
        SelectionStrategy.__init__(self, *args, **kwargs)
        
        # Other stuff might have screwed with the state of random.random - so I'm taking my own.
        self._random = random.Random()
        self._random.seed()
        
    
    def select(self, collection):   
        '''
        Given a collection of items to select from, selects an item and returns it (along with its index).
        @param collection: The indexable collection to select from.
        '''
        



def main():
    data = orange.ExampleTable(r"C:\Users\Charles Mc\Dropbox\College\4th Year\FYP\Source\cmmc1_fyp\iris.arff")
    rndind = orange.MakeRandomIndices2(data, p0=0.5)
    
    unlabelled = data.select(rndind, 0)
    test = data.select(rndind, 1)
    
    case_base = orange.ExampleTable(unlabelled.domain)

    random.seed()

    def select(exampleTable):
        upTo = len(exampleTable) - 1
        r = random.randint(0, upTo)
        return r
    
    remaining = 10
    def is_expended():
        return remaining <= 0
    
    def classify(example):
        return example.get_class()
    
    xs = []
    ys = []
    
    while not is_expended():
        selectedIndex = select(unlabelled)
        
        selectedExample = orange.Example(unlabelled[selectedIndex])
        del(unlabelled[selectedIndex])
        selectedExample.set_class(classify(selectedExample))
        
        case_base.append(selectedExample)
        
        classifier = orange.kNNLearner(case_base, k=1)
        
        testResults = orngTest.testOnData([classifier], test)
        CAs = orngStat.CA(testResults)
        
        number_in_case_base = len(case_base)
        classification_accuracy = CAs[0]
        
        xs.append(number_in_case_base)
        ys.append(classification_accuracy)
        
        print("Accuracy for %3d = %f" % (number_in_case_base ,classification_accuracy))
        
        remaining -= 1
    
    plt.plot(xs, ys)
    plt.savefig("test.png")
    
if __name__ == "__main__":
    main()

