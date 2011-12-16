'''
Created on Nov 26, 2011

@author: charles
'''
import unittest
import Knn


class Test(unittest.TestCase):
    def setUp(self):
        data = [
                (1, 'blue'),
                (2, 'blue'),
                (5, 'red' ),
                (6, 'red' )
                ]
        dist_meas = lambda a, b: abs(a[0] - b[0])
        true_oracle = lambda b: b[1]
        possible_classes = ['red', 'blue']
        
        self.knn = Knn.KNN(data, 2, dist_meas, true_oracle, possible_classes)

    def test_get_probabilities(self):
        probs = self.knn.get_probabilities((1, ''))
        self.assertItemsEqual([('blue', 1.0)], probs)
        
    def test_find_nearest(self):
        nearest = self.knn.find_nearest((1, ''))
        self.assertItemsEqual([(1, 'blue'), (2, 'blue')], nearest)
    
    def test_classify(self):
        classification = self.knn.classifiy((1, ''))
        self.assertEqual("blue", classification)
        
    def test_get_probabilities1(self):
        probs = self.knn.get_probabilities((3.5, ''))
        self.assertItemsEqual([('blue', 0.5), ('red', 0.5) ], probs)
        
    def test_find_nearest1(self):
        nearest = self.knn.find_nearest((3.5, ''))
        self.assertItemsEqual([(5, 'red'), (2, 'blue')], nearest)
    
    def test_classify1(self):
        self.knn.classification_tie_breaker = lambda classes: list(classes)[1]
        classification = self.knn.classifiy((3.5, ''))
        self.assertEqual("red", classification)
    
    def test_classify2(self):
        classification = self.knn.classifiy((3.5, ''))
        self.assertEqual("blue", classification)
        
    def test_find_nearest2(self):
        self.knn.k = 1
        nearest = self.knn.find_nearest((1.5, ''))
        self.assertItemsEqual([(1, 'blue')], nearest)
        
    def test_find_nearest3(self):
        self.knn.k = 1
        def stuff(insts):
            return list(insts)[1]
        self.knn.instance_tie_breaker = stuff
        nearest = self.knn.find_nearest((1.5, ''))
        self.assertItemsEqual([(2, 'blue')], nearest)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()