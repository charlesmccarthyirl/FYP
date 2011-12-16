# Not mine - adapted from sklearn.cross_validation

from math import ceil


class KFold(object):
    """K-Folds cross validation iterator

    Provides train/test indices to split data in train test sets. Split
    dataset into k consecutive folds (without shuffling).

    Each fold is then used a validation set once while the k - 1 remaining
    fold form the training set.

    Parameters
    ----------
    n: int
        Total number of elements

    k: int
        Number of folds

    Notes
    -----
    All the folds have size trunc(n_samples / n_folds), the last one has the
    complementary.
    """

    def __init__(self, n, k):
        assert k > 0, ValueError('Cannot have number of folds k below 1.')
        assert k <= n, ValueError('Cannot have number of folds k=%d, '
                                  'greater than the number '
                                  'of samples: %d.' % (k, n))
        self.n = n
        self.k = k

    def __iter__(self):
        n = int(self.n)
        k = int(self.k)
        j = int(ceil(n / k))

        for i in xrange(k):
            test_index = [False]*n
            start = i*j
            end  = (i+1)*j if i < k - 1 else n
            for x in xrange(start, end):
                test_index[x] = True
            train_index = (not x for x in test_index)

            yield train_index, test_index

    def __repr__(self):
        return '%s.%s(n=%i, k=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.k,
        )

    def __len__(self):
        return self.k