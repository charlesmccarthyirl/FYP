import os
from operator import gt, lt
from StringIO import StringIO

class MyStringIO(StringIO):
    def __init__(self, do_delete_on_close=True, *args, **kwargs):
        self.do_delete_on_close = do_delete_on_close
        StringIO.__init__(self, *args, **kwargs)
    
    def close(self):
#        self.closed = True
        if self.do_delete_on_close:
            StringIO.close(self)

def stream_getter(filename, none_on_exists=False):
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
        
    if os.path.exists(filename) and none_on_exists:
        return None
    
    return open(filename, 'wb')

def op_multiple(the_list, op, key=None):
    ms = []
    m_key = None
    
    for thing in the_list:
        thing_key = thing if key is None else key(thing)
        if m_key is None or op(thing_key, m_key):
            m_key = thing_key
            ms = [thing]
        elif thing_key == m_key:
            ms.append(thing)
    return ms

def max_multiple(the_list, key=None):
    '''
    Finds the list of maximum elements in a given list.
    @param the_list: The list to find the maximum in.
    @param key: The key function to compare on.
    
    >>> m_list = [('a', 3), ('b', 2), ('c', 3)]
    >>> max_multiple(m_list, key=lambda x: x[1])
    [('a', 3), ('c', 3)]
    >>> m_list
    [('a', 3), ('b', 2), ('c', 3)]
    '''
    return op_multiple(the_list, gt, key)

def min_multiple(the_list, key=None):
    return op_multiple(the_list, lt, key)

def average(iterable):
    total = 0
    length = 0
    for i in iterable:
        total += i
        length += 1
        
    return total / length

# Taken from http://www.peterbe.com/plog/uniqifiers-benchmark
def uniqueify(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]

""" 
Calculate mean and standard deviation of data x[]: 
    mean = {\sum_i x_i \over n} 
    std = sqrt(\sum_i (x_i - mean)^2 \over n-1) 
Taken from http://www.physics.rutgers.edu/~masud/computing/WPark_recipes_in_python.html
""" 
def meanstdv(x):
    from math import sqrt 
    n, mean, std = len(x), 0, 0 
    for a in x: 
        mean = mean + a 
    mean = mean / float(n) 
    for a in x: 
        std = std + (a - mean)**2 
    std = sqrt(std / float(n-1)) 
    return mean, std

# http://stackoverflow.com/questions/390852/is-there-any-built-in-way-to-get-the-length-of-an-iterable-in-python
def count_iterable(i):
    return sum(1 for e in i)

def try_convert_to_num(cell):
    if cell.isdigit():
        return int(cell)
    try:
        return float(cell)
    except ValueError:
        return cell

# http://preshing.com/20110924/timing-your-code-using-pythons-with-statement
from monotonic import monotonic_time
class Timer:
    def __enter__(self):
        self.start = monotonic_time()
        return self

    def __exit__(self, *args):
        self.end = monotonic_time()
        self.interval = self.end - self.start

def maybe_make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
