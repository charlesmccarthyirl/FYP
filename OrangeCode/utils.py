import os

def stream_getter(filename):
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    return open(filename, 'wb')

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
    
    ms = []
    m_key = None
    
    for thing in the_list:
        thing_key = thing if key is None else key(thing)
        if m_key is None or thing_key > m_key:
            m_key = thing_key
            ms = [thing]
        elif thing_key == m_key:
            ms.append(thing)
    return ms