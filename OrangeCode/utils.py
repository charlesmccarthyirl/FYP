import os

def stream_getter(filename):
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    return open(filename, 'wb')