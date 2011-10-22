'''
Created on 22 Oct 2011

@author: Charles Mc
'''

def index(self, needle):
    i = 0
    for element in self:
        if element == needle:
            return i
        i += i
    
    return -1