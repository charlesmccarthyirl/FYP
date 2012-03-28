'''
Created on Mar 21, 2012

@author: charles
'''
bad_hosts = ["cs1-17-06","cs1-17-11","cs1-17-19", "csg24-32", "cs1-31-16", "csg22-06", 
             "csg22-20", "csg26-02", "cs1-09-09", "csg22-19", "cs1-09-31", "cs1-09-32"]
bad_hosts += ["mmlab%02d" % i for i in range(3, 26)] 
bad_hosts += ["cs1-31-%02d" % i for i in range(2, 16)]

hosts = ["csg21-%02d" % i for i in range(2, 48)] #g21
hosts += ["cs1-17-%02d" % i for i in range(3, 22)] #1-17 
hosts += ["csg24-%02d" % i for i in range(2, 48)] + ["cs1-09-%02d" % i for i in range(1, 25)] + ["csg20-26", "csg20-37", "cs1-31-16"] #g24
hosts += ["cs1-12-11"] #1-12
hosts += ["cs1-11-%02d" % i for i in range(1, 18)] + ["cs1-12-04"] #1-11
hosts += (["cs1-31-%02d" % i for i in range(2, 14)] 
          + ["csg22-%02d" % i for i in range(2, 24)] 
          + ["csg20-%02d" % i for i in range(2, 48)]
          + ["cs1-31-%02d" % i for i in range(14, 16)]
          + ["cs1-17-%02d" % i for i in range(4, 06)]   ) #G20

hosts += ["csg26-%02d" % i for i in range(2, 38)] #G26 (#G25 unpingable)
hosts += ["csg19-%02d" % i for i in range(2, 22)] #G19
hosts += ["cs1-19-%02d" % i for i in range(2, 17)] #1-19
hosts += (["cs1-32-%02d" % i for i in range(2, 11)] 
          + ["mmlab%02d" % i for i in range(2, 26)] 
          + ["cs1-09-%02d" % i for i in range(31, 33)]   ) #1-09

for h in bad_hosts:
    hosts.remove(h)