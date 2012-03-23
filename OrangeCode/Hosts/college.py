'''
Created on Mar 21, 2012

@author: charles
'''
bad_hosts = ["cs1-17-06","cs1-17-11","cs1-17-19", "csg24-32", "cs1-31-16"]
hosts = ["csg21-%02d" % i for i in range(2, 48)] #g21
hosts += ["cs1-17-%02d" % i for i in range(3, 22)] #1-17 
hosts += ["csg24-%02d" % i for i in range(2, 48)] + ["cs-1-09-%02d" % i for i in range(1, 25)] + ["csg20-26", "csg20-37", "cs1-31-16"] #g24
hosts += ["cs1-12-11"]

for h in bad_hosts:
    hosts.remove(h)