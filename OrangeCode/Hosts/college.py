'''
Created on Mar 21, 2012

@author: charles
'''
bad_hosts = ["cs1-17-06","cs1-17-11","cs1-17-19"]
hosts = ["csg21-%02d" % i for i in range(2, 48)] + ["cs1-17-%02d" % i for i in range(3, 22)] #+ ["cs1-12-11"]

for h in bad_hosts:
    hosts.remove(h)