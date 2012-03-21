'''
Created on Feb 5, 2012

@author: charles
'''

import os
from subprocess import call, Popen, STDOUT
from glob import glob
import shutil
import logging
from pprint import pprint
from itertools import chain
from utils import maybe_make_dirs
import optparse

def my_call(args, block=True, logfn=None):
    fn = call if block else Popen
    pprint(" ".join(args))
    extra_kwargs = {}
    if logfn:
        logfn = os.path.expanduser(logfn)
        dn = os.path.dirname(logfn)
        maybe_make_dirs(dn)
        print logfn
        my_file = open(logfn, 'ab')
        extra_kwargs = dict(stderr=STDOUT, stdout=my_file)

    return fn(args, **extra_kwargs)

STORAGE_DIR = "~/FYP/data_dir/"
REPORT_DIR = "~/FYP/experiment_outputs/"
CODE_DIR = "~/FYP/FYP/OrangeCode/"

non_textual_dir_name = "non_textual"
textual_dir_name = "textual"

username = "cmmc1"
hosts = ["cs1-12-11"]
num_procs_per_host = 2

def logfngetter(host, proc_num):
    return os.path.join(STORAGE_DIR, 'logs', "%s.%d.log" % (host, proc_num)) 

if __name__ == '__main__':
    parser = optparse.OptionParser("usage: %prog [options]")
    parser.add_option('--head', dest='head', default=False, action='store_true')
    parser.add_option('--remote', dest='remote', default=False, action='store_true')
    (options, args) = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
    
    
    REPORT_DIR = os.path.expanduser(REPORT_DIR)
    CODE_DIR = os.path.expanduser(CODE_DIR)
    
    maybe_make_dirs(REPORT_DIR)

    runs = [
            (non_textual_dir_name, "DataSets"),
            (textual_dir_name, "TextualDataSets")
            ]

    logfn = args[0] if len(args) > 0 else None
    head = args[-1] if len(args) > 0 else None
    
    os.chdir(CODE_DIR)
    
    #TODO: Define a call-on style thing - call on up to n hosts, m procs per host) 
    
    def ssh_call(host, pn, block=True, extra=None):
        extra = extra and extra(host) or ""
        return my_call(["ssh", "%s@%s" %(username, host),
                                "'nohup python -O " + '"'
                                + os.path.join(CODE_DIR, os.path.basename(__file__)) 
                                + '" "' +  logfngetter(host, pn) + '" ' + extra + " > /dev/null 2>&1'"], block=block)
    
    if options.head:
        for (cat_name, dsfn) in runs:
            d = os.path.join(STORAGE_DIR, cat_name)
            logging.info("Beginning experiment execution on %s" % dsfn)
            if not logfn:
                logfn = logfngetter(head, 0)
            p = my_call(["nohup", "python", "-O", "test.py", "--nocolour", "--latexencode", 
                    "experiment1", dsfn, d, "--genonly", "--multi"], block=False, logfn=logfn)
            for h in hosts:
                for pn in xrange(num_procs_per_host):
                    ssh_call(h, pn, block=False, extra="--remote")
            p.wait()
    elif options.remote:
        my_call(["nohup", "python", "-O", "mincemeat.py", "-p" "changeme", head], logfn=logfn)
    else:
        ssh_call(hosts[0], 10, extra=lambda h: h)
        
        
        