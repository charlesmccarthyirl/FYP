'''
Created on Feb 5, 2012

@author: charles
'''

import os
from subprocess import call, Popen, STDOUT
import logging
from pprint import pprint
from utils import maybe_make_dirs, my_import, checkLogFor
from multiprocessing import cpu_count
import optparse
import time

STORAGE_DIR = "~/FYP/data_dir/"
CODE_DIR = "~/FYP/FYP/OrangeCode/"
LOGS_DIR = os.path.expanduser(os.path.join(STORAGE_DIR, 'logs'))

def get_full_logfn(logfn):
    return os.path.join(LOGS_DIR, logfn)

def my_call(args, block=True, logfn=None, shell=False):
    func = Popen
    pretty_args = " ".join(args) if not isinstance(args, str) else args
    pprint(pretty_args)
    extra_kwargs = {}
    if logfn:
        logs_dir = os.path.dirname(logfn)
        maybe_make_dirs(logs_dir)
        my_file = open(logfn, 'ab')
        extra_kwargs = dict(stderr=STDOUT, stdout=my_file)
    
    if shell:
        extra_kwargs['shell'] = shell
    
    p = func(args, **extra_kwargs)
    if block:
        p.wait()
    return p

def ssh_call(host, head, block=True, extra=""):
    return my_call("ssh -o ConnectTimeout=1 '" +  host + 
                            "' 'nohup python -O " + '"'
                            + os.path.join(CODE_DIR, os.path.basename(__file__)) 
                            + '" "' +host+ '" "' + head + '" '  + extra 
                            + " > " + '"' + os.path.join(LOGS_DIR, "%s.ssh.log" % host) + '"' + " 2>&1'", 
                   block=block, shell=True)

def call_on_upto(n, head, extra=None, early_halt=None):
    '''
    Call this script on up to n hosts, returning the number of hosts <= n that it was called on.
    '''
    called_on = 0
    for h in hosts:
        if (early_halt is not None and early_halt()) or called_on >= n:
            break
        p = ssh_call(h, head, block=True, extra=extra)
        if p.returncode == 0:
            called_on += 1
            logging.info("Success for %s. Now have %d" % (h, called_on))
        else:
            logging.info("Failed for %s" % h)
    return called_on

if __name__ == '__main__':
    parser = optparse.OptionParser("usage: %prog [options]")
    parser.add_option('--remote', dest='remote', default=False, action='store_true')
    parser.add_option('--hosts', dest='hosts', default="", action='store')
    parser.add_option('--upto', dest='upto', type="int", default=10, action='store')
    parser.add_option('--password', help='Password to use when performing distributed computation', dest='password',
                      default="changeme", action='store')
    parser.add_option('--nohead', dest='no_head', default=False, action='store_true')
    (options, args) = parser.parse_args()
    
    if options.hosts:
        hosts = my_import(options.hosts).hosts
    
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)

    CODE_DIR = os.path.expanduser(CODE_DIR)
    
    #Host will be head if nothing else provided . . .
    host = args[0] if len(args) > 0 else None
    head = args[-1] if len(args) > 0 else None
    
    os.chdir(CODE_DIR)
    
    if not options.remote:
        cat_name, dsfn = 'all', "AllDataSets"
        d = os.path.join(STORAGE_DIR, cat_name)
        logging.info("Beginning experiment execution on %s" % dsfn)
        logfn = get_full_logfn("%s.master.log" % host)
        
        if options.no_head:
            early_halt = lambda: False
        else:
            p = my_call(["nohup", "python", "-O", "test.py", "--nocolour", "--latexencode", 
                    "experiment1", dsfn, d, "--genonly", "--multi", "--password", options.password], block=False, logfn=logfn)
            early_halt = lambda: p.poll() is not None
            while not (checkLogFor(logfn, "Beginning mincemeat server")
                       or early_halt()): # In case there was some error in the test.py, no point waiting forever if it's dead.  
                time.sleep(5)
            
        call_on_upto(options.upto, head, extra="--remote", early_halt=early_halt)
        
        p.wait()
    elif options.remote:
        for cn in xrange(cpu_count()):
            logfn = get_full_logfn("%s.%d.log" % (host, cn))
            my_call(["python", "-O", "mincemeat.py", "-p", "changeme", head, "--verbose"], logfn=logfn, block=False)
        
        
        