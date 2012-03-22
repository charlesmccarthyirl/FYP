'''
Created on Feb 5, 2012

@author: charles
'''

import os
from subprocess import call, Popen, STDOUT
import logging
from pprint import pprint
from utils import maybe_make_dirs, my_import
from multiprocessing import cpu_count
import optparse
import time


STORAGE_DIR = "~/FYP/data_dir/"
CODE_DIR = "~/FYP/FYP/OrangeCode/"
LOGS_DIR = os.path.expanduser(os.path.join(STORAGE_DIR, 'logs'))

def my_call(args, block=True, logfn=None, shell=False):
    func = Popen
    pretty_args = " ".join(args) if not isinstance(args, str) else args
    pprint(pretty_args)
    extra_kwargs = {}
    if logfn:
        logs_dir = os.path.expanduser(LOGS_DIR)
        maybe_make_dirs(logs_dir)
        logfn = os.path.join(logs_dir, logfn)
        my_file = open(logfn, 'ab')
        extra_kwargs = dict(stderr=STDOUT, stdout=my_file)
    
    if shell:
        extra_kwargs['shell'] = shell
    
    p = func(args, **extra_kwargs)
    if block:
        p.wait()
    return p

non_textual_dir_name = "non_textual"
textual_dir_name = "textual"

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
    (options, args) = parser.parse_args()
    
    if options.hosts:
        hosts = my_import(options.hosts).hosts
    
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)

    CODE_DIR = os.path.expanduser(CODE_DIR)
    
    runs = [
            (non_textual_dir_name, "DataSets"),
            (textual_dir_name, "TextualDataSets")
            ]

    #Host will be head if nothing else provided . . .
    host = args[0] if len(args) > 0 else None
    head = args[-1] if len(args) > 0 else None
    
    os.chdir(CODE_DIR)
    
    if not options.remote:
        for (cat_name, dsfn) in runs:
            d = os.path.join(STORAGE_DIR, cat_name)
            logging.info("Beginning experiment execution on %s" % dsfn)
            logfn = "%s.master.log" % host
            p = my_call(["nohup", "python", "-O", "test.py", "--nocolour", "--latexencode", 
                    "experiment1", dsfn, d, "--genonly", "--multi"], block=False, logfn=logfn)
            time.sleep(5)
            call_on_upto(options.upto, head, extra="--remote", early_halt=lambda: p.poll() is not None)
            p.wait()
    elif options.remote:
        for cn in xrange(cpu_count()):
            logfn = "%s.%d.log" % (host, cn)
            #TODO: make password configurable. Will need to put arg in test.py aswell.
            my_call(["python", "-O", "mincemeat.py", "-p" "changeme", head, "--verbose"], logfn=logfn, block=False)
        
        
        