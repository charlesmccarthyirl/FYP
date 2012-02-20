'''
Created on Feb 5, 2012

@author: charles
'''

import os
from subprocess import call
from glob import glob
import shutil
import logging
from pprint import pprint
from itertools import chain
from utils import maybe_make_dirs

def my_call(args):
    pprint(" ".join(args))
    call(args)

STORAGE_DIR = "~/FYP/data_dir/"
REPORT_DIR = "~/FYP/experiment_outputs/"

non_textual_dir_name = "non_textual"
textual_dir_name = "textual"

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
    
    STORAGE_DIR = os.path.expanduser(STORAGE_DIR)
    REPORT_DIR = os.path.expanduser(REPORT_DIR)
    
    maybe_make_dirs(STORAGE_DIR)
    maybe_make_dirs(REPORT_DIR)
    
    runs = [
            (non_textual_dir_name, "DataSets"),
            (textual_dir_name, "TextualDataSets")
            ]
    
    for (cat_name, dsfn) in runs:
        d = os.path.join(STORAGE_DIR, cat_name)
        maybe_make_dirs(d)
        logging.info("Beginning experiment execution on %s" % dsfn)
        my_call(["python", "-O", "test.py", "experiment1", dsfn, d])
        for fn in glob(os.path.join(d, "*.pdf")):
            shutil.copyfile(fn, os.path.join(REPORT_DIR, os.path.basename(fn)))
        shutil.copyfile(os.path.join(d, "summary.csv"),
                        os.path.join(REPORT_DIR, cat_name + "_summary.csv"))
        
        # TODO: Should really check that it has all the data stats first.
        data_stat_fn = os.path.join(REPORT_DIR, cat_name + "_data_stats.csv")
        if not os.path.exists(data_stat_fn):
            logging.info("Generating data stats")
            args = ["python", "-O", "gen_data_stats.py", '--cite', dsfn,
                    data_stat_fn]
            my_call(args)
    
    dsn = 'WinXwin'
    timings_fn = os.path.join(STORAGE_DIR, dsn + ".csv")
    if not os.path.exists(timings_fn):
        logging.info("Generating timings for %s" % dsn)
        my_call(["python", "-O", "CaseProfilingTimings.py", dsn,
             timings_fn])
        
    logging.info("Plotting timings")
    for (name, columns) in [('incr_only', [1]), ('all', [1, 2, 3])]:
        timings_pdf_fn = os.path.join(REPORT_DIR, dsn + "_timings_" + name + ".pdf")
        my_call(["python", "-O", "plotter.py", timings_fn, timings_pdf_fn] 
                +list(chain(*[("-c", str(c)) for c in columns])) + ["--y_title",  "Time to build RCDL profiles (/s)", "--name", dsn])
    
    logging.info("Generating selection graphs")
    dsn = 'zoo'
    rsdir = os.path.join(REPORT_DIR, 'selection_graphs')
    maybe_make_dirs(rsdir)
    r_dir = os.path.join(STORAGE_DIR, non_textual_dir_name, dsn, 'raw_results')
    logging.info("Generating selection graphs")
    my_call(["python", "-O", "gen_selection_graphs.py", dsn, r_dir, rsdir, '--experiment experiment1']) 
        