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

def maybe_make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
        logging.info("Generating data stats")
        args = ["python", "-O", "gen_data_stats.py", dsfn,
             os.path.join(REPORT_DIR, cat_name + "_data_stats.csv")]
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
                +list(chain((("-c", str(c)) for c in columns))) + ["--y_title",  "Time to build RCDL profiles (/s)", "--name", dsn])
    
    logging.info("Generating selection graphs")
    dsn = 'zoo'
    ssdir = os.path.join(STORAGE_DIR, 'selection_graphs')
    rsdir = os.path.join(STORAGE_DIR, 'selection_graphs')
    maybe_make_dirs(ssdir)
    maybe_make_dirs(rsdir)
    for fn in glob(os.path.join(STORAGE_DIR, dsn, "*.tar.gz")):
        vn = os.path.splitext(os.path.splitext(os.path.basename(fn))[0])[0]
        out_file = os.path.join(ssdir, vn + ".pdf")
        if not os.path.exists(out_file):
            logging.info("Generating selection graph for variation %s" % vn)
            my_call(["python", "-O", "gen_selection_graphs.py", dsn, fn, out_file]) 
        shutil.copyfile(out_file,
                        os.path.join(rsdir, os.path.basename(out_file)))
        