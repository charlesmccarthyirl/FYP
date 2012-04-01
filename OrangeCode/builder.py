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
from utils import maybe_make_dirs, my_import

def my_call(args):
    print(" ".join(args))
    call(args)

STORAGE_DIR = os.path.expanduser("~/FYP/data_dir/")
REPORT_DIR = os.path.expanduser("~/FYP/experiment_outputs/")
ALL_DIR = os.path.join(STORAGE_DIR, 'all')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
    
    
    maybe_make_dirs(STORAGE_DIR)
    maybe_make_dirs(REPORT_DIR)
    
    runs = [
            ('all_non_textual', "Datasets.non_textual", "Experiments.all_experiments", False, True),
            ('all_textual', "Datasets.textual", "Experiments.all_experiments", False, True),
            ('all_selected', "Datasets.selected", "Experiments.all_experiments", False, False),
            ('selected_baseline', 'Datasets.selected', "Experiments.baselines", True, False),
            ('selected_sparsity', 'Datasets.selected', "Experiments.baseline_sparsity", True, False)
            ]
    
    runs = [(cat_name, os.path.join(STORAGE_DIR, cat_name), dsfn, 
             expn, do_plots, gen_stats) 
            for (cat_name, dsfn, expn, do_plots, gen_stats) in runs]
    
    for (_, d, dsfn, _, _, _) in runs:
        named_data_sets = my_import(dsfn).named_data_sets
        ds_names = [nds[0] for nds in named_data_sets]
        shutil.rmtree(d, True)
        for dsn in ds_names:
            shutil.copytree(os.path.join(ALL_DIR, dsn), os.path.join(d, dsn))
    
    for (cat_name, d, dsfn, expn, do_plots, gen_stats) in runs:
        maybe_make_dirs(d)
        report_dir = os.path.join(REPORT_DIR, cat_name)
        maybe_make_dirs(report_dir)
        logging.info("Beginning experiment execution on %s" % dsfn)
        my_call(["python", "-O", "test.py", "--nocolour", "--docreatesummary", "--latexencode",
                 expn, dsfn, d] + (["--docreateplots", "-keyonlast"] if do_plots else []))
        for fn in glob(os.path.join(d, "*.pdf")):
            shutil.copyfile(fn, os.path.join(report_dir, os.path.basename(fn)))
        for (extra_opts_name, extra_opts) in [('_abbreviated', ["--abbreviatepast", "35"]), ("", [])]:
            my_call(["python", "-O", "data_summary_processor.py", "--includeranks", 
                     "--addavgrankcol", os.path.join(d, "summary.csv"), os.path.join(report_dir, "summary%s.csv" % extra_opts_name)]
                    + extra_opts)
        
        # TODO: Should really check that it has all the data stats first.
        data_stat_fn = os.path.join(REPORT_DIR, cat_name + "_data_stats.csv")
        if gen_stats and not os.path.exists(data_stat_fn):
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
    
    logging.info("Generating single fold selection graphs")
    dsn = 'zoo'
    rsdir = os.path.join(REPORT_DIR, 'selection_graphs')
    variations = ['Maximum Diversity Sampling', 'Sparsity Minimization']
    maybe_make_dirs(rsdir)
    r_dir = os.path.join(ALL_DIR, dsn, 'raw_results')
    logging.info("Generating selection graphs")
    for var in variations:
        vrfn = os.path.join(r_dir, var + ".tar.gz")
        my_call(["python", "-O", "gen_selection_graphs.py", dsn, vrfn, rsdir, 
              "--nocolour"]) 
        