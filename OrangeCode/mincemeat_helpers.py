'''
Created on Mar 20, 2012

@author: charles
'''
import logging
from WorkerUnit import *
from collections import OrderedDict

def work_reducer(variation_info, work_unit_results):
    from SelectionStrategyEvaluator import MultiResultSet
    from utils import stream_getter
    work_unit_results = sorted(work_unit_results, key=lambda wur: wur.work_unit.fold_num)
    all_results = [wur.result for wur in work_unit_results]
    variation_result = MultiResultSet(all_results)

    with stream_getter(variation_info.raw_results_file) as stream:
        variation_result.serialize(stream)

def mapfn(k, work_unit):
    from WorkerUnit import main_gen_work_unit_result
    print "Mapping %s" % str(work_unit)
    res = (work_unit.variation_info, main_gen_work_unit_result(work_unit))
    yield res

def main_gen_raw_results(experiment, named_data_sets, experiment_directory, do_multi, password="changeme", *margs, **mkwargs):      
    logging.info("Generating work units")
    work_units = list(gen_work_units_iterable(experiment, named_data_sets, experiment_directory))
    
    if len(work_units) == 0:
        return
    
    if do_multi:
        import mincemeat
        logging.info("Beginning mincemeat server")
        s = mincemeat.Server(lambda variation_info, wurs: len(wurs) >= variation_info.total_folds)
        s.datasource = OrderedDict(((wu, wu) for wu in work_units))
        s.mapfn = mapfn
        s.reducefn = work_reducer
        s.run_server(password=password)
        logging.info("Ending mincemeat server")
    else:
        work_unit_results = (main_gen_work_unit_result(work_unit) for work_unit in work_units)
    
        for key, group in groupby(work_unit_results, lambda wur: wur.work_unit.variation_info):
            group = list(group)
            work_reducer(key, group)