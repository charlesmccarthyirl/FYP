#! /usr/bin/python
from SelectionStrategyEvaluator import *
import logging
import cProfile
import itertools
from utils import uniqueify
import logging
import latexcodec
import optparse
from main_runner_utils import *
from mincemeat_helpers import main_gen_raw_results as wu_mgrr
from traditional_runner import main_gen_raw_results as tr_mgrr
        
def main(experiment, named_data_sets, experiment_directory,
        do_create_summary=True, latex_encode=True, gen_only=True, 
        do_colour_plots=True, do_multi=True,
        main_gen_raw_results_func=wu_mgrr):
    if gen_only:
        logging.info("Beginning generating raw results")
        main_gen_raw_results_func(experiment, named_data_sets, experiment_directory, do_multi)
        logging.info("Ending generating raw results.")
        return
    
    logging.info("Beginning Nicity Processing.")
    experiment = get_experiment_obj(experiment)
    named_data_sets = get_named_data_sets_obj(named_data_sets)
    
    summary_results = OrderedDict()
    
    for (data_set_name, data_set_generator) in named_data_sets:
        logging.info("Beginning processing on %s" % data_set_name)
        
        l_experiment, data_info_generator = get_exp_ds_pair(experiment, data_set_generator)
        
        full_result_path, raw_results_dir = get_frp_rrp(experiment_directory, data_set_name)
        
        name_to_file_stream_getter_pairs = get_existing_variation_results_nfs_pairs(raw_results_dir)
        
        existing_results = ExperimentResult()
        existing_results.load_from_csvs(name_to_file_stream_getter_pairs)
        
        stream_from_name_getter = get_stream_from_name_getter_for(raw_results_dir)
        results = l_experiment.execute_on(data_info_generator, existing_results, 
                                          stream_from_name_getter=stream_from_name_getter)

        if do_create_summary:
            summary_results[data_set_name] = OrderedDict([(variation_name, var_result.AULC()) 
                                                   for (variation_name, var_result) 
                                                   in results.items()])
            
        try:
            g = results.generate_graph(data_set_name, colour=do_colour_plots)
            g.writePDFfile(os.path.abspath(full_result_path)) # Yes this is intentional, want it in the experiment directory, but with the same name as the folder.
        except ImportError, ex:
            logging.info("Unable to generate graph for %s data set. Graphing module unavailable in system: %s" %(data_set_name, ex)) 

    if do_create_summary:
        logging.info("Beginning summary csv generation")
        
        with open(os.path.join(experiment_directory, "summary.csv"), 'wb') as summary_stream:
            writer = csv.writer(summary_stream)
            writerow = writer.writerow
            format_num = lambda n: n if n is None else "%.3f" % n
            
            if latex_encode:
                str_encoder = lambda s: s.encode('latex')
                highlight = lambda x: "\\textbf{%s}" % x
            else:
                str_encoder = lambda s: s
                highlight = lambda x: x
            
            # Get all the union of all the variations names. 
            variations = uniqueify(itertools.chain(*[r.keys() for r in summary_results.values()]))
            data_names = summary_results.keys()
            
            top_results = [max(r.values()) for r in (summary_results[dn] for dn in data_names)]
            
            #None at start to leave column for variation names
            writerow([None] + map(str_encoder, data_names))
            
            for variation in variations:
                variation_results = [r.get(variation, None) for r in (summary_results[dn] for dn in data_names)]
                variation_results_highlighted = [highlight(format_num(v)) if v == t else format_num(v) for (v, t) in zip(variation_results, top_results)]
                row = [str_encoder(variation)] + variation_results_highlighted
                writerow(row)
        logging.info("Ending summary csv generation")

if __name__ == "__main__":
    parser = optparse.OptionParser("usage: %prog [options] experiment_module [datasets_module] experiment_directory")
    parser.add_option('--debug', help='boolean option which enables debug mode logging and execution', dest='debug',
                      default=False, action='store_true')
    parser.add_option('--latexencode', help='boolean option which forces latex encoding of outputs', dest='latexencode',
                      default=False, action='store_true')
    parser.add_option('--genonly', help='Generates the raw results only, no graphs, summaries, etc.', 
                      dest='gen_only',
                      default=False, action='store_true')
    parser.add_option('--old', help='boolean option which forces old multiprocessing style computation', 
                      dest='old',
                      default=False, action='store_true')
    parser.add_option('--multi', help='boolean option to enable multi-processing (local/distributed)', 
                      dest='multi',
                      default=False, action='store_true')
    parser.add_option('--nocolour', help='boolean option forces greyscale plotting', dest='colour',
                      default=True, action='store_false')
    parser.add_option('--profile', help='Profile the experiment, storing the profile results in the specified file', dest='profile',
                      default=None, action='store')
    parser.add_option('--password', help='Password to use when performing distributed computation', dest='password',
                      default="changeme", action='store')
    (options, args) = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s %(message)s',
                        level=(logging.DEBUG if options.debug else logging.INFO))
    experiment = args[0]
    named_data_sets = args[-2]
    experiment_directory = os.path.expanduser(args[-1])
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)
    if options.profile is not None:
        if not os.path.exists(os.path.dirname(options.profile)):
            os.makedirs(os.path.dirname(options.profile))
        cProfile.run("main(experiment, named_data_sets, experiment_directory, do_multi=False, gen_only=True)", options.profile)
    else:
        main_gen_raw_results_func = tr_mgrr if options.old else wu_mgrr
        main(experiment, named_data_sets, experiment_directory, gen_only=options.gen_only, 
             do_multi=options.multi, 
             do_colour_plots=options.colour, latex_encode=options.latexencode, 
             main_gen_raw_results_func=main_gen_raw_results_func,
             password=options.password)
