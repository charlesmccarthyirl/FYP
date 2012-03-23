'''
Created on Mar 23, 2012

@author: charles
'''
import optparse
import csv
from collections import OrderedDict
from utils import average, try_convert_to_num as convert, uniqueify
from itertools import izip

class Result:
    def __init__(self, score, rank=None):
        self.score = score
        self.rank = rank
        
if __name__ == '__main__':
    parser = optparse.OptionParser("usage: %prog [options] input_file output_file")
    parser.add_option('--includeranks', dest='include_ranks', default=False, action='store_true')
    parser.add_option('--highlightbest', dest='highlight_best', default=False, action='store_true')
    parser.add_option('--addavgrankcol', dest='add_avg_rank_col', default=False, action='store_true')
    parser.add_option('--avgrankcolname', dest='avg_rank_col_name', default="Avg. Rank", action='store')
    (options, args) = parser.parse_args()
    
    input_f, output_f = args[:2]
    
    with open(input_f, 'r') as input_s:
        reader = csv.reader(input_s)
        rows = [map(convert, row) for row in reader if len(row) > 1]
    
    dsn_to_results_dict = OrderedDict()
    
    dataset_names = rows[0][1:]
    for dsn in dataset_names:
        dsn_to_results_dict[dsn] = []
    
    strats = []
    for r in rows[1:]:
        strats.append(r[0])
        for (v, (dsn, results)) in  izip(r[1:], dsn_to_results_dict.iteritems()):
            results.append(Result(v, None))
    
    for (dsn, results) in dsn_to_results_dict.iteritems():
        result_scores = [r.score for r in results]
        result_scores_for_indexing = sorted(uniqueify(result_scores), reverse=True)
        for r in results:
            score = result_scores_for_indexing.index(r.score) + 1
            r.rank = score
    
    with open(output_f, 'wb') as output_s:
        writer = csv.writer(output_s)
        writer.writerow([""] + dsn_to_results_dict.keys() 
                        + [options.avg_rank_col_name] if options.add_avg_rank_col else [] )
        strats_results = zip(*dsn_to_results_dict.values())
        names_to_strats_results = zip(strats, strats_results)
        for (strat_name, strat_results) in names_to_strats_results:
            if options.include_ranks:
                formatter = lambda res: "%.03f (%d)" % (res.score, res.rank) 
            else:
                formatter = lambda res: "%.03f" % res.score
                
            if options.highlight_best:
                old_f = formatter
                formatter = lambda s: r"\textbf{%s}" % s
            
            row = [strat_name] + map(formatter, strat_results)
            if options.add_avg_rank_col:
                row += ["%.02f" % average((r.rank for r in strat_results))]
            
            writer.writerow(row)
        
            
    
    
    