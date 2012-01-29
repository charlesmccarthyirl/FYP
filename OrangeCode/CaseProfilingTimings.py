'''
Created on Jan 28, 2012

@author: charles
'''
import csv, sys, os
from CaseProfilingTests import testIncrementalRcdl
from TextualDataSets import named_data_sets as textual_data_sets
from DataSets import named_data_sets as non_textual_data_sets
from itertools import chain, islice
import pyx

if __name__ == '__main__':
    name = sys.argv[1]
    
    if name.endswith('.csv'):
        d = os.path.dirname(name)
        fn = name
        name = os.path.splitext(os.path.basename(name))[0]
        with open(fn, 'r') as f:
            reader = csv.reader(f)
            timings = [(int(cb), float(inc), float(bf))
                       for (cb, inc, bf) in islice(reader, 1, None)]
    else:
        d = sys.argv[2] if len(sys.argv) > 2 else '.'
        if not os.path.exists(d):
            os.mkdir(d)
        
        textual_dict = dict(chain(textual_data_sets, non_textual_data_sets))
        
        data_info_loader = textual_dict[name]
        
        timings = testIncrementalRcdl(data_info_loader)
    
    with open(os.path.join(d, name+'.csv'), 'wb') as f:
        w = csv.writer(f)
        w.writerow(('Case Base Size', 'Incremental', 'Brute Force'))
        w.writerows(timings)
    
    g = pyx.graph.graphxy(width=10,
                  height=10, # Want a square graph . . .
                  x=pyx.graph.axis.linear(title="Case Base Size"), 
                  y=pyx.graph.axis.linear(title="Time for building RCDL Profiles for Single Addition (/s)"),
                  key=pyx.graph.key.key(pos="br")) 
    
    # either provide lists of the individual coordinates
    points = [pyx.graph.data.values(x=[t[0] for t in timings], 
                                y=[t[i] for t in timings], 
                                title=n) 
              for (n, i) in [("Incremental", 1), ("Brute Force", 2)]]
    
    g.plot(points, [pyx.graph.style.line([pyx.color.gradient.ReverseRainbow])])
    
    name = name.replace("_", r"\_")
    g.text(g.width/2, 
           g.height + 0.2, 
           name,
           [pyx.text.halign.center, pyx.text.valign.bottom, pyx.text.size.Large])
    g.writePDFfile(os.path.join(d, name + '.pdf'))
    