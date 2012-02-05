'''
Created on Feb 5, 2012

@author: charles
'''
import optparse
import csv
import pyx
import os
from utils import try_convert_to_num

if __name__ == '__main__':
    parser = optparse.OptionParser("usage: %prog [options] csv_file output_file")
    parser.add_option('-c', '--column', help='Column to include', dest='y_columns',
                      default=[], action='append', type='int')
    parser.add_option('--x_column', action="store", type="int", default=0, dest='x_column')
    parser.add_option('--y_title', action="store", type="string", default="", dest='y_title')
    parser.add_option('--pos', '-p', action="store", type="string", default="br", dest='pos') 
    parser.add_option('--name', '-n', action="store", type="string", default="", dest='name')

    
    (options, args) = parser.parse_args()
    
    csv_fn, output_file = args
    
    with open(csv_fn, 'r') as f:
        reader = csv.reader(f)
        header = reader.next()
        rows = [map(try_convert_to_num, row) for row in reader]
    
    y_columns = options.y_columns
    x_column = options.x_column
    name = options.name
    pos = options.pos
    
    if not y_columns:
        # Need to fill in everything that isn't 
        y_columns = range(len(header))
        y_columns.remove(x_column)
        
    
    y_name_to_indexes = [(header[i], i) for i in y_columns]
    
    g = pyx.graph.graphxy(width=10,
                  height=10, # Want a square graph . . .
                  x=pyx.graph.axis.linear(title=header[x_column]), 
                  y=pyx.graph.axis.linear(title=options.y_title),
                  key=pyx.graph.key.key(pos=pos)) 
    
    # either provide lists of the individual coordinates
    points = [pyx.graph.data.values(x=[t[x_column] for t in rows], 
                                y=[t[i] for t in rows], 
                                title=n) 
              for (n, i) in y_name_to_indexes]
    
    g.plot(points, [pyx.graph.style.line([pyx.color.gradient.ReverseRainbow])])
    
    name = name.replace("_", r"\_")
    g.text(g.width/2, 
           g.height + 0.2, 
           name,
           [pyx.text.halign.center, pyx.text.valign.bottom, pyx.text.size.Large])
    g.writePDFfile(output_file)