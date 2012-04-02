import csv
import os
import uuid

def printFigure(exp_dir, caption=None, label=None):
    u = str(uuid.uuid1()).replace('-', '')
    fn = os.path.join(exp_dir, 'summary.csv')
    print r'\DTLloaddb{%s}{%s}' %(u, fn)
    with open(fn) as f:
        r = csv.reader(f)
        r = list(r)
        datasets = r[0][1:-1]
        print r'\begin{figure}[h!]'
        print r'\centering'
        for (i, ds) in enumerate(datasets):  
            ds_fn = os.path.join(exp_dir, ds)
            print r'\subfigure{' 
            print r'\includegraphics[height=0.25\textheight]{"%s"}' % (ds_fn)
            print '}'
        if label:
            print r'\label{%s}' % label
        if caption:
            print r'\caption{%s}' % caption
        
    print r'\end{figure}'
    print
    print r'\resizebox{\textwidth}{!} {'
    print r'\DTLdisplaydb{%s}' % u
    print r'}'
    print
    