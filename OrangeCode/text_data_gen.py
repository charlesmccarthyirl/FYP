'''
Created on Jan 20, 2012

@author: charles
'''
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import Vectorizer
from time import time
from PrecomputedDistance import DataInfo
import os
import gzip

def get_20newsgroups_data_info_for_categories(categories):
    data = fetch_20newsgroups(subset='all', categories=categories, shuffle=False)
    vectorizer = Vectorizer()
    t0 = time()
    
    tfidf = vectorizer.fit_transform(data.data)
    
    pairwise_similarity = (tfidf * tfidf.T).todense().tolist()
    print "done in %fs" % (time() - t0)
    
    labels = [data.target_names[i] for i in data.target]
    payloads = [os.sep.join(e.split(os.sep)[-3:]) for e in data.filenames]
    
    # Similarity is from Zero to One - so (1-s) gives distance from 0 to 1.
    distances = [[(1-s) for s in row[:col_to+1]]for (col_to, row) in enumerate(pairwise_similarity)]
    
    # Fix the very slight off-ness involved in precision-conversion
    for row in distances:
        row[-1] = 0 
    
    pcd_tuples = zip(payloads, labels, distances)
    
    di = DataInfo.deserialize_pcd_tuples(pcd_tuples)
    return di

if __name__ == '__main__':
    name_cats_pairs = [
                        ('WinXwin', ['comp.windows.x', 'comp.os.ms-windows.misc']),
                        ('Comp', ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']),
                        ('Talk', ['talk.religion.misc', 'alt.atheism']),
                        ('Vehicle', ['rec.autos', 'rec.motorcycles'])
                        ]
    for (filename, categories) in name_cats_pairs:
        do_zip = True
        dir = '../Datasets/'
        
        filename = "%s.pcdb.gz" % filename
        
        di = get_20newsgroups_data_info_for_categories(categories)
        open_func = gzip.open if do_zip else open
        
        with open_func(os.path.join(dir, filename), 'wb') as stream:
            di.serialize(stream, DataInfo.SerializationMethod.proto, lambda inst: inst.payload)
    
    
    