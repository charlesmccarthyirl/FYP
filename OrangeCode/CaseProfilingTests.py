'''
Created on Dec 21, 2011

@author: charles
'''
import unittest
from DataSets import named_data_sets as non_textual_data_sets
from TextualDataSets import named_data_sets as textual_data_sets
from BreadAndButter import k, classifier_generator as _classifier_generator, nns_getter_generator as _nns_getter_generator
from CaseProfiling import build_rcdl_profiles_brute_force, CaseProfileBuilder
from utils import Timer
from itertools import chain
import random, csv, os, time

def do_nothing(*args, **kwargs):
    pass

def build_rcdl_incr_bf(
                case_base, 
                blank_profile_getter):
    profile_builder = blank_profile_getter()
    for case in case_base:
        profile_builder.put(case)
    
    return profile_builder.case_info_lookup

        
def testIncrementalRcdl(data_info_loader, assertTrue=None, do_cumulative_incremental=False):
    assertTrue = assertTrue or do_nothing

    data_info = data_info_loader()
    data = data_info.data
    oracle = data_info.oracle
    distance_constructor = data_info.distance_constructor
    possible_classes = data_info.possible_classes
    
    data = list(data)
    random.Random(5).shuffle(data) # just so they're not all clustered together
    
    classifier_generator = lambda training_data: _classifier_generator(training_data, distance_constructor, possible_classes, oracle)
    nns_getter_generator = lambda training_data: _nns_getter_generator(training_data, distance_constructor, possible_classes, oracle)
    
    profile_builder_getter = lambda: CaseProfileBuilder(k, _classifier_generator, distance_constructor, 
                                         _nns_getter_generator, oracle, possible_classes)
    profile_builder = profile_builder_getter()
    
    timings = []
    i = 1
    for case in data:
        # Use the profile builder to get the rcdl profiles
        start = time.clock()
        profile_builder.put(case)
        end = time.clock()
        pb_interval = end - start
        
        profile_builder_rcdl_profiles = profile_builder.case_info_lookup
        
        # Just make sure it added stuff to the case base ok
        assertTrue(len(profile_builder.case_base) == i)
        assertTrue(case in profile_builder.case_base)
        assertTrue(profile_builder.case_info_lookup.has_key(case))
        
        # Use Brute Force to get the rcdl profiles
        distance_measurer = distance_constructor(profile_builder.case_base)
        nns_getter = nns_getter_generator(profile_builder.case_base)
        
        start = time.clock()
        brute_force_rcdl_profiles = build_rcdl_profiles_brute_force(profile_builder.case_base, classifier_generator, distance_measurer, nns_getter, oracle)
        end = time.clock()
        bf_interval = end - start
        
        if do_cumulative_incremental:
            # Use incremental brute force to get rcdl profiles
            start = time.clock()
            bf_incr_rcdl_profiles = build_rcdl_incr_bf(profile_builder.case_base, 
                                                       profile_builder_getter)
            end = time.clock()
            bf_incr_interval = end - start
        
        # Make sure that they're the same size so I don't 'miss' any
        assertTrue(len(profile_builder_rcdl_profiles) == len(brute_force_rcdl_profiles))
        if do_cumulative_incremental:
            assertTrue(len(profile_builder_rcdl_profiles) == len(bf_incr_rcdl_profiles))
        
        # Iterate through each profile, and make sure its brute force equivalent is the same
        for (pb_case, pb_case_profile) in profile_builder_rcdl_profiles.items():
            assertTrue(pb_case_profile == brute_force_rcdl_profiles[pb_case])
            if do_cumulative_incremental:
                assertTrue(pb_case_profile == bf_incr_rcdl_profiles[pb_case])
        
        ts = [i, pb_interval, bf_interval]
        if do_cumulative_incremental:
            ts.append(bf_incr_interval)
        
        timings.append(ts)
        
        i += 1
    
    return timings

class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testIncrementalRcdl(self):
        for (name, data_info_loader) in chain(non_textual_data_sets, textual_data_sets):
            testIncrementalRcdl(data_info_loader, assertTrue=self.assertTrue)
            
if __name__ == "__main__":
    unittest.main()