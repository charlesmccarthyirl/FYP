'''
Created on Dec 21, 2011

@author: charles
'''
import unittest
from DataSets import named_data_sets
from BreadAndButter import k, classifier_generator as _classifier_generator, nns_getter_generator as _nns_getter_generator
from CaseProfiling import build_rcdl_profiles_brute_force, CaseProfileBuilder
import random

class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testIncrementalRcdl(self):
        for (name, data_info_loader) in named_data_sets[:3]:
            data_info = data_info_loader()
            data = data_info.data
            oracle = data_info.oracle
            distance_constructor = data_info.distance_constructor
            possible_classes = data_info.possible_classes
            
            data = list(data)
            random.Random(5).shuffle(data) # just so they're not all clustered together
            
            classifier_generator = lambda training_data: _classifier_generator(training_data, distance_constructor, possible_classes, oracle)
            nns_getter_generator = lambda training_data: _nns_getter_generator(training_data, distance_constructor, possible_classes, oracle)
            
            profile_builder = CaseProfileBuilder(k, _classifier_generator, distance_constructor, 
                                                 _nns_getter_generator, oracle, possible_classes)
            
            i = 1
            for case in data:
                print "No: %s, Case %s" % (i, case)
                
                # Use the profile builder to get the rcdl profiles
                profile_builder.put(case)
                profile_builder_rcdl_profiles = profile_builder.case_info_lookup
                
                # Just make sure it added stuff to the case base ok
                self.assertTrue(len(profile_builder.case_base) == i)
                self.assertTrue(case in profile_builder.case_base)
                self.assertTrue(profile_builder.case_info_lookup.has_key(case))
                
                # Use Brute Force to get the rcdl profiles
                distance_measurer = distance_constructor(profile_builder.case_base)
                nns_getter = nns_getter_generator(profile_builder.case_base)
                brute_force_rcdl_profiles = build_rcdl_profiles_brute_force(profile_builder.case_base, classifier_generator, distance_measurer, nns_getter, oracle)
                
                # Make sure that they're the same size so I don't 'miss' any
                self.assertTrue(len(profile_builder_rcdl_profiles) == len(brute_force_rcdl_profiles))
                
                # Iterate through each profile, and make sure its brute force equivalent is the same
                for (pb_case, pb_case_profile) in profile_builder_rcdl_profiles.items():
                    self.assertEqual(pb_case_profile, brute_force_rcdl_profiles[pb_case])
                
                i += 1

if __name__ == "__main__":
    unittest.main()