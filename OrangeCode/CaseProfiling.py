from itertools import *
import orange

class RcdlCaseProfile:
    def __init__(self, 
                 reachability_set=(), 
                 coverage_set=(), 
                 dissimilarity_set=(), 
                 liability_set=(), 
                 nearest_neighbours=(),
                 reverse_nearest_neighbours=()):
        self.reachability_set = set(reachability_set)
        self.coverage_set = set(coverage_set)
        self.dissimilarity_set = set(dissimilarity_set)
        self.liability_set = set(liability_set)
        self.nearest_neighbours = set(nearest_neighbours)
        self.reverse_nearest_neighbours = set(reverse_nearest_neighbours)
    
    def difference_update(self, other_case_profile):
        for attr in ("reachability_set", "coverage_set", "dissimilarity_set", "liability_set", 
                     "nearest_neighbours", "reverse_nearest_neighbours"):
            getattr(self, attr).difference_update(getattr(other_case_profile, attr))
    
    def update(self, other_case_profile):
        for attr in ("reachability_set", "coverage_set", "dissimilarity_set", "liability_set", 
                     "nearest_neighbours", "reverse_nearest_neighbours"):
            getattr(self, attr).update(getattr(other_case_profile, attr))

class AddRemovalStore:
    def __init__(self):
        self.added = RcdlCaseProfile()
        self.removed = RcdlCaseProfile()
    
    def apply_to(self, profile):
        profile.difference_update(self.removed)
        profile.update(self.added)

class CaseProfileBuilder:
    # TODO - add unit test here. Build an entire 1. Check to NNs match against brute force.
    def nns_getter(self, case_base, case):
        try:
            return self.__get_classifier(case_base).find_nearest(case, classifier.k)
        except:
            assert(len(case_base) == 0)
            return []
    
    def __get_classifier(self, cases):
        if not (type(cases) is list or type(cases) is orange.ExampleTable):
            cases = list(cases)
        return self.__classifier_generator(cases, self.__distance_constructor)
    
    def __init__(self, classifier_generator, distance_constructor):
        '''
        Initializes a new instance of the CaseProfileBuilder class.
        
        @param classifier_generator: Func which takes a case_base and generates a classifier trained on it
        '''
        
        #TODO: Possibly include the Oracle
        
        self.__classifier_generator = classifier_generator
        self.__distance_constructor = distance_constructor

        self.case_info_lookup = {}
    
    def __get_or_create(case):
        if not self.case_info_lookup.has_key(case):
            self.case_info_lookup[case] = AddRemovalStore()
        return self.case_info_lookup[case]
    
    def put(self, _case):
        add_removals_dict = self.suppose(_case)
        
        for (case, add_removal) in add_removals_dict.items():
            case_profile = self.__get_or_create(case)
            add_removal.apply_to(case_profile)
        
        return add_removals_dict
    
    def __suppose_nn(self, case):
        assert(not self.case_info_lookup.has_key(case)) # Just don't want to deal with the hassle right now.

        add_removals_dict = {}
        
        def get_or_create(_case):
            if not add_removals_dict.has_key(_case):
                add_removals_dict[_case] = AddRemovalStore()
            return add_removals_dict[_case]
        
        case_changes = get_or_create(case)
        
        case_nns = self.nns_getter(self.case_info_lookup.keys(), case)
        
        for (other_case, other_case_profile) in self.case_info_lookup.items():
            other_case_new_nns = self.nns_getter(chain(other_case_profile.nearest_neighbours, (case,)), other_case)
            if case not in other_case_new_nns:
                continue
            
            other_case_changes = get_or_create(other_case)
            
            # Heuston - we have a reversed nearest neighbour - 
            case_changes.reverse_nearest_neighbours.add(other_case)
            other_case_changes.added.nearest_neighbours.add(case)
            
            # now, did it shunt out another one for the position
            difference = set(other_case_profile.nearest_neighbours).difference(other_case_new_nns)
            assert(len(difference) <= 1)
            if len(difference) == 0:
                continue 
            
            shunted = difference[0]
            shunted_changes = get_or_create(shunted)
            
            other_case_changes.removed.nearest_neighbours.add(shunted)
            shunted_changes.removed.reverse_nearest_neighbours.add(other_case)
        
        return add_removals_dict
    
    def suppose(self, _case):
        
        add_removals_dict = self.__suppose_nn(_case)
        
        def get_or_create(case):
            if not add_removals_dict.has_key(case):
                add_removals_dict[case] = AddRemovalStore()
            return add_removals_dict[case]
        
        for (case, case_add_removals) in list(add_removals_dict.items()): # list, because I'll be updating as I go
            # Only really need to deal with the added and removed nearest neighbours,
            # since for each there will be a corresponding added/removed reversed nearest neighbour
            #
            # Truth table for possibilities of a given case.
            # NN Added | NN Removed
            #    1     |     1
            #    1     |     0      <- Happens only at start, as the NN is filling to capacity. Same as added above.
            #    0     |     1      <- Can never actually happen - would have to be another replace it.
            #                          TODO - if I add a suppose_remove, accounting for this case would complete the pic
            #    0     |     0      <- Uninteresting - it's R_NN will be covered elsewhere.
            
            case_info = None
            if self.case_info_lookup.has_key(case):
                case_info = self.case_info_lookup[case]
               
            if not case_info: 
                assert(case == _case)
            
            # Deal with NN removals
            for removed_nn_case in case_add_removals.removed.nearest_neighbours:
                # might affect R/D of me, C/L of other.
                # removed_nn_case should definitely be in add_removals_dict, as should be in removed r_nn
                assert(add_removals_dict.has_key(removed_nn_case))
                removed_nn_case_add_removals = add_removals_dict[removed_nn_case]
                assert(case in removed_nn_case_add_removals.removed.reverse_nearest_neighbours)
                
                assert(self.case_info_lookup.has_key(removed_nn_case))
                removed_nn_case_case_info = self.case_info_lookup[removed_nn_case]
                
                if removed_nn_case in case_info.reachability_set:
                    case_add_removals.removed.reachability_set.add(removed_nn_case)
                    assert(case in removed_nn_case_case_info.coverage_set)
                    removed_nn_case_add_removals.removed.coverage_set.add(case)
                elif removed_nn_case in case_info.dissimilarity_set:
                    case_add_removals.removed.dissimilarity_set.add(removed_nn_case)
                    assert(case in removed_nn_case_case_info.liability_set)
                    removed_nn_case_add_removals.removed.liability_set.add(case)
                
                # Not worrying about if it changed the class or anything, since that'll
                # be dealt with anyway in the corresponding add.
                # Note: This will need to be changed if I want to support suppose_remove
            
            if len(case_add_removals.added.nearest_neighbours) == 0:
                continue
            
            case_old_class = None 
            case_new_nearest_neighbours = set(case_add_removals.added.nearest_neighbours)

            if case_info is not None:
                case_new_nearest_neighbours.update(case_info.nearest_neighbours)
                case_new_nearest_neighbours.difference_update(case_add_removals.removed.nearest_neighbours)
                
                case_old_class = self.__get_classifier(case_info.nearest_neighbours)(case)
            
            # If the case changed as a result of the adds, scrubbing needed.
            # Otherwise, just need to check if the added case helps or hinders this one, and if so, 
            case_new_class = self.__get_classifier(case_new_nearest_neighbours)(case)
            
            if case_old_class is not None and case_old_class != case_new_class:
                # Case has changed - need to go a-scrubbing.
                # Generate the list of things which 'helped' in the classification
                to_remove = [c for c in case_info.nearest_neighbours if c.get_class() == case_old_class]
                
                if case.get_class() == case_old_class:
                    # Need to remove from this cases reachability, and from each of those, this case from their coverage
                    assert(len(set(to_remove)).difference(case_info.reachability_set) == 0)
                    case_add_removals.removed.reachability_set.update(to_remove)
                    for r in to_remove:
                        assert(case in self.case_info_lookup[r].coverage_set)
                        get_or_create(r).removed.coverage_set.add(case)
                else:
                    # case's dissimilarity set, and those things liability set need updating.
                    assert(len(set(to_remove)).difference(case_info.dissimilarity_set) == 0)
                    case_add_removals.removed.dissimilarity_set.update(to_remove)
                    for r in to_remove:
                        assert(case in self.case_info_lookup[r].liability_set)
                        get_or_create(r).removed.liability_set.add(case)
                
            # Now any scrubbing related the old class is done. Time to deal with adding
            to_add = [c for c in case_info.nearest_neighbours if c.get_class() == case_new_class]
            
            # Need to filter to what I'm *actually* going to add (what's not there already).
            if case_info is not None:
                to_add = [c for c in to_add if c not in case_info.reachability_set] 
            
            case_add_removals.added.reachability_set.update(to_add)
            for a in to_add:
                get_or_create(a).added.coverage_set.add(case)
        
        return add_removals_dict
    