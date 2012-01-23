from itertools import *
import logging
from collections import defaultdict
from utils import max_multiple

class RcdlCaseProfile:
    def __init__(self, 
                 reachability_set=(), 
                 coverage_set=(), 
                 dissimilarity_set=(), 
                 liability_set=(), 
                 nearest_neighbours=(),
                 reverse_nearest_neighbours=(),
                 classification=None):
        self.reachability_set = set(reachability_set)
        self.coverage_set = set(coverage_set)
        self.dissimilarity_set = set(dissimilarity_set)
        self.liability_set = set(liability_set)
        self.nearest_neighbours = set(nearest_neighbours)
        self.reverse_nearest_neighbours = set(reverse_nearest_neighbours)
        self.classification = classification
        
        self.metainfo = ["reachability_set", "coverage_set", "dissimilarity_set",
                         "liability_set", "nearest_neighbours", "reverse_nearest_neighbours", 
                         "classification"]
    
    def __eq__(self, other):
        return all(hasattr(self, a) == hasattr(other, a) 
            and getattr(self, a) == getattr(other, a) for a in self.metainfo)
    
    def __hash__(self):
        return reduce(lambda a, b: a^b, (hash(getattr(self, m)) for m in self.metainfo))
    
    def __str__(self):
        return ", ".join("%s = %s" % (s, getattr(self, s)) for s in self.metainfo)
    
    def __repr__(self):
        return str(self)
    
    def difference_update(self, other_case_profile):
        for attr in ("reachability_set", "coverage_set", "dissimilarity_set", "liability_set", 
                     "nearest_neighbours", "reverse_nearest_neighbours"):
            getattr(self, attr).difference_update(getattr(other_case_profile, attr))
    
    def update(self, other_case_profile):
        for attr in ("reachability_set", "coverage_set", "dissimilarity_set", "liability_set", 
                     "nearest_neighbours", "reverse_nearest_neighbours"):
            getattr(self, attr).update(getattr(other_case_profile, attr))

def build_rcdl_profiles_brute_force(case_base, classifier, distance_measure, nns_getter, oracle):
    case_to_profile_dict = defaultdict(RcdlCaseProfile)
    
    # Sort NNs / rNNs
    for case in case_base:
        case_rcdl = case_to_profile_dict[case]
        case_nns = nns_getter(case)
        case_rcdl.nearest_neighbours.update(case_nns)
        for nn in case_nns:
            case_to_profile_dict[nn].reverse_nearest_neighbours.add(case)

    for case in case_base:
        case_rcdl = case_to_profile_dict[case]
        case_actual_class = oracle(case)
        case_classified_class = classifier(case)
        case_rcdl.classification = case_classified_class
        
        if case_actual_class == case_classified_class:
            helping_cases = [nn for nn in case_rcdl.nearest_neighbours if oracle(nn) == case_actual_class]
            case_rcdl.reachability_set.update(helping_cases)
            for hc in helping_cases:
                case_to_profile_dict[hc].coverage_set.add(case)
        else:
            hurting_cases = [nn for nn in case_rcdl.nearest_neighbours if oracle(nn) != case_actual_class]
            case_rcdl.dissimilarity_set.update(hurting_cases)
            for hc in hurting_cases:
                case_to_profile_dict[hc].liability_set.add(case)
    
    return case_to_profile_dict

class AddRemovalStore:
    def __init__(self):
        self.added = RcdlCaseProfile()
        self.removed = RcdlCaseProfile()
    
    def __eq__(self, other):
        return self.added == other.added and self.removed == other.removed
    
    def __hash__(self):
        return hash(self.added)^hash(self.removed)
    
    def __str__(self):
        return "Added: %s, Removed: %s" % (self.added, self.removed)
    
    def apply_to(self, profile):
        profile.difference_update(self.removed)
        profile.update(self.added)
        if self.removed.classification is not None or self.added.classification is not None:
            profile.classification = self.added.classification

class CaseProfileBuilder:
    def __init__(self, k, classifier_generator, distance_constructor, 
                 nns_getter_generator, oracle, possible_classes, *args, **kwargs):
        '''
        Initializes a new instance of the CaseProfileBuilder class.
        
        @param classifier_generator: Func which takes a case_base and generates a classifier trained on it
        '''
        self.__classifier_generator = classifier_generator
        self.__distance_constructor = distance_constructor
        self.__k = k
        self.__nns_getter_generator = nns_getter_generator
        self.__oracle = oracle
        self.__possible_classes = possible_classes

        self.case_info_lookup = {}
        self.case_base = [] # keys() generates new dict. Want to be able to access in O(1) time, not O(n)
    
    def __get_oracle(self):
        return self.__oracle
    
    def __get_generator_args(self):
        oracle = self.__get_oracle()
        return (self.__distance_constructor, self.__possible_classes, oracle)
    
    def __get_or_create(self, case):
        if not self.case_info_lookup.has_key(case):
            self.case_info_lookup[case] = RcdlCaseProfile()
            self.case_base.append(case)
        return self.case_info_lookup[case]
    
    def put(self, _case):
        add_removals_dict = self.suppose(_case, self.__get_oracle()(_case))
        
        for (case, add_removal) in add_removals_dict.items():
            case_profile = self.__get_or_create(case)
            add_removal.apply_to(case_profile)
        
        return add_removals_dict
    
    def classify(self, training_data, case):
        return self.__classifier_generator(training_data, *self.__get_generator_args())(case)
    
    def nns_getter(self, training_data, case):
        return self.__nns_getter_generator(training_data, *self.__get_generator_args())(case)
    
    def __suppose_nn(self, case):
        '''
        Determine the changes that would occur to all NN rNN sets if case were added.
        
        @param case: The case to suppose the addition of.
        '''
        assert(not self.case_info_lookup.has_key(case)) # Just don't want to deal with the hassle right now.
        
        add_removals_dict = {}
        
        def get_or_create(_case):
            if not add_removals_dict.has_key(_case):
                add_removals_dict[_case] = AddRemovalStore()
            return add_removals_dict[_case]
        
        case_changes = get_or_create(case)
        
        case_nns = self.nns_getter(self.case_base, case)
        
        case_changes.added.nearest_neighbours.update(case_nns)
        for nn in case_nns:
            nn_changes = get_or_create(nn)
            nn_changes.added.reverse_nearest_neighbours.add(case)
        
        for (other_case, other_case_profile) in self.case_info_lookup.items():
            if len(other_case_profile.nearest_neighbours) < self.__k: # Less than, as I'm going to be adding to (in which case if it was k-1, it's going to become k)
                other_case_new_nns = chain(other_case_profile.nearest_neighbours, (case,))
                shunted = None
            else:
                # TODO: There is the issue of comparability here. Change to pass distance_measurer to __init__ instead of distance_constructor
                dist_meas = self.__distance_constructor(other_case_profile.nearest_neighbours)
                maxes = max_multiple(((other_case_nn, dist_meas(other_case, other_case_nn))
                                for other_case_nn 
                                in other_case_profile.nearest_neighbours), key=lambda el: el[1])
                max_ex, max_dist = maxes[0]
                case_dist = dist_meas(other_case, case)
                
                if case_dist > max_dist:
                    continue
                elif case_dist == max_dist or len(maxes) > 1:
                    # Unsure - deferreing decision to nn finder for tie breaking
                    other_case_new_nns = self.nns_getter(chain(other_case_profile.nearest_neighbours, (case,)), other_case)
                    if case not in other_case_new_nns:
                        continue
                    
                    difference = set(other_case_profile.nearest_neighbours).difference(other_case_new_nns)
                    assert(len(difference) == 1)
                    shunted = difference.pop()
                else:
                    other_case_new_nns = [nn for nn in other_case_profile.nearest_neighbours 
                                          if nn is not max_ex]
                    other_case_new_nns.append(case)
                    shunted = max_ex
            
            other_case_changes = get_or_create(other_case)
            
            # Heuston - we have a reversed nearest neighbour - 
            case_changes.added.reverse_nearest_neighbours.add(other_case)
            other_case_changes.added.nearest_neighbours.add(case)
            
            # now, did it shunt out another one for the position
            if shunted is None:
                continue 

            shunted_changes = get_or_create(shunted)
            
            other_case_changes.removed.nearest_neighbours.add(shunted)
            shunted_changes.removed.reverse_nearest_neighbours.add(other_case)
        
        assert(all(ca not in add_removed.added.nearest_neighbours for (ca, add_removed) in add_removals_dict.items()))
        return add_removals_dict
    
    def suppose(self, _case, _class):
        '''
        Determines the changes that would occur if _case were to be added with label _class to the case base,
        where a change is considered anything with an affected NN, rNN, Coverage, Reachability, Dissimilarity
        or Liability set.
        
        @param _case: The case to suppose the addition of.
        @param _class: The label to suppose that the case would be added with.
        '''
        assert(not self.case_info_lookup.has_key(_case)) # Just don't want to deal with the hassle right now.
        
        add_removals_dict = self.__suppose_nn(_case)
        
        oracle = self.__get_oracle()
        def get_class(m_case):
            return _class if m_case == _case else oracle(m_case) # want the fake class if I'm dealing with the input _case, otherwise, the case's real class
        
        def get_or_create(case):
            if not add_removals_dict.has_key(case):
                add_removals_dict[case] = AddRemovalStore()
            return add_removals_dict[case]
        
        if len(self.case_base) == 0: # Nothing in the case base, so no neighbours or the likes.
            _case_add_removal = get_or_create(_case)
            _case_add_removal.added.classification = self.classify([], _case)
        
        for (case, case_add_removals) in list(add_removals_dict.items()): # list, because I'll be updating as I go
            # Only really need to deal with the added and removed nearest neighbours,
            # since for each there will be a corresponding added/removed reversed nearest neighbour
            #
            # Truth table for possibilities of a given case.
            # NN Added | NN Removed
            #    1     |     1
            #    1     |     0      <- Happens only at start, as the NN is filling to capacity. Same as added above.
            #    0     |     1      <- Can never actually happen - would have to be another replace it.
            #                          Note: if I add a suppose_remove, accounting for this case would complete the pic
            #    0     |     0      <- Uninteresting - it's R_NN will be covered elsewhere.
            
            case_info = None
            case_old_class = None
            if self.case_info_lookup.has_key(case):
                case_info = self.case_info_lookup[case]
                case_old_class = case_info.classification
               
            if not case_info: 
                assert(case == _case)

            if len(case_add_removals.added.nearest_neighbours) == 0:
                continue # There can't be any adds if there were no removals
                # Note: This will need to be changed if I want to support suppose_remove
            
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
            
            case_new_nearest_neighbours = set(case_add_removals.added.nearest_neighbours)
    
            if case_info is not None:
                case_new_nearest_neighbours.update(case_info.nearest_neighbours)
                # Only could be removals if there was stuff there before
                case_new_nearest_neighbours.difference_update(case_add_removals.removed.nearest_neighbours)
            
            # If the classification changed as a result of the adds, scrubbing needed.
            # Otherwise, just need to check if the added case helps or hinders this one, and if so, update as approrpiate.
            case_new_class = self.classify(case_new_nearest_neighbours, case)
            case_actual_class = get_class(case)
            
            old_correct_classification_status = case_old_class == case_actual_class
            new_correct_classification_status = case_new_class == case_actual_class
            
            if case_old_class is not None \
               and old_correct_classification_status != new_correct_classification_status: 
                # Case's correct-classification-status has changed - need to go a-scrubbing.
                # Generate the list of things which 'helped' in the classification
                assert(len(case_info.reachability_set) == 0 or len(case_info.dissimilarity_set) == 0)
                to_remove = case_info.reachability_set or case_info.dissimilarity_set
                
                direct, reverse = self.get_direct_reverse_names(case_old_class, get_class(case))
    
                # case's dissimilarity set, and those things liability set need updating.
                assert(len(set(to_remove).difference(getattr(case_info, direct))) == 0)
                getattr(case_add_removals.removed, direct).update(to_remove)
                for r in to_remove:
                    assert(case in getattr(self.case_info_lookup[r], reverse))
                    getattr(get_or_create(r).removed, reverse).add(case)
            
            if case_old_class != case_new_class:
                case_add_removals.removed.classification = case_old_class
                case_add_removals.added.classification = case_new_class
            
            direct, reverse = self.get_direct_reverse_names(case_new_class, get_class(case))
            
            # Now any scrubbing related the old class is done. Time to deal with adding
            to_add = [c for c in case_new_nearest_neighbours 
                      if (get_class(c) == get_class(case)) == new_correct_classification_status]
            
            # Need to filter to what I'm *actually* going to add (what's not there already).
            if case_info is not None:
                to_add = [c for c in to_add if c not in getattr(case_info, direct)] 
            
            getattr(case_add_removals.added, direct).update(to_add)
            for a in to_add:
                getattr(get_or_create(a).added, reverse).add(case)
    
        return add_removals_dict
    
    def get_direct_reverse_names(self, actual_class, proposed_class):
        if (actual_class == proposed_class):
            direct = "reachability_set"
            reverse = "coverage_set"
        else:
            direct = "dissimilarity_set"
            reverse = "liability_set"
        
        return (direct, reverse)