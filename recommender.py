import numpy as np
import os, sys
import random
import operator

# dummy function for initial testing
def beta_crit(ht_facts):
	pick = np.random.randint(ht_facts.shape[0])
	crit = ht_facts[pick]

	# remove from pool and return in (node, rel) format
	ht_facts = np.delete(ht_facts, pick, axis=0)
	if crit[0] == -1.0: return (crit[2], crit[1]), ht_facts
	else: return (crit[0], crit[1]), ht_facts

### inputs: facts about the ground truth and facts about the recommended items as well as the critique mode and the dictionary containing popularities of each object
### Also, we input the facts in which the gt is placed in their tail to differentiate between objects and relations
def select_critique(critique_selection_data, rec_facts, critique_mode, pop_counts, items_facts_tail_gt):
	facts_diff = {}
	#we want to count how many times each ground truth fact is satisfied by the facts about recommended items
	for fact in critique_selection_data:
		condition = (fact[0] == rec_facts[:, 0]) & (fact[1] == rec_facts[:, 1])
		facts_diff[tuple(fact)] = np.count_nonzero(condition)

	if critique_mode=="random":
		# in this case, we want to randomly choose a critique from either head or tail
		if critique_selection_data.size:
			#we're only keeping the facts that are not satisfied by all recommended items
			candidate_facts = [k for k, v in facts_diff.items() if v < 20]
			critique_fact = random.choice(candidate_facts)
			if any((items_facts_tail_gt[:] == np.array((critique_fact)+(-1,))).all(1)):
				obj = critique_fact[0]
			else:
				obj = critique_fact[1]
		else:
			obj , critique_fact = None, None

	# In this case, we want to select the fact about the most famous object as the crtique
	if critique_mode == "pop":
		if critique_selection_data.size:
			candidate_facts = [k for k, v in facts_diff.items() if v < 20]
			popularities = {}
			facts = {}
			for fact in candidate_facts:
				# Checking if the obj is in the head (gt in the tail)
				if any((items_facts_tail_gt[:] == np.array((fact)+(-1,))).all(1)):
					candidate_obj = fact[0]
					popularities[candidate_obj] = pop_counts[fact[0]]
					facts[candidate_obj] = fact
				else:
					candidate_obj = fact[1]
					popularities[candidate_obj] = pop_counts[fact[1]]
					facts[candidate_obj] = fact
			obj = max(popularities.items(), key=operator.itemgetter(1))[0]
			critique_fact = facts[obj]
		else:
			obj , critique_fact = None, None

    # in this case, we should take the fact that deviates the most from the facts related to recommended items, i.e. least satisfaction counts
	if critique_mode == "diff":        
		if bool(facts_diff):
			# this is the number of times the most deviating fact is satisfied
			minval = min(facts_diff.values())
			# in the most deviating fact, gt is in the tail 
			# we might have multiple facts satisfied minval times. In this case, we choose the most famous object as the critique
			candidates_list = [k for k, v in facts_diff.items() if v == minval]
			if len(candidates_list)>1:
				popularities = {}
				facts = {}
				for fact in candidates_list:
					if any((items_facts_tail_gt[:] == np.array((fact)+(-1,))).all(1)):
						candidate_obj = fact[0]
						popularities[candidate_obj] = pop_counts[fact[0]]
						facts[candidate_obj] = fact
					else:
						candidate_obj = fact[1]
						popularities[candidate_obj] = pop_counts[fact[1]]
						facts[candidate_obj] = fact
				obj = max(popularities.items(), key=operator.itemgetter(1))[0]
				critique_fact = facts[obj]
			else:
				if any((items_facts_tail_gt[:] == np.array((candidates_list[0])+(-1,))).all(1)):
					obj = candidates_list[0][0]
				else:
					obj = candidates_list[0][1]
					critique_fact = candidates_list[0]

		else:
			obj , critique_fact = None, None

	return obj , critique_fact

def remove_chosen_critiques( critiquing_candidate, previous_critiques):
	for end in ["head", "tail"]:
		for record in previous_critiques[end]:
			if len(critiquing_candidate[end]) > 1:
				critiquing_candidate[end].remove(record)

	return critiquing_candidate

def obj2item(items,obj,data):
	objkg=data[np.where((data[:, 0] == obj) | (data[:, 2] == obj))]
	objkg=np.delete(objkg,1,1)
	mapped_items=np.intersect1d(items,data)
	return mapped_items