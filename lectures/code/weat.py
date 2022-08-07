# Attribution: From MDS-CL notes 
# paper: https://purehost.bath.ac.uk/ws/portalfiles/portal/168480066/CaliskanEtAl_authors_full.pdf
# Code: https://github.com/kmccurdy/w2v-gender/tags

import numpy as np
from itertools import combinations, filterfalse
import math
import logging

class WEAT(object):

    def __init__(self, model):
        self.model = model
        
    def check_vocab(self, word_set):
        '''make sure the words in word_set are in the vocabulary of the model'''
        missing = [w for w in word_set if w not in self.model]
        if missing:
            print("missing:",missing)
        return([w for w in word_set if w in self.model])
    
    def check_vocab_sets(self, *word_sets):
        '''check all the wordsets for this run of WEAT'''
        return([self.check_vocab(ws) for ws in word_sets])
    
    def avg_sim(self, word, word_set):
        '''calculate the average cosine similarity between a word and a set of words'''
        similarities = self.model.n_similarity([word], word_set)
        return(np.mean(similarities))
    
    def sim_set_diff(self, word, word_set1, word_set2):
        '''get the difference in average similarities of a word compared to the words two different word_sets'''
        return(self.avg_sim(word, word_set1) - self.avg_sim(word, word_set2))
    
    def _weat_test(self, tw1, tw2, aw1, aw2):
        '''Do basic WEAT test by calculating average difference in similarities comparing the two attribute
        word_sets aw1, aw2 to all the words in each of the targets word lists (tw1, tw2) '''
        targ_sims1 = [self.sim_set_diff(w, aw1, aw2) for w in tw1]
        targ_sims2 = [self.sim_set_diff(w, aw1, aw2) for w in tw2]
        test_stat = sum(targ_sims1) - sum(targ_sims2)
        return(test_stat)
    
    def _weat_pval(self, tw1, tw2, aw1, aw2):
        '''calculate a p-value for WEAT by observing the probablity of a result of that size relative to all possible
        groupings of the words in tw1 and tw2'''
        test_stat = self._weat_test(tw1, tw2, aw1, aw2)
        observed_over = []
        all_targets = tw1 + tw2
        for c in combinations(all_targets, len(tw1)):
            not_c = filterfalse(lambda x: x in c, all_targets)
            stat = self._weat_test(c, not_c, aw1, aw2)
            observed_over.append(stat > test_stat)
        return(np.mean(observed_over))
    
    def _weat_effect_size(self, tw1, tw2, aw1, aw2):
        '''calculate an effect size for WEAT by normalizing (using mean and standard deviation) the individual
        differences calculated by sim_set_diff'''
        targ_sims1 = np.array([self.sim_set_diff(w, aw1, aw2) for w in tw1])
        targ_sims2 = np.array([self.sim_set_diff(w, aw1, aw2) for w in tw2])
        numerator = np.mean(targ_sims1) - np.mean(targ_sims2)
        denominator = np.std(np.concatenate([targ_sims1, targ_sims2]))
        return(numerator/denominator)
    
    def weat(self, targ_words1, targ_words2, attr_words1, attr_words2):
        '''print WEAT effect size and p-value for provided targets and attributes'''
        tw1, tw2, aw1, aw2 = self.check_vocab_sets(targ_words1, targ_words2, attr_words1, attr_words2)
        effect_size = self._weat_effect_size(tw1, tw2, aw1, aw2)
        print(f"Effect size: {effect_size}")
        pval = self._weat_pval(tw1, tw2, aw1, aw2)
        return(f"P-value: {pval}")
    
    