# Models for word alignment

class TranslationModel:
    "Models conditional distribution over trg words given a src word."

    def __init__(self, src_corpus, trg_corpus):
        self._src_trg_counts = {} # Statistics
        self._trg_given_src_probs = {} # Parameters

    def get_conditional_prob(self, src_token, trg_token):
        "Return the conditional probability of trg_token given src_token."
        if src_token not in self._trg_given_src_probs:
            return 1.0
        if trg_token not in self._trg_given_src_probs[src_token]:
            return 1.0
        return self._trg_given_src_probs[src_token][trg_token]

    def collect_statistics(self, src_tokens, trg_tokens, posterior_matrix):
        "Accumulate fractional alignment counts from posterior_matrix."
        assert len(posterior_matrix) == len(trg_tokens)
        for trg_index, trg_token in enumerate(trg_tokens):                          #<---
            posterior = posterior_matrix[trg_index]                                 #<---
            assert len(posterior) == len(src_tokens)
            for src_index, src_token in enumerate(src_tokens):                      #<---
                if src_token not in self._src_trg_counts:                           #<---
                    self._src_trg_counts[src_token] = {}                            #<---
                if trg_token not in self._src_trg_counts[src_token]:                #<---
                    self._src_trg_counts[src_token][trg_token] = 0                  #<---
                self._src_trg_counts[src_token][trg_token] += posterior[src_index]  #<---

    def recompute_parameters(self):
        "Reestimate parameters from counts then reset counters"
        for src_token, src_counts in self._src_trg_counts.items():                  #<---
            src_count = sum(src_counts.values())                                    #<---
            self._trg_given_src_probs[src_token] = {}                               #<---
            for trg_token, trg_count in src_counts.items():                         #<---
                self._trg_given_src_probs[src_token][trg_token] = trg_count / src_count #<---
        self._src_trg_counts = {}                                                   #<---

class PriorModel:
    "Models the prior probability of an alignment given only the sentence lengths and token indices."

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for more sophisticated models."
        self._distance_counts = {}
        self._distance_probs = {}

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a uniform prior probability."
        return 1.0 / src_length

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Extract the necessary statistics from this matrix if needed."
        assert len(posterior_matrix) == trg_length                                  #<---
        for trg_index, posterior in enumerate(posterior_matrix):                    #<---
            assert len(posterior) == src_length                                     #<---
            for src_index, prob in enumerate(posterior):                            #<---
                if src_index not in self._distance_counts:                          #<---
                    self._distance_counts[src_index] = {}                           #<---
                if trg_index not in self._distance_counts[src_index]:               #<---
                    self._distance_counts[src_index][trg_index] = 0                 #<---
                self._distance_counts[src_index][trg_index] += prob                 #<---
            
    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        for src_index, src_index_counts in self._distance_counts.items():           #<---
            src_index_count = sum(src_index_counts.values())                        #<---
            self._distance_probs[src_index] = {}                                    #<---
            for trg_index, trg_index_count in src_index_counts.items():             #<---
                self._distance_probs[src_index][trg_index] = trg_index_count / src_index_count #<---
        self._distance_counts = {}                                                  #<---
