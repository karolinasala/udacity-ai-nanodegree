import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # DONE implement model selection based on BIC scores

        n = len(self.X[0])
        logN = np.log(self.X.shape[0])
        best_score = float("inf") #initialize best score
        best_model = None # Initialize the best model

        try:
            for component in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(component)
                p = component ** 2 + 2 * component * n - 1
                logL = model.score(self.X, self.lengths)
                # Calculate score
                bic = -2 * logL + p * logN
                if bic < best_score:
                    best_score = bic
                    best_model = model
        except:
            pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # DONE implement model selection based on DIC scores
        best_score = float('-inf')
        best_model = None
        try:
            for component in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(component)  # Train model
                score = model.score(self.X, self.lengths)  # log(P(X(i))
                avg_logL = 0
                sum_words = 0

                for word in self.words:
                    if word is not self.this_word:
                        X, lenghts = self.hwords[word]
                        score_word = model.score(X, lenghts)
                        avg_logL += score_word
                        sum_words += 1

                # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                dic = score - (1 / (sum_words - 1)) * avg_logL

                if dic > best_score:
                    best_score = dic
                    best_model = model

        except:
            pass
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        splitter = KFold()

        for component in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            try:
                for cv_train_idx, cv_test_idx in splitter.split(self.sequences):
                    # Generate training and test data
                    train_x, train_length = combine_sequences(cv_train_idx, self.sequences)
                    test_x, test_length = combine_sequences(cv_test_idx, self.sequences)
                    # Train and score model
                    cv_model = self.base_model(n_components=component).fit(train_x, train_length)
                    score = cv_model.score(test_x, test_length)
                    scores.append(score)

                #update best score
                mean_scores = np.mean(scores)
                if mean_scores > best_score:
                    best_score = mean_scores
                    self.n_constant = component
            except:
                pass

        return self.base_model(self.n_constant)