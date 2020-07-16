#_*_coding: utf-8 _*_

import numpy as np
from scipy.special import expit
from scipy.special import logsumexp

class SigmoidTermination:
    def __init__(self, rng, nfeatures):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi

class OneStepTermination:
    def sample(self, phi):
        return 1

    def pmf(self, phi):
        return 1.
    
class FixedTermination:
    def __init__(self, rng, nfeatures):
        self.rng = rng
        self.weights = np.array([0.5] * nfeatures) # TODO
    def pmf(self, phi):
        return self.weights[phi]
    def sample(self, phi):
        return int(self.rng.uniform() < self.weights[phi])
    def grad(self, phi):
        return 0, phi

class HybridTermination:
    def __init__(self, rng, nfeatures, fixed_range):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))
        self.fixed_range = np.array(list(fixed_range))

    def pmf(self, phi):
        if phi in self.fixed_range:
            return self.weights[phi]
        else:
            return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        if phi in self.fixed_range:
            return self.weights[phi]
        else:
            return int(self.rng.uniform() < self.pmf(phi))
        
    def grad(self, phi):
        if phi in self.fixed_range:
            terminate = self.pmf(phi)
            return terminate*(1. - terminate), phi
        else:
            return 0, phi