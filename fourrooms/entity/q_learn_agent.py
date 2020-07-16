import numpy as np
import math
import logging
from .policy import SoftmaxPolicy, EgreedyPolicy
from .tabular import Tabular
from .reward_shaping import CumulativeSubgoalRewardWithPenalty,\
                            NaiveSubgoalRewardShaping,\
                            SubgoalThroughPotentialRewardShaping

import csv

logger = logging.getLogger(__name__)


class QLearningAgent:
    def __init__(self, discount, epsilon, lr, nfeatures, nactions, temperature, rng, q_value={}):
        self.discount = discount
        self.policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)
        self.critic = QLearning(discount, lr, self.policy.weights)
        self.features = Tabular(nfeatures)
        self.total_shaped_reward = 0
        for state, value in q_value.items():
            phi = self.features(state)
            self.critic.initialize(phi, value)

    def act(self, state):
        return self.policy.sample(self.features(state))
    
    def update(self, state, action, next_state, reward, done):
        phi = self.features(state)
        next_phi = self.features(next_state)
        _ = self.critic.update(phi, action, next_phi, reward, done)


class SubgoalRSQLearningAgent(QLearningAgent):
    def __init__(self, discount, epsilon, lr, nfeatures, nactions, temperature, rng, subgoals, eta, rho=0):
        super(SubgoalRSQLearningAgent, self).__init__(discount, epsilon, lr, nfeatures, nactions, temperature, rng)
        # self.reward_shaping = CumulativeSubgoalRewardWithPenalty(discount, eta, subgoals, nfeatures, rho)
        # self.reward_shaping = NaiveSubgoalRewardShaping(discount, eta, subgoals, nfeatures)
        self.reward_shaping = SubgoalThroughPotentialRewardShaping(discount, eta, subgoals, nfeatures, rho) 
        
    def update(self, state, action, next_state, reward, done):
        phi = self.features(state)
        next_phi = self.features(next_state)
        reward += self.reward_shaping.value(state, done)
        self.total_shaped_reward += reward
        _ = self.critic.update(phi, action, next_phi, reward, done)


class QLearning:
    def __init__(self, discount, lr, weights):
        self.lr = lr
        self.discount = discount
        self.weights = weights
    
    def initialize(self, phi, q_value, action=None):
        if action is None:
            self.weights[phi, :] = q_value
        else:
            self.weights[phi, action] = q_value

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def advantage(self, phi, action=None):
        values = self.value(phi)
        advantages = values - np.max(values)
        if action is None:
            return advantages
        return advantages[action]

    def update(self, phi, action, next_phi, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            next_values = self.value(next_phi)
            update_target += self.discount * (np.max(next_values))
        # Dense gradient update step
        tderror = update_target - self.value(phi, action)
        self.weights[phi, action] += self.lr * tderror
        return update_target
