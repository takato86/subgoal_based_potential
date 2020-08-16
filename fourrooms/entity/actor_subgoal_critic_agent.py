import numpy as np 
import math
import logging
from entity.policy import SoftmaxPolicy, FixedActionPolicy, SoftmaxStepPolicy
from entity.tabular import Tabular
from entity.reward_shaping import CumulativeSubgoalRewardWithPenalty, NaiveSubgoalRewardShaping
import csv

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ActorSubgoalCriticAgent:
    def __init__(self, discount, eta, lr_critic, lr_policy, nfeatures, nactions, temperature, rng, subgoals, rho=0):
        self.discount = discount
        self.policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)
        self.critic = QLearning(discount, lr_critic, self.policy.weights)
        self.policy_improvement = PolicyGradient(self.policy, lr_policy)
        self.features = Tabular(nfeatures)
        # self.subgoal_reward = NegativeSubgoalRewardShaping(discount, eta, subgoals, nfeatures, rho)
        self.subgoal_reward = CumulativeSubgoalRewardWithPenalty(discount, eta, subgoals, nfeatures, rho)
        # self.subgoal_reward = SubgoalTemporalReward(discount, eta, subgoals, nfeatures)
        # self.subgoal_reward = NaiveSubgoalRewardShaping(discount, eta, subgoals, nfeatures)
        self.eta = eta
        self.total_shaped_reward = 0

    def act(self, state):
        return self.policy.sample(self.features(state))

    def update(self, state, action, next_state, reward, done):
        phi = self.features(state)
        next_phi = self.features(next_state)
        reward += self.subgoal_reward.value(next_state, done)
        # if reward != 0:
        #     logger.info("Reward value is positive.")
        #     logger.info(f"state: {state}, reward value: {reward}")
        critic_feedback = self.critic.update(phi, action, next_phi, reward, done)
        # if args.baseline:
        critic_feedback -= self.critic.value(phi, action)
        self.policy_improvement.update(phi, action, critic_feedback)
        self.total_shaped_reward += reward

    # def start(self, state, action):
    #     phi = self.features(state)
    #     self.critic.start(phi, action)
    
    def export(self):
        self.subgoal_reward.export("res/potential_values.csv")


class PolicyGradient:
    def __init__(self, policy, lr):
        self.lr = lr
        self.policy = policy

    def update(self, phi, action, critic):
        actions_pmf = self.policy.pmf(phi)
        self.policy.weights[phi, :] -= self.lr*critic*actions_pmf
        self.policy.weights[phi, action] += self.lr*critic


class QLearning:
    def __init__(self, discount, lr, weights):
        self.lr = lr
        self.discount = discount
        self.weights = weights

    # def start(self, phi, action):
    #     self.last_phi = phi
    #     self.last_action = action
    #     self.last_value = self.value(phi, action)

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
            current_values = self.value(next_phi)
            update_target += self.discount*(np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.value(phi, action)
        self.weights[phi, action] += self.lr*tderror

        # if not done:
        #     self.last_value = current_values[action]
        # self.last_action = action
        # self.last_phi = phi

        return update_target