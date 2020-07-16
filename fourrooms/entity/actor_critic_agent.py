import numpy as np 
from entity.policy import SoftmaxPolicy, FixedActionPolicy
from entity.tabular import Tabular


class ActorCriticAgent:
    def __init__(self, discount, lr_critic, lr_policy, nfeatures, nactions, temperature, rng):
        self.policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)
        self.critic = QLearning(discount, lr_critic, self.policy.weights)
        self.policy_improvement = PolicyGradient(self.policy, lr_policy)
        self.features = Tabular(nfeatures)
    
    def act(self, state):
        return self.policy.sample(self.features(state))
    
    def update(self, state, action, reward, done):
        phi = self.features(state)
        critic_feedback = self.critic.update(phi, action, reward, done)
        # if args.baseline:
        critic_feedback -= self.critic.value(phi, action)
        self.policy_improvement.update(phi, action, critic_feedback)
    
    def start(self, state, action):
        phi = self.features(state)
        self.critic.start(phi, action)


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

    def start(self, phi, action):
        self.last_phi = phi
        self.last_action = action
        self.last_value = self.value(phi, action)

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

    def update(self, phi, action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi)
            update_target += self.discount*(np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_action] += self.lr*tderror

        if not done:
            self.last_value = current_values[action]
        self.last_action = action
        self.last_phi = phi

        return update_target