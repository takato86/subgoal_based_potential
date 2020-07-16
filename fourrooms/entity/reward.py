from gym_fourrooms.envs.shaping_reward import ShapingRewardBase
import csv


class ShapingReward(ShapingRewardBase):
    def __init__(self, gamma, subgoal_values):
        super(ShapingReward, self).__init__()
        self.gamma = gamma
        self.subgoal_values = subgoal_values

    def perform(self, state, next_state):
        phi = 0
        phi_dash = 0
        if state in self.subgoal_values:
            phi = self.subgoal_values[state]
        if next_state in self.subgoal_values:
            phi_dash = self.subgoal_values[next_state]
        return self.gamma * phi_dash - phi
