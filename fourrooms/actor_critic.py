import gym
from gym import wrappers
import numpy as np
import pandas as pd
import gym_fourrooms

import os

import pickle
import copy
import csv
import configparser
from entity.tabular import Tabular
from entity.sg_parser import parser
from entity.policy import EgreedyPolicy, SoftmaxPolicy, FixedActionPolicy
from entity.termination import OneStepTermination
from entity.reward import ShapingReward
import matplotlib.pyplot as plt

'''
avg_duration: 1つのOptionが続けられる平均ステップ数
step        : 1エピソードに要したステップ数
'''

inifile = configparser.ConfigParser()
inifile.read("../../config.ini")

class IntraOptionQLearning:
    def __init__(self, discount, lr, terminations, weights):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights

    def start(self, phi, option):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi, option)

    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)

    def advantage(self, phi, option=None):
        values = self.value(phi)
        advantages = values - np.max(values)
        if option is None:
            return advantages
        return advantages[option]

    def update(self, phi, option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_option] += self.lr*tderror

        if not done:
            self.last_value = current_values[option]
        self.last_option = option
        self.last_phi = phi

        return update_target

class HumanSubgoalTransfer:
    def __init__(self, lr, terminations, epoch):
        self.terminations = terminations
        self.lr = lr
        self.epoch = epoch
    
    def fit(self, landmarks, states):
        print(landmarks)
        for termination in self.terminations:
            for i in range(self.epoch):
                for state in states:
                    if state in landmarks:
                        grad, phi = termination.grad(state)
                        beta = termination.pmf(state)
                        termination.weights[state] += self.lr * (1.0 - beta) * grad * phi
                    else:
                        # TODO
                        grad, phi = termination.grad(state)
                        beta = termination.pmf(state)
                        termination.weights[state] += self.lr * (0.5 - beta) * grad * phi
    
    def eval(self, landmarks, states):
        rmse = 0
        mae = 0
        for termination in self.terminations:
            for state in states:
                beta = termination.pmf(state)
                if state in landmarks:
                    rmse += (1-beta)**2
                    mae += abs(1-beta)
                else:
                    rmse += (0-beta)**2
                    mae += abs(0-beta)
        mae = mae/len(states)
        rmse = (rmse/len(states))**0.5
        print("rmse: {}, mae: {}".format(rmse, mae))


def load_landmarks_from_csv(file_path):
    landmarks_df = pd.read_csv(file_path)
    landmarks = []
    landmark_list = list(set(landmarks_df["next_state1"]))
    for landmark in landmark_list:
        landmarks.append(np.array([landmark]))
    return landmarks

def load_landmarks_from_pickle(file_path):
    with open(file_path, "rb") as f:
        landmarks = pickle.load(f)
    return landmarks

def export_analysis_report(file_path, policy):
    # "exploit-and-explore.txt"
    export_lines = []
    with open(file_path, 'w', encoding='utf-8') as f:
        export_lines.append(f"exploit: {policy.exploit_count},\
explore: {policy.explore_count}\n")
        f.writelines(export_lines)
        
def export_state_values(file_path, env, policy):
    state_values = policy.get_values(env).tolist()
    with open(file_path, 'w', encoding='utf-8') as f:
        for state_value in state_values:
            f.write(",".join(list(map(str, state_value))) + "\n")

if __name__ == '__main__':
    parser.add_argument('--ac', action='store_true')
    parser.add_argument('--video', action='store_true')
    
    args = parser.parse_args()
 
    rng = np.random.RandomState(1234)
    env_to_wrap = gym.make(args.env_id)

    kind = "actor_critic"
    if args.video:
        movie_folder = f'res/{kind}/movies'
        if not os.path.exists(movie_folder):
            os.makedirs(movie_folder)
        env = wrappers.Monitor(env_to_wrap, movie_folder, force=True, video_callable=(lambda ep: ep%100 == 0 or (ep>1000 and ep%10==0 and ep<1100)))
    else:
        env = env_to_wrap

    if "Fourrooms" in args.env_id:
        initial_possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]
        possible_next_goals = [91, 90, 72, 89, 70, 99, 82, 103, 81, 93, 100, 79, 78, 102, 69, 101, 88, 92, 71, 80, 68]

    elif "Tworooms" in args.env_id:
        initial_possible_next_goals = list(range(6, 15)) + list(range(21, 30)) + list(range(36, 45)) + list(range(51, 60)) + list(range(66, 75))
        possible_next_goals = np.random.permutation(initial_possible_next_goals).tolist()

    elif "Oneroom" in args.env_id:
        initial_possible_next_goals = list(range(28, 33)) + list(range(39, 44)) + list(range(50, 55))
        possible_next_goals = np.random.permutation(initial_possible_next_goals).tolist()
    
    if "SubGoal" in args.env_id:
        with open(args.subgoal_path, 'r', encoding='utf-8') as f:
            dict_reader = csv.DictReader(f)
            subgoals = {}
            for row in dict_reader:
                subgoals[int(float(row['state1']))] = 5e-4
        env.set_subgoals(subgoals)
    
    if "Shaping" in args.env_id:
        state_values = []
        subgoal_values = {}
        with open("res/Fourrooms-v0-0-values.csv", "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                state_values.append(row)
        for i, row_state_values in enumerate(state_values):
            for j, state_value in enumerate(row_state_values):
                if env.tostate.get((i, j)) is not None:
                    subgoal_values[env.tostate[(i, j)]] = float(state_value)
        # import pdb; pdb.set_trace()
        # subgoal_values = {25:0.5, 51:0.5, 62:1.0, 88:1.0} # hall ways
        # subgoals = [62, 88] # horizon
        shaping_reward = ShapingReward(args.discount, subgoal_values)
        env.set_shaping_reward(shaping_reward)

    if "Flexible" in args.env_id:
        # 左上
        initial_possible_next_goals = [25]
        possible_next_goals = [25]
        env.set_init_states([(i, j) for i in range(1,6) for j in range(1,6)])
        env.set_goal((3,6))
        
        # 右上
        # initial_possible_next_goals = [62]
        # possible_next_goals = [62]
        # env.set_init_states([(i, j) for i in range(1,7) for j in range(7,12)])
        # env.set_goal((7, 9))

        # 左下
        # initial_possible_next_goals = [88]
        # possible_next_goals = [88]
        # env.set_init_states([(i, j) for i in range(7,11) for j in range(1,6)])
        # env.set_goal((10, 6))


    for run in range(args.nruns):
        rng = np.random.RandomState(run)
        features = Tabular(env.observation_space.n)

        nfeatures, nactions = len(features), env.action_space.n

        option_policies = [FixedActionPolicy(act, nactions) for act in range(nactions)]

        option_terminations = [OneStepTermination() for _ in range(nactions)]

        # E-greedy policy over options
        #policy = EgreedyPolicy(rng, nfeatures, args.noptions, args.epsilon)
        policy = SoftmaxPolicy(rng, nfeatures, nactions, args.temperature)

        kind = "actor_critic"
        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, policy.weights)
        initial_option_terminations = copy.deepcopy(option_terminations)
        # Learn Qomega separately
        action_weights = np.zeros((nfeatures, nactions, 1))
        steps = []
        subgoals = [[25, 62], [51, 88], [62], [88]]
        for episode in range(args.nepisodes):
            if episode == 1000:
                if len(possible_next_goals) > 0:
                    env.goal = possible_next_goals.pop()
                else:
                    env.goal = rng.choice(initial_possible_next_goals)
                print('************* New goal : ', env.goal)
            phi = features(env.reset())
            option = policy.sample(phi)
            action = option_policies[option].sample(phi)
            critic.start(phi, option)
            cumreward = 0.
            duration = 1
            option_switches = 0
            avgduration = 0.
            for step in range(args.nsteps):
                observation, reward, done, _ = env.step(action)
                phi = features(observation)
                if option_terminations[option].sample(phi):
                    option = policy.sample(phi)
                    option_switches += 1
                    avgduration += (1./option_switches)*(duration - avgduration)
                    duration = 1
                action = option_policies[option].sample(phi)
                # Critic update
                update_target = critic.update(phi, option, reward, done)
                cumreward += reward
                duration += 1
                if done:
                    break
            steps.append(step)
            print('Run {} episode {} steps {} cumreward {} avg. duration {} switches {}'.format(run, episode, step, cumreward, avgduration, option_switches))
            if episode == 999:
                file_name = os.path.basename(__file__) + "_" + \
                                f"{episode}_exploit-and-explore_before.txt"
                export_analysis_report(os.path.join("res",
                                                "analysis",
                                                file_name),
                                       policy)
                policy.reset_count()
            elif episode == 1999:

                file_name = os.path.basename(__file__) + "_" + \
                                f"{episode}_exploit-and-explore_after.txt"
                export_analysis_report(os.path.join("res",
                                                "analysis",
                                                file_name),
                                       policy)

        export_state_values(os.path.join("res", "values", f"{args.env_id}-{run}-values.csv"), env_to_wrap, policy)
        with open(os.path.join("res", "steps", f"{args.env_id}-{run}-steps.csv"), 'w') as f:
            f.write('\n'.join(list(map(str,steps))))
        # plt.plot(list(range(len(steps))), steps)
        # plt.show()
    