import gym
from gym import wrappers
import numpy as np
import pandas as pd
import gym_fourrooms
import logging
import os
import time
import configparser
from entity.sg_parser import parser
import matplotlib.pyplot as plt
from entity.actor_critic_agent import ActorCriticAgent
from entity.q_learn_agent import QLearningAgent, SubgoalRSQLearningAgent
from entity.actor_subgoal_critic_agent import ActorSubgoalCriticAgent
from entity.sarsa_agent import SubgoalRSSarsaAgent, SarsaRSSarsaAgent, SarsaAgent

'''
avg_duration: 1つのOptionが続けられる平均ステップ数
step        : 1エピソードに要したステップ数
'''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
inifile = configparser.ConfigParser()
inifile.read("../../config.ini")


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


def export_policy(file_path, env, policy):
    policy = policy.get_policy(env).tolist()
    with open(file_path, 'w', encoding='utf-8') as f:
        for action in policy:
            f.write(",".join(list(map(str, action))) + "\n")


def export_episodes(file_path, episodes):
    df = pd.DataFrame(episodes, columns=["episode", "step", "state", "action", "next_state"])
    df.to_csv(file_path)


def export_env(file_path, env):
    to_state = np.full(env.occupancy.shape, -1)
    for k, val in env.tostate.items():
        to_state[k[0]][k[1]] = val
    pd.DataFrame(to_state).to_csv(file_path)


def export_runtimes(file_path, runtimes):
    runtimes_df = pd.DataFrame(runtimes, columns=["runtime"])
    runtimes_df.to_csv(file_path)


def prep_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def load_subgoals(file_path, task_id=None):
    subgoal_df = pd.read_csv(file_path)
    if task_id is not None:
        subgoal_df = subgoal_df[subgoal_df["task_id"] == task_id]
    subg_serieses_df = subgoal_df.groupby(["user_id", "task_id"]).agg(list)
    subg_serieses = []
    for subg_series in list(subg_serieses_df['state'].values):
        subg_serieses.append([subg_series])
    return subg_serieses


def learning_loop(nruns, nepisodes, nsteps, discount, lr_term, lr_intra,
                  lr_critic, epsilon, eta, rho, video, id, env_id,
                  subgoals, temperature, learn_id):
    logger.info(f"Start learning in the case of eta={eta}, rho={rho}")
    rng = np.random.RandomState(1234)
    env_to_wrap = gym.make(env_id)
    episodes = []

    if video:
        movie_folder = prep_dir(os.path.join('res', 'movies', id))
        env = wrappers.Monitor(env_to_wrap, movie_folder, force=True,
                               video_callable=(lambda ep: ep%100 == 0 or (ep>30 and ep<35)))
    else:
        env = env_to_wrap

    export_env(os.path.join(env_dir, f"{env_id}.csv"), env_to_wrap) 
    runtimes = []
    for run in range(nruns):
        start_time = time.time()
        rng = np.random.RandomState(run)
        nfeatures, nactions = env.observation_space.n, env.action_space.n
        # subgoals = [[25, 62], [51, 88]]  # hallway-4
        # aggr_set = [upper_left, upper_right, lower_left, lower_right]
        # subgoal_values = [[100 for s in subgoal_series] for subgoal_series in subgoals]
        # subgoal_values = [[1, 10]]  
        subgoal_values = None
        agent = ActorSubgoalCriticAgent(discount, eta, lr_critic,
                                        lr_intra, nfeatures, nactions, temperature,
                                        rng, subgoals, rho)
        # agent = QLearningAgent(discount, epsilon, lr_critic, nfeatures,
        #                        nactions, temperature, rng)
        # agent = SubgoalRSQLearningAgent(discount, epsilon, lr_critic, nfeatures, nactions,
        #                                 temperature, rng, subgoals, eta, rho)
        # agent = SubgoalRSSarsaAgent(discount, epsilon, lr_critic, nfeatures, nactions,
        #                             temperature, rng, subgoals, eta, rho, subgoal_values=subgoal_values)
        # agent = SarsaRSSarsaAgent(discount, epsilon, lr_critic, nfeatures, nactions, temperature,
        #                           rng, aggr_set)
        # agent = SarsaAgent(discount, epsilon, lr_critic, nfeatures, nactions, temperature, rng)
        steps = []
        for episode in range(nepisodes):
            next_observation = env.reset()
            logger.info(f"start state: {next_observation}")
            cumreward = 0
            logger.info(f"goal is at {env.goal}")
            for step in range(nsteps):
                observation = next_observation
                action = agent.act(observation)
                next_observation, reward, done, _ = env.step(action)
                # Critic update
                agent.update(observation, action, next_observation, reward, done)
                episodes.append([episode, step, observation, action, next_observation])
                cumreward += reward
                if done:
                    print("true goal: {}, actual goal: {}, reward: {}"
                          .format(env.goal, next_observation, reward))
                    break
            steps.append(step)
            print('Run {} episode {} steps {} cumreward {}'
                  .format(run, episode, step, agent.total_shaped_reward))
        runtimes.append(time.time() - start_time)
        export_state_values(
            os.path.join(
                val_dir,
                f"{env_id}-{learn_id}-{run}-{id}-eta={eta}-rho={rho}.csv"),
            env_to_wrap,
            agent.policy
        )
        export_policy(
            os.path.join(
                policy_dir,
                f"{env_id}-{learn_id}-{run}-{id}-eta={eta}-rho={rho}.csv"),
            env_to_wrap,
            agent.policy
        )
        export_episodes(
            os.path.join(
                episode_dir,
                f"{env_id}-{learn_id}-{run}-{id}-eta={eta}-rho={rho}.csv"),
            episodes
        )
        episodes = []
        with open(os.path.join(steps_dir,
                               f"{env_id}-{learn_id}-{run}-{id}-eta={eta}-rho={rho}.csv"),
                  'w') as f:
            f.write('\n'.join(list(map(str, steps))))
    export_runtimes(
        os.path.join(
            runtimes_dir,
            f"{env_id}-{learn_id}-{id}-eta={eta}-rho={rho}.csv"
        ),
        runtimes
    )
    env.close()


def main():
    subgoals = load_subgoals(args.subgoal_path, task_id=1)
    logger.info(f"subgoals: {subgoals}")
    for learn_id, subgoal in enumerate(subgoals):
        learning_loop(args.nruns, args.nepisodes, args.nsteps, args.discount,
                      args.lr_term, args.lr_intra, args.lr_critic, args.epsilon,
                      args.eta, args.rho, args.video, args.id, args.env_id,
                      subgoal, args.temperature, learn_id)

def main_hypara():
    subgoal = [[25, 62]]
    logger.info(f"hyper parameter tuning.")
    etas = [0.1, 0.5, 1.0, 10]
    rhos = [0, 1e-5, 1e-3, 1e-1, 1]
    for i, eta in enumerate(etas):
        for j, rho in enumerate(rhos):
            learn_id = i*10 + j
            learning_loop(args.nruns, args.nepisodes, args.nsteps, args.discount,
                          args.lr_term, args.lr_intra, args.lr_critic, args.epsilon,
                          eta, rho, args.video, args.id, args.env_id,
                          subgoal, args.temperature, learn_id)


if __name__ == '__main__':
    parser.add_argument('--ac', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--id', default='no_name', type=str)
    parser.add_argument('--rho', default=0.05, type=float)
    args = parser.parse_args()

    upper_left = [
                    0, 1, 2, 3, 4, 
                    10, 11, 12, 13, 14,
                    20, 21, 22, 23, 24, 25,
                    31, 32, 33, 34, 35,
                    41, 42, 43, 44, 45, 51
    ]  
    upper_right = [
                    5, 6, 7 ,8, 9,
                    15, 16, 17, 18, 19,
                    26, 27, 28, 29, 30,
                    36, 37, 38, 39, 40,
                    46, 47, 48, 49, 50,
                    52, 53, 54, 55, 56, 62
    ]
    lower_left = [
                    57, 58, 59, 60, 61,
                    63, 64, 65, 66, 67,
                    73, 74, 75, 76, 77,
                    83, 84, 85, 86, 87, 88,
                    94, 95, 96, 97, 98
    ]
    lower_right = [
                    68, 69, 70, 71, 72,
                    78, 79, 80, 81, 82,
                    89, 90, 91, 92, 93,
                    99, 100, 101, 102, 103
    ]
    env_dir = prep_dir(os.path.join("res", "env"))
    val_dir = prep_dir(os.path.join("res", "values"))
    episode_dir = prep_dir(os.path.join("res", "episode"))
    policy_dir = prep_dir(os.path.join("res", "policy"))
    analysis_dir = prep_dir(os.path.join("res", "analysis"))
    steps_dir = prep_dir(os.path.join("res", "steps"))
    runtimes_dir = prep_dir(os.path.join("res", "runtime"))
    main()
    # main_hypara()