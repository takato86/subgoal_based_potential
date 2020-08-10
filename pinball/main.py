import argparse
import time
import gym
import logging
from datetime import datetime
import os
from gym import wrappers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from entity.ac_agent import SubgoalACAgent, ActorCriticAgent
import gym_pinball
from tqdm import tqdm, trange
from visualizer import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_csv(file_path, file_name, array):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    array = pd.DataFrame(array)
    saved_path = os.path.join(file_path, file_name)
    array.to_csv(saved_path)


def moved_average(data, window_size):
    b = np.ones(window_size)/window_size
    return np.convolve(data, b, mode='same')


def load_subgoals(file_path):
    subgoals_df = pd.read_csv(file_path)
    subgoals = subgoals_df.groupby(["user_id", "task_id"]).agg(list)
    xs = subgoals["x"].values.tolist()
    ys = subgoals["y"].values.tolist()
    rads = subgoals["rad"].values.tolist()
    subg_serieses = []
    for x, y, rad in zip(xs, ys, rads):
        subg_series = []
        for x_i, y_i, rad_i in zip(x, y, rad):
            subg_series.append({
                "pos_x": x_i,
                "pos_y": y_i,
                "rad": rad_i
                })
        subg_serieses.append([subg_series])
    return subg_serieses


def learning_loop(run, env_id, episode_count, model, visual, exe_id, rho, eta, subgoals, l_id):
    logger.info(f"start run {run}")
    subg_confs = list(itertools.chain.from_iterable(subgoals))
    env = gym.make(env_id, subg_confs=subg_confs)
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = SubgoalACAgent(run, env.action_space, env.observation_space, rho=rho, eta=eta, subgoals=subgoals)
    # agent = ActorCriticAgent(run, env.action_space, env.observation_space)
    vis = Visualizer(["ACC_X", "ACC_Y", "DEC_X", "DEC_Y", "NONE"])
    if model:
        agent.load_model(model)
    reward = 0
    done = False
    total_reward_list = []
    steps_list = []
    max_q_list = []
    max_q_episode_list = []
    runtimes = []
    max_q = 0.0
    date = datetime.now().strftime("%Y%m%d")
    # time = datetime.now().strftime("%H%M")
    saved_dir = os.path.join("data", date)  # , time
    start_time = time.time()
    for i in trange(episode_count):
        total_reward = 0
        total_shaped_reward = 0
        n_steps = 0
        ob = env.reset()
        action = agent.act(ob)
        pre_action = action
        is_render = False
        while True:
            if (i+1) % 20 == 0 and visual:
                env.render()
                is_render = True
            pre_obs = ob
            ob, reward, done, _ = env.step(action)
            # TODO
            reward = 0 if reward < 0 else reward
            n_steps += 1
            # rand_basis = np.random.uniform()
            pre_action = action
            action = agent.act(ob)
            shaped_reward = agent.update(pre_obs, pre_action, reward, ob, action, done)
            total_reward += reward
            total_shaped_reward += shaped_reward
            tmp_max_q = agent.get_max_q(ob)
            max_q_list.append(tmp_max_q)
            max_q = tmp_max_q if tmp_max_q > max_q else max_q
            if done:
                logger.info("episode: {}, steps: {}, total_reward: {}, total_shaped_reward: {}, max_q: {}, max_td_error: {}"
                        .format(i, n_steps, total_reward, int(total_shaped_reward), int(max_q), int(agent.get_max_td_error())))
                total_reward_list.append(total_reward)
                steps_list.append(n_steps)
                break
            if is_render:
                vis.set_action_dist(agent.vis_action_dist, action)
                vis.pause(.0001)
        max_q_episode_list.append(max_q)
        saved_model_dir = os.path.join(saved_dir, 'model')
        agent.save_model(saved_model_dir, i)
    # export process
    runtimes.append(time.time() - start_time)
    saved_res_dir = os.path.join(saved_dir, 'res')
    export_csv(saved_res_dir, f"{exe_id}_total_reward_{l_id}_{run}_eta={eta}_rho={rho}.csv", total_reward_list)
    td_error_list = agent.td_error_list
    export_csv(saved_res_dir, f"{exe_id}_td_error_{l_id}_{run}_eta={eta}_rho={rho}.csv", td_error_list)
    total_reward_list = np.array(total_reward_list)
    steps_list = np.array(steps_list)
    max_q_list = np.array(max_q_list)
    logger.info("Average return: {}".format(np.average(total_reward_list)))
    steps_file_path = os.path.join(saved_res_dir, f"{exe_id}_steps_{l_id}_{run}_eta={eta}_rho={rho}.csv")
    pd.DataFrame(steps_list).to_csv(steps_file_path)
    runtime_file_path = os.path.join(saved_res_dir, f"{exe_id}_runtime_{l_id}_{run}_eta={eta}_rho={rho}.csv")
    pd.DataFrame(runtimes, columns=["runtime"]).to_csv(runtime_file_path)
    # save model
    # saved_model_dir = os.path.join(saved_dir, 'model')
    # agent.save_model(saved_model_dir)
    env.close()


def main():
    learning_time = time.time()
    rhos = [0]
    etas = [5000]
    # rhos = [1e-02, 1e-03, 1e-04, 1e-05]
    # etas = [1, 10, 100, 1000, 10000]
    if len(args.subg_path) == 0:
        logger.info("Nothing subgoal path.")
        subg_serieses = [[[{"pos_x":0.512, "pos_y": 0.682, "rad":0.04}, {"pos_x":0.683, "pos_y":0.296, "rad":0.04}]]] # , {"pos_x":0.9 , "pos_y":0.2 ,"rad": 0.04}
    else:
        subg_serieses = load_subgoals(args.subg_path)

    for rho in rhos:
        for eta in etas:
            for l_id, subg_series in enumerate(subg_serieses):
                logger.info(f"learning: {l_id}/{len(subg_serieses)}")
                logger.info(f"subgoals: {subg_series}")
                logger.info(f"rho: {rho}, eta: {eta}")
                for run in range(args.nruns):
                    learning_loop(run, args.env_id, args.nepisodes, args.model,
                                  args.vis, args.id, rho, eta, subg_series, l_id)
                    # Close the env and write monitor result info to disk
    duration = time.time() - learning_time
    logger.info("Learning time: {}m {}s".format(int(duration//60), int(duration%60)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Actor-Critic Learning.')
    parser.add_argument('env_id', nargs='?', default='Pinball-Subgoal-v0', help='Select the environment to run.')
    parser.add_argument('--vis', action='store_true', help='Attach when you want to look visual results.')
    parser.add_argument('--model', help='Input model dir path')
    parser.add_argument('--nepisodes', default=250, type=int)
    parser.add_argument('--nruns', default=25, type=int)
    parser.add_argument('--id', default='', type=str)
    parser.add_argument('--subg-path', default='', type=str)
    args = parser.parse_args()
    main()
