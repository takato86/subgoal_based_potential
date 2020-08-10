import pandas as pd
import matplotlib.pyplot as plt
import argparse
import gym
import gym_fourrooms


def load(path):
    in_df = pd.read_csv(path)
    return in_df


def main():
    subgoal_df = load(args.subgs)
    env = gym.make('Fourrooms-v0')
    layout = env.get_layout()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgs", type=str)
    args = parser.parse_args()
    main()
