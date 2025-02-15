import numpy as np
import logging
import copy

logger = logging.getLogger('__main__')


class NaiveRewardShaping:
    def __init__(self, subg_serieses, gamma, eta):
        print("Naive Reward Shaping!")
        self.subg_serieses = [[{'pos_x':0.2, 'pos_y': 0.9, 'rad': 0}]] + subg_serieses
        self.gamma = gamma
        self.curr_potential = 0
        self.curr_subgoal_index = (0, 0)
        self.eta = eta

    def reset(self):
        self.curr_val = 0
        self.curr_subgoal_index = (0, 0)

    def value(self, pre_obs, pre_a, reward, obs, done):
        potential = self.get_potential(obs)
        prev_potential = self.curr_potential
        self.curr_potential = potential
        return self.gamma * potential - prev_potential

    def at_subgoal(self, obs, subgoal):
        # サブゴールの領域に到達したかどうか？状態は[x, y, vx, vy]だから位置情報だけを用いる。
        return (
            np.linalg.norm(np.array(obs[:2]) - np.array([subgoal['pos_x'], subgoal['pos_y']]))
                < subgoal['rad']
        )

    def get_subgoal(self, state, subg_series):
        for subg in subg_series:
            if self.at_subgoal(state, subg):
                return subg_series.index(subg)
        return -1

    def get_potential(self, state):
        self.curr_val = 0
        if self.curr_subgoal_index == (0, 0):
            for i, subgoal_series in enumerate(self.subg_serieses):
                if self.at_subgoal(state, subgoal_series[0]):
                    logger.debug(f"Hit subgoal {state}")
                    self.curr_subgoal_index = (i, 0)
                    # print(f"subgoal_index: {self.curr_subgoal_index}")
                    # print(f"potential: {self.eta}, prev: {self.curr_potential}")
                    return self.eta
        else:
            subgoal_series = self.subg_serieses[self.curr_subgoal_index[0]]
            j = self.curr_subgoal_index[1] + 1
            if len(subgoal_series) <= j:
                return 0
            if self.at_subgoal(state, subgoal_series[j]):
                logger.debug(f"Hit subgoal {state}")
                self.curr_subgoal_index = (self.curr_subgoal_index[0], j)
                # print(f"subgoal_index: {self.curr_subgoal_index}")
                # print(f"potential: {self.eta}, prev: {self.curr_potential}")
                return self.eta
        return 0


class SubgoalSarsaRewardShaping:
    def __init__(self, subg_serieses, gamma, gamma_v, lr_v, fourier_basis):
        self.subg_serieses = [[{'pos_x':0.2, 'pos_y': 0.9, 'rad': 0}]] + subg_serieses
        self.gamma = gamma
        self.gamma_v = gamma_v
        self.lr_v = lr_v
        self.cur_index = 0
        self.series_index = 0
        self.is_reach_subgoal = False
        self.fourier_basis = fourier_basis
        self.subg_features = [[self.init_feature(subg_conf) for subg_conf in subg_series] for subg_series in self.subg_serieses]
        self.internal_reward = 0
        self.pre_subg = (0, 0) # subg_seriesesの(y, x)
        n_features = self.fourier_basis.getNumBasisFunctions()
        self.critic = SubgoalCritic(n_features, gamma_v, lr_v)
        self.l_subepisodes = 0

    def init_feature(self, subg_conf):
        subg_feature = {"count": 0}
        subg_feature["value"] = self.fourier_basis([subg_conf['pos_x'], subg_conf['pos_y'], 0, 0])
        return subg_feature

    def reset(self):
        self.series_index = 0
        self.is_reach_subgoal = False
        self.cur_index = 0
        self.pre_subg = (0, 0)
        self.internal_reward = 0
        self.l_subepisodes = 0

    def value(self, pre_obs, pre_a, reward, obs, done):
        cur_subg = self.aggregate_state(obs)
        h_reward = self.get_high_level_reward(reward, done)
        feat = self.get_feature(cur_subg)
        pre_feat = self.get_feature(self.pre_subg)
        if done and h_reward == 0:
            print(f"h_reward: {h_reward}, t: {self.l_subepisodes}")
        if h_reward != 0 or cur_subg != self.pre_subg:
            self.critic.update(pre_feat, reward, h_reward, feat, done, self.l_subepisodes)
            self.l_subepisodes = 0
        prev_potential = self.critic.value(pre_feat)
        potential = self.critic.value(feat)
        if self.is_reach_subgoal or done:
            print(f"previous: {self.pre_subg}, current: {cur_subg}, h_reward: {h_reward:.3f}")
            print(f"prev_potential: {int(prev_potential)}, potential: {int(potential)}, shaping: {int(self.gamma*potential - prev_potential)}")    
            print("-----------------------------------------------")

        self.pre_subg = cur_subg
        # print(f"prev_potential: {prev_potential}, potential: {potential}")
        return self.gamma * potential - prev_potential

    def aggregate_state(self, obs):
        next_index = self.cur_index + 1
        if self.series_index == 0:
            # サブゴールに一度も到達していない状態elf.cur_index)
            for index, subg_series in enumerate(self.subg_serieses[1:]):
                subg_index = self.get_subg_index(obs, subg_series)
                if subg_index != -1:
                    self.series_index = index + 1
                    # print(f"1. Reach a subgoal, abstract state: {(self.series_index, self.cur_index)}")
                    self.is_reach_subgoal = True
                    self.cur_index = subg_index
                    return (self.series_index, self.cur_index)
        elif len(self.subg_serieses[self.series_index]) > next_index:
            # サブゴールに一度到達したあとの判定
            subg_index = self.get_subg_index(obs, self.subg_serieses[self.series_index],
                                             next_index)
            if subg_index != -1:
                self.cur_index = subg_index
                # print(f"2. Reach a subgoal, abstract state: {(self.series_index, self.cur_index)}")
                self.is_reach_subgoal = True
                return (self.series_index, self.cur_index)
        self.is_reach_subgoal = False
        return (self.series_index, self.cur_index)

    def get_subg_index(self, obs, subg_series, start=0):
        # サブゴールを達成する順序に自由（sg1->sg3を許す）
        # for index, subg in enumerate(subg_series[start:]):
        #     if self.at_subgoal(obs, subg):
        #         return start + index
        # return -1
        # sg1->sg3を許さない。sg1->sg2のみ。
        if self.at_subgoal(obs, subg_series[start]):
            return start
        return -1

    def at_subgoal(self, obs, subgoal):
        # サブゴールの領域に到達したかどうか？状態は[x, y, vx, vy]だから位置情報だけを用いる。
        return (
            np.linalg.norm(np.array(obs[:2]) - np.array([subgoal['pos_x'], subgoal['pos_y']]))
                < subgoal['rad']
        )

    def get_feature(self, subg):
        subg_feat = self.subg_features[subg[0]][subg[1]]
        return subg_feat["value"]

    def update_feature(self, obs, subg):
        feature = self.fourier_basis(obs)
        subg_feat = self.subg_features[subg[0]][subg[1]]
        # 平均の逐次計算
        subg_feat["value"] += (subg_feat["count"] * subg_feat["value"] + feature) / (subg_feat["count"] + 1)
        subg_feat["count"] += 1

    def get_high_level_reward(self, reward, done):
        # サブゴールに達成するまでは加算し続ける。サブゴールに到達した時点でフィードバックを累計報酬として返す。
        # return reward
        # if reward < 0:
        #     reward = 0
        self.internal_reward += self.gamma**self.l_subepisodes * reward
        self.l_subepisodes += 1
        # self.internal_reward += reward
        if done:
            # 順番にすべてのサブゴールを達成してからゴールに達成した場合に報酬
            # import pdb; pdb.set_trace()
            next_index = self.cur_index + 1
            if self.series_index != 0 and len(self.subg_serieses[self.series_index]) <= next_index:
                r = self.internal_reward
                self.internal_reward = 0
                return r
            else:
                return 0

        if self.is_reach_subgoal:
            # ゴールに到達した場合
            r = self.internal_reward
            self.internal_reward = 0
            return r
        else:
            return 0


class SubgoalCritic:
    def __init__(self, n_features, gamma, lr):
        self.w = np.zeros(n_features)
        self.gamma = gamma
        self.lr = lr
    
    def update(self, feat, reward, h_reward, next_feat, done, t):
        lr = self.lr/np.linalg.norm(feat)
        update_target = h_reward
        if not done:
            update_target += self.gamma**t * self.value(next_feat)
        td_error = update_target - self.value(feat)
        self.w += lr * td_error * feat
        print(f"h_reward: {h_reward}, t: {t}")
    
    def value(self, feat):
        return np.dot(self.w, feat)
        # if v > 0:
        #     return v
        # else:
        #     return 0


class SubgoalPotentialRewardShaping:
    def __init__(self, subgoals, gamma, eta, rho):
        self.subgoals = subgoals  # [[{'pos_x':, 'pos_y':, 'rad':float},[{'pos':()}]]
        self.gamma = gamma
        self.eta = eta
        self.rho = rho
        self.is_reach_subgoal = False
        self.next_subgoals = self.init_next_subgoals(subgoals)  # [{"index": (), "content": {'pos_x",...}}]
        self.potential_basis = 0
        self.potential = 0

    def init_next_subgoals(self, subgoals):
        next_subgoals = []
        for i, subgoal in enumerate(subgoals):
            j = 0
            next_subgoals.append({
                "index": (i, j),
                "content": subgoal[j]})
        return next_subgoals

    def achieve(self, index):
        logger.info(f"Achieve the subgoal: {self.subgoals[index[0]][index[1]]}")
        if index[1] + 1 < len(self.subgoals[index[0]]):
            subgoal = self.subgoals[index[0]][index[1] + 1]
            self.next_subgoals = [{"index": (index[0], index[1]+1),
                                   "content": subgoal}]
        else:
            self.next_subgoals = []

    def value(self, obs, t, next_obs, next_t):
        # TODO self.potential(obs, t)、ここの値が間違い
        # potential = self.elliptic_potential(obs, t)
        # next_potential = self.elliptic_potential(next_obs, next_t)
        next_potential = self.reduction_potential(next_obs, next_t)
        # print(potential, next_potential)
        value = self.gamma * next_potential - self.potential
        if value != 0:
            logger.debug(f"Value is not zero, potential: {self.potential} and next: {next_potential}")
        self.potential = next_potential
        # print(f"next_potential: {next_potential}, potential: {potential}")
        return value

    def reset(self):
        self.is_reach_subgoal = False
        self.next_subgoals = self.init_next_subgoals(self.subgoals)
        self.potential_basis = 0
        self.potential = 0

    def reduction_potential(self, obs, t):
        if self.is_subgoal(obs):
            return self.eta * self.potential_basis
        else:
            return max(self.eta * self.potential_basis - self.rho * t, 0)

    def elliptic_potential(self, obs, t):
        if self.is_subgoal(obs):
            # サブゴール達成時はtを考慮しない。
            phi = self.eta * self.potential_basis
        else:
            basis = 1 - t**2 / self.rho**2
            if basis > 0:
                phi = self.eta * self.potential_basis * np.sqrt(basis)
            else:
                phi = 0
        return phi

    def is_subgoal(self, obs):
        # サブゴールの判定
        for subgoal in self.next_subgoals:
            if self.at_subgoal(obs, subgoal["content"]):
                self.achieve(subgoal["index"])
                self.is_reach_subgoal = True
                self.potential_basis += 1
                return True
        self.is_reach_subgoal = False
        return False

    def at_subgoal(self, obs, subgoal):
        # サブゴールの領域に到達したかどうか？状態は[x, y, vx, vy]だから位置情報だけを用いる。
        return (
            np.linalg.norm(np.array(obs[:2]) - np.array([subgoal['pos_x'], subgoal['pos_y']]))
                < subgoal['rad']
        )

    def get_is_reach_subgoal(self):
        return self.is_reach_subgoal
