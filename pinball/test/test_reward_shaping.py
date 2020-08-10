import pytest
from entity import SubgoalPotentialRewardShaping, FourierBasis


@pytest.fixture
def reward_shaping():
    subgoals = [[{
                "pos_x": 0.59,
                "pos_y": 0.65,
                "rad": 0.04},
                {
                "pos_x": 0.64,
                "pos_y": 0.32,
                "rad": 0.04
                }]]
    kwargs = {"subgoals": subgoals, "gamma": 0.99, "eta": 500, "rho":1e-08}
    return SubgoalPotentialRewardShaping(**kwargs)


def test_is_subgoal_case1(reward_shaping):
    obs1 = [0.2, 0.9, 0, 0]
    obs2 = [0.58, 0.64, 0, 0]
    obs3 = [0.64, 0.30, 0, 0]
    # import pdb; pdb.set_trace()
    assert reward_shaping.is_subgoal(obs1) == False
    assert reward_shaping.is_subgoal(obs2) == True
    assert reward_shaping.is_subgoal(obs2) == False
    assert reward_shaping.is_subgoal(obs3) == True


def test_is_subgoal_case2(reward_shaping):
    obs1 = [0.2, 0.9, 0, 0]
    obs2 = [0.58, 0.64, 0, 0]
    obs3 = [0.64, 0.30, 0, 0]
    assert reward_shaping.is_subgoal(obs3) == False
    assert reward_shaping.is_subgoal(obs2) == True
    assert reward_shaping.is_subgoal(obs2) == False
    assert reward_shaping.is_subgoal(obs3) == True
    assert reward_shaping.is_subgoal(obs3) == False



