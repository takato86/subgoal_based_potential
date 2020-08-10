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
    assert not reward_shaping.is_subgoal(obs1)
    assert reward_shaping.is_subgoal(obs2)
    assert not reward_shaping.is_subgoal(obs2)
    assert reward_shaping.is_subgoal(obs3)


def test_is_subgoal_case2(reward_shaping):
    obs2 = [0.58, 0.64, 0, 0]
    obs3 = [0.64, 0.30, 0, 0]
    assert reward_shaping.potential_basis == 0
    assert not reward_shaping.is_subgoal(obs3)
    assert reward_shaping.is_subgoal(obs2)
    assert reward_shaping.potential_basis == 1
    assert not reward_shaping.is_subgoal(obs2)
    assert reward_shaping.is_subgoal(obs3)
    assert not reward_shaping.is_subgoal(obs3)
    assert reward_shaping.potential_basis == 2


def test_reest(reward_shaping):
    obs2 = [0.58, 0.64, 0, 0]
    reward_shaping.is_subgoal(obs2)
    reward_shaping.reset()
    assert reward_shaping.next_subgoals == [{"index": (0, 0), "content": {"pos_x": 0.59,"pos_y": 0.65,"rad": 0.04}}]
    assert not reward_shaping.is_reach_subgoal
    assert reward_shaping.potential_basis == 0
