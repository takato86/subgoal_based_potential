import pytest
from entity import CumulativeSubgoalRewardWithPenalty


@pytest.fixture
def reward_shaping():
    kwargs = {"discount": 0.99, "eta": 0.5, "subgoal_serieses": [[25, 65]],
              "nfeatures": 1}
    return CumulativeSubgoalRewardWithPenalty(**kwargs)


def test_at_subgoal(reward_shaping):
    assert reward_shaping.at_subgoal(25)
    assert not reward_shaping.at_subgoal(26)


def test_init_subgoals(reward_shaping):
    assert reward_shaping.init_next_subgoals([[25, 65]]) == {25: (0, 0)}


def test_achieve(reward_shaping):
    index = [0, 0]
    assert reward_shaping.achieve(index) == {65: (0, 1)}


def test_penalty(reward_shaping):
    rs = reward_shaping
    assert rs.penalty() == 0
    rs.value(1, False)
    assert rs.penalty() == 100*1


def test_constant_value(reward_shaping):
    rs = reward_shaping
    assert rs.constant_value() == 0
    rs.value(1, False)
    assert rs.constant_value() == 0
    rs.value(25, False)
    assert rs.constant_value() == 0.5
    rs.value(1, False)
    assert rs.constant_value() == 0.5
