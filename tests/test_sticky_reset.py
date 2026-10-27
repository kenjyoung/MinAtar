import numpy as np
import pytest

from minatar import Environment
from minatar.gym import BaseEnv


@pytest.mark.parametrize(
    "game", ["asterix", "breakout", "freeway", "seaquest", "space_invaders"]
)
@pytest.mark.parametrize("previous_action", [1, 3])
@pytest.mark.parametrize("sticky_probability", [0.0, 1.0])
def test_reset_discards_previous_episode_action(
    game, previous_action, sticky_probability
):
    env = Environment(game, sticky_action_prob=0.0)
    env.seed(0)
    env.reset()
    env.act(previous_action)
    assert env.last_action == previous_action

    reference = Environment(game, sticky_action_prob=sticky_probability)
    env.sticky_action_prob = sticky_probability
    for seed in [7, 42]:
        env.seed(seed)
        reference.seed(seed)
        env.reset()
        reference.reset()
        assert env.last_action == 0
        np.testing.assert_array_equal(env.state(), reference.state())
        for action in [2, 4, 0]:
            assert env.act(action) == reference.act(action)
            np.testing.assert_array_equal(env.state(), reference.state())


@pytest.mark.parametrize(
    "game", ["asterix", "breakout", "freeway", "seaquest", "space_invaders"]
)
@pytest.mark.parametrize("minimal", [False, True])
def test_gym_reset_starts_with_no_sticky_action(game, minimal):
    env = BaseEnv(game, use_minimal_action_set=minimal, sticky_action_prob=0.0)
    reference = BaseEnv(game, use_minimal_action_set=minimal, sticky_action_prob=1.0)
    try:
        env.reset(seed=0)
        env.step(1)
        env.game.sticky_action_prob = 1.0
        obs, info = env.reset(seed=7)
        expected_obs, expected_info = reference.reset(seed=7)
        assert env.game.last_action == 0
        np.testing.assert_array_equal(obs, expected_obs)
        assert info == expected_info
        actual = env.step(1)
        expected = reference.step(1)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
    finally:
        env.close()
        reference.close()
