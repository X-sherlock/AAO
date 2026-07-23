import numpy as np

from asset_allocation.policy_models import DirectionScalePolicy, direction_and_scale


def test_direction_and_scale_invariants():
    direction, alpha = direction_and_scale(
        np.array([2.0, -1.0, 0.5, 4.0]), scale_logit=0.0, alpha_max=0.2
    )
    assert abs(direction.sum()) < 1e-12
    assert abs(np.abs(direction).sum() - 1.0) < 1e-12
    assert alpha == 0.1


def test_constant_logits_produce_safe_zero_direction():
    direction, alpha = direction_and_scale(
        np.ones(3), scale_logit=100.0, alpha_max=0.12
    )
    np.testing.assert_array_equal(direction, np.zeros(3))
    assert 0.0 <= alpha <= 0.12


def test_policy_is_seed_reproducible():
    left = DirectionScalePolicy(5, 3, 8, 0.1, seed=7)
    right = DirectionScalePolicy(5, 3, 8, 0.1, seed=7)
    state = np.arange(5, dtype=float)
    left_result = left(state)
    right_result = right(state)
    np.testing.assert_allclose(left_result[0], right_result[0])
    assert left_result[1] == right_result[1]
