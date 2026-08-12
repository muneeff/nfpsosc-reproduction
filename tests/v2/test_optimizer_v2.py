import numpy as np
import pytest

from nfpsosc.v2.optimizer import (
    PSOConfig,
    _apply_boundary,
    constriction_coefficients,
    particle_swarm_optimize,
)


def sphere(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def test_constriction_coefficients_match_frozen_protocol():
    chi, c1, c2 = constriction_coefficients(2.05, 2.05)

    assert chi == pytest.approx(
        0.7298437881283576,
        rel=0.0,
        abs=1e-15,
    )
    assert c1 == pytest.approx(
        1.496179765663133,
        rel=0.0,
        abs=1e-15,
    )
    assert c2 == pytest.approx(
        1.496179765663133,
        rel=0.0,
        abs=1e-15,
    )


def test_projection_projects_to_box_and_reflects_velocity():
    previous = np.array([0.5, 0.5])
    candidate = np.array([1.2, -0.3])
    velocity = np.array([0.7, -0.8])
    lower = np.array([0.0, 0.0])
    upper = np.array([1.0, 1.0])

    position, new_velocity, outside, accepted = _apply_boundary(
        previous,
        candidate,
        velocity,
        lower,
        upper,
        "project",
    )

    assert accepted is True
    np.testing.assert_allclose(position, [1.0, 0.0])
    np.testing.assert_array_equal(outside, [True, True])
    np.testing.assert_allclose(new_velocity, [-0.7, 0.8])


def test_feasible_rejection_retains_whole_previous_position():
    previous = np.array([0.5, 0.5])
    candidate = np.array([1.2, 0.6])
    velocity = np.array([0.7, 0.1])
    lower = np.array([0.0, 0.0])
    upper = np.array([1.0, 1.0])

    position, new_velocity, outside, accepted = _apply_boundary(
        previous,
        candidate,
        velocity,
        lower,
        upper,
        "feasible_rejection",
    )

    assert accepted is False
    np.testing.assert_allclose(position, previous)
    np.testing.assert_array_equal(outside, [True, False])
    np.testing.assert_allclose(new_velocity, [-0.7, 0.1])


def test_projection_and_rejection_are_not_equivalent():
    previous = np.array([0.5, 0.5])
    candidate = np.array([1.2, 0.6])
    velocity = np.array([0.7, 0.1])
    lower = np.array([0.0, 0.0])
    upper = np.array([1.0, 1.0])

    projected, _, _, project_accepted = _apply_boundary(
        previous,
        candidate,
        velocity,
        lower,
        upper,
        "project",
    )

    rejected, _, _, rejection_accepted = _apply_boundary(
        previous,
        candidate,
        velocity,
        lower,
        upper,
        "feasible_rejection",
    )

    assert project_accepted is True
    assert rejection_accepted is False
    assert not np.array_equal(projected, rejected)


@pytest.mark.parametrize(
    "dynamics",
    ["nonconstricted", "constricted"],
)
@pytest.mark.parametrize(
    "boundary",
    ["project", "feasible_rejection"],
)
def test_all_factorial_variants_keep_best_position_feasible(
    dynamics,
    boundary,
):
    config = PSOConfig(
        iterations=8,
        particles=6,
        seed=31001,
        dynamics=dynamics,
        boundary=boundary,
    )

    result = particle_swarm_optimize(
        cost_function=sphere,
        initial_position=np.array([0.8, -0.7]),
        lower_bound=np.array([-1.0, -1.0]),
        upper_bound=np.array([1.0, 1.0]),
        config=config,
    )

    assert np.all(result.best_position >= -1.0)
    assert np.all(result.best_position <= 1.0)
    assert np.isfinite(result.best_cost)


def test_seed_reproducibility():
    config = PSOConfig(
        iterations=10,
        particles=7,
        seed=31002,
        dynamics="constricted",
        boundary="project",
    )

    kwargs = dict(
        cost_function=sphere,
        initial_position=np.array([0.6, -0.4, 0.2]),
        lower_bound=-1.0,
        upper_bound=1.0,
        config=config,
    )

    first = particle_swarm_optimize(**kwargs)
    second = particle_swarm_optimize(**kwargs)

    np.testing.assert_array_equal(
        first.best_position,
        second.best_position,
    )
    assert first.best_cost == second.best_cost

    for key in first.history:
        np.testing.assert_array_equal(
            first.history[key],
            second.history[key],
        )


def test_initial_nonfinite_cost_is_fatal():
    def bad_cost(_: np.ndarray) -> float:
        return np.nan

    config = PSOConfig(
        iterations=2,
        particles=3,
        seed=31001,
    )

    with pytest.raises(
        ValueError,
        match="Initial cost contains non-finite values",
    ):
        particle_swarm_optimize(
            cost_function=bad_cost,
            initial_position=np.array([0.0]),
            lower_bound=-1.0,
            upper_bound=1.0,
            config=config,
        )


def test_default_v2_budget():
    config = PSOConfig()

    assert config.particles == 12
    assert config.iterations == 40
    assert config.velocity_fraction == pytest.approx(0.10)
class _DeterministicRNG:
    """Small deterministic RNG used only for optimizer behavioral tests."""

    def __init__(self, initial_positions):
        self.initial_positions = np.asarray(
            initial_positions,
            dtype=float,
        )
        self.random_calls = 0

    def uniform(self, low, high, size):
        assert tuple(size) == tuple(self.initial_positions.shape)
        return self.initial_positions.copy()

    def random(self, n):
        # PSO calls random() twice per particle:
        # r1 first, then r2.
        self.random_calls += 1

        if self.random_calls % 2 == 1:
            return np.zeros(n, dtype=float)

        return np.ones(n, dtype=float)


def test_feasible_rejection_does_not_evaluate_infeasible_candidate(
    monkeypatch,
):
    import nfpsosc.v2.optimizer as optimizer_module

    fake_rng = _DeterministicRNG(
        initial_positions=np.array(
            [
                [0.0],
                [0.05],
            ]
        )
    )

    monkeypatch.setattr(
        optimizer_module.np.random,
        "default_rng",
        lambda seed: fake_rng,
    )

    evaluations = []

    def counted_cost(x):
        value = float(np.asarray(x)[0])
        evaluations.append(value)
        return value * value

    config = PSOConfig(
        iterations=1,
        particles=2,
        seed=31001,
        dynamics="constricted",
        boundary="feasible_rejection",
    )

    result = particle_swarm_optimize(
        cost_function=counted_cost,
        initial_position=np.array([0.0]),
        lower_bound=np.array([0.0]),
        upper_bound=np.array([1.0]),
        config=config,
    )

    # Two initialization evaluations.
    # Particle 0 remains feasible and is evaluated.
    # Particle 1 crosses below zero and MUST be rejected before
    # cost_function is called.
    assert len(evaluations) == 3

    assert result.history["boundary_rejections"][0] == pytest.approx(1.0)
    assert result.history["boundary_hits"][0] == pytest.approx(1.0)

    # No infeasible point was ever sent to the objective.
    assert all(0.0 <= x <= 1.0 for x in evaluations)


def test_global_best_is_updated_sequentially_within_iteration(
    monkeypatch,
):
    import nfpsosc.v2.optimizer as optimizer_module

    fake_rng = _DeterministicRNG(
        initial_positions=np.array(
            [
                [5.0],
                [4.0],
            ]
        )
    )

    monkeypatch.setattr(
        optimizer_module.np.random,
        "default_rng",
        lambda seed: fake_rng,
    )

    evaluations = []

    def target_three(x):
        value = float(np.asarray(x)[0])
        evaluations.append(value)
        return (value - 3.0) ** 2

    config = PSOConfig(
        iterations=1,
        particles=2,
        seed=31001,
        dynamics="constricted",
        boundary="project",
    )

    result = particle_swarm_optimize(
        cost_function=target_three,
        initial_position=np.array([5.0]),
        lower_bound=np.array([0.0]),
        upper_bound=np.array([100.0]),
        config=config,
    )

    chi, _, c2 = constriction_coefficients(2.05, 2.05)

    # Initial global best is particle 1 at x=4.
    expected_particle0 = 5.0 + c2 * (4.0 - 5.0)

    # Because updates are sequential, particle 1 must see particle 0's
    # newly improved global best, not the old x=4 global best.
    expected_particle1 = (
        4.0
        + c2 * (expected_particle0 - 4.0)
    )

    assert len(evaluations) == 4

    assert evaluations[2] == pytest.approx(
        expected_particle0,
        rel=0.0,
        abs=1e-12,
    )

    assert evaluations[3] == pytest.approx(
        expected_particle1,
        rel=0.0,
        abs=1e-12,
    )

    # Under synchronous-gbest semantics particle 1 would remain at 4.
    assert evaluations[3] != pytest.approx(4.0)

    assert result.best_cost <= (expected_particle0 - 3.0) ** 2


def test_nonconstricted_inertia_damps_once_per_iteration():
    config = PSOConfig(
        iterations=4,
        particles=4,
        seed=31001,
        dynamics="nonconstricted",
        boundary="project",
        inertia=1.0,
        inertia_damping=0.99,
    )

    result = particle_swarm_optimize(
        cost_function=sphere,
        initial_position=np.array([0.5, -0.5]),
        lower_bound=-1.0,
        upper_bound=1.0,
        config=config,
    )

    np.testing.assert_allclose(
        result.history["inertia"],
        [
            1.0,
            0.99,
            0.9801,
            0.970299,
        ],
        rtol=0.0,
        atol=1e-15,
    )


def test_constricted_mode_does_not_apply_inertia_damping():
    config = PSOConfig(
        iterations=4,
        particles=4,
        seed=31001,
        dynamics="constricted",
        boundary="project",
    )

    result = particle_swarm_optimize(
        cost_function=sphere,
        initial_position=np.array([0.5, -0.5]),
        lower_bound=-1.0,
        upper_bound=1.0,
        config=config,
    )

    chi, _, _ = constriction_coefficients(2.05, 2.05)

    np.testing.assert_allclose(
        result.history["inertia"],
        np.full(4, chi),
        rtol=0.0,
        atol=1e-15,
    )


def test_later_nonfinite_candidate_cost_is_not_fatal():
    calls = {"n": 0}

    def becomes_nonfinite(x):
        calls["n"] += 1

        # Initial swarm evaluation is finite.
        if calls["n"] <= 3:
            return sphere(x)

        # Later candidate evaluations become invalid.
        return np.nan

    config = PSOConfig(
        iterations=2,
        particles=3,
        seed=31001,
        dynamics="constricted",
        boundary="project",
    )

    result = particle_swarm_optimize(
        cost_function=becomes_nonfinite,
        initial_position=np.array([0.5]),
        lower_bound=-1.0,
        upper_bound=1.0,
        config=config,
    )

    # Protocol: initial non-finite is fatal, later non-finite is converted
    # to maximal finite cost instead of crashing the optimizer.
    assert np.isfinite(result.best_cost)
    assert np.all(np.isfinite(result.history["mean_cost"]))

def test_invalid_constriction_sum_is_rejected():
    with pytest.raises(
        ValueError,
        match="phi1 \\+ phi2 > 4",
    ):
        constriction_coefficients(2.0, 2.0)