from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


Dynamics = Literal["nonconstricted", "constricted"]
BoundaryStrategy = Literal["project", "feasible_rejection"]


@dataclass(frozen=True)
class PSOConfig:
    iterations: int = 40
    particles: int = 12

    # Non-constricted dynamics
    inertia: float = 1.0
    inertia_damping: float = 0.99
    cognitive: float = 1.0
    social: float = 2.0

    # Shared
    velocity_fraction: float = 0.10
    seed: int = 31001

    # V2 factorial separation
    dynamics: Dynamics = "constricted"
    boundary: BoundaryStrategy = "project"

    # Clerc-style constriction
    constriction_phi1: float = 2.05
    constriction_phi2: float = 2.05


@dataclass
class PSOResult:
    best_position: np.ndarray
    best_cost: float
    history: dict[str, np.ndarray]


def _swarm_diameter(position: np.ndarray) -> float:
    diff = position[:, None, :] - position[None, :, :]
    return float(np.max(np.linalg.norm(diff, axis=2)))


def constriction_coefficients(
    phi1: float,
    phi2: float,
) -> tuple[float, float, float]:
    phi = float(phi1 + phi2)

    if phi <= 4.0:
        raise ValueError("Constriction requires phi1 + phi2 > 4.")

    chi = 2.0 / (phi - 2.0 + np.sqrt(phi * phi - 4.0 * phi))
    c1 = chi * phi1
    c2 = chi * phi2

    return float(chi), float(c1), float(c2)


def _dynamics_coefficients(
    config: PSOConfig,
    current_inertia: float,
) -> tuple[float, float, float]:
    if config.dynamics == "nonconstricted":
        return (
            float(current_inertia),
            float(config.cognitive),
            float(config.social),
        )

    if config.dynamics == "constricted":
        chi, c1, c2 = constriction_coefficients(
            config.constriction_phi1,
            config.constriction_phi2,
        )
        return chi, c1, c2

    raise ValueError(f"Unsupported dynamics: {config.dynamics}")


def _apply_boundary(
    previous_position: np.ndarray,
    candidate: np.ndarray,
    velocity: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    strategy: BoundaryStrategy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Apply the frozen V2 boundary rule.

    project:
        Reflect offending velocity components, then coordinate-wise project
        the proposed point into the feasible box.

    feasible_rejection:
        If any coordinate is infeasible, reject the whole proposed move,
        retain the previous position, reverse offending velocity components,
        and do not evaluate the infeasible candidate.
    """
    previous_position = np.asarray(previous_position, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    velocity = np.asarray(velocity, dtype=float).copy()

    outside = (candidate < lower) | (candidate > upper)

    if np.any(outside):
        velocity[outside] *= -1.0

        if strategy == "project":
            new_position = np.clip(candidate, lower, upper)
            return new_position, velocity, outside, True

        if strategy == "feasible_rejection":
            return previous_position.copy(), velocity, outside, False

        raise ValueError(f"Unsupported boundary strategy: {strategy}")

    if strategy not in {"project", "feasible_rejection"}:
        raise ValueError(f"Unsupported boundary strategy: {strategy}")

    return candidate.copy(), velocity, outside, True


def particle_swarm_optimize(
    cost_function: Callable[[np.ndarray], float],
    initial_position: np.ndarray,
    lower_bound: np.ndarray | float,
    upper_bound: np.ndarray | float,
    config: PSOConfig,
) -> PSOResult:
    """
    V2 particle swarm optimizer.

    Dynamics and boundary handling are deliberately independent factors.
    This is required for the frozen NFPSO / P-NFPSO / C-NFPSO /
    PC-NFPSO factorial ablation.
    """
    x0 = np.asarray(initial_position, dtype=float).reshape(-1)
    n_var = len(x0)

    lower = np.broadcast_to(
        np.asarray(lower_bound, dtype=float),
        (n_var,),
    ).copy()
    upper = np.broadcast_to(
        np.asarray(upper_bound, dtype=float),
        (n_var,),
    ).copy()

    if np.any(upper <= lower):
        raise ValueError(
            "Each upper bound must be greater than its lower bound."
        )

    if config.iterations < 1 or config.particles < 2:
        raise ValueError(
            "PSO requires at least one iteration and two particles."
        )

    if config.velocity_fraction <= 0.0:
        raise ValueError("velocity_fraction must be positive.")

    if config.dynamics not in {"nonconstricted", "constricted"}:
        raise ValueError(f"Unsupported dynamics: {config.dynamics}")

    if config.boundary not in {"project", "feasible_rejection"}:
        raise ValueError(
            f"Unsupported boundary strategy: {config.boundary}"
        )

    rng = np.random.default_rng(config.seed)

    # Frozen V2 initialization:
    # particle 0 = clipped subtractive-clustering antecedent vector;
    # remaining particles = uniform within the frozen box.
    position = rng.uniform(
        lower,
        upper,
        size=(config.particles, n_var),
    )
    position[0] = np.clip(x0, lower, upper)

    # Frozen V2 initialization: zero velocity.
    velocity = np.zeros_like(position)

    cost = np.array(
        [float(cost_function(p)) for p in position],
        dtype=float,
    )

    # Initial non-finite objectives are fatal.
    if not np.all(np.isfinite(cost)):
        raise ValueError("Initial cost contains non-finite values.")

    personal_position = position.copy()
    personal_cost = cost.copy()

    best_index = int(np.argmin(personal_cost))
    global_position = personal_position[best_index].copy()
    global_cost = float(personal_cost[best_index])

    span = upper - lower
    velocity_max = config.velocity_fraction * span
    velocity_min = -velocity_max

    current_inertia = float(config.inertia)

    history = {
        "best_cost": np.empty(config.iterations),
        "mean_cost": np.empty(config.iterations),
        "mean_velocity_norm": np.empty(config.iterations),
        "swarm_diameter": np.empty(config.iterations),
        "best_parameter_drift": np.empty(config.iterations),
        "boundary_hits": np.empty(config.iterations),
        "boundary_rejections": np.empty(config.iterations),
        "inertia": np.empty(config.iterations),
    }

    previous_global = global_position.copy()

    for iteration in range(config.iterations):
        boundary_hits = 0
        boundary_rejections = 0

        w, c1, c2 = _dynamics_coefficients(
            config,
            current_inertia,
        )

        # Deliberately sequential.
        # global_position may change between particles in the same iteration.
        for i in range(config.particles):
            r1 = rng.random(n_var)
            r2 = rng.random(n_var)

            velocity[i] = (
                w * velocity[i]
                + c1
                * r1
                * (personal_position[i] - position[i])
                + c2
                * r2
                * (global_position - position[i])
            )

            velocity[i] = np.clip(
                velocity[i],
                velocity_min,
                velocity_max,
            )

            previous_position = position[i].copy()
            candidate = previous_position + velocity[i]

            (
                proposed_position,
                reflected_velocity,
                outside,
                accepted,
            ) = _apply_boundary(
                previous_position=previous_position,
                candidate=candidate,
                velocity=velocity[i],
                lower=lower,
                upper=upper,
                strategy=config.boundary,
            )

            velocity[i] = reflected_velocity
            boundary_hits += int(np.count_nonzero(outside))

            if not accepted:
                # Frozen feasible-rejection semantics:
                # retain old point and old cost; never evaluate infeasible
                # candidate.
                boundary_rejections += 1
                continue

            position[i] = proposed_position

            candidate_cost = float(cost_function(position[i]))

            # Later non-finite objectives become maximal finite cost.
            if not np.isfinite(candidate_cost):
                candidate_cost = np.finfo(float).max

            cost[i] = candidate_cost

            if cost[i] < personal_cost[i]:
                personal_cost[i] = cost[i]
                personal_position[i] = position[i].copy()

            if cost[i] < global_cost:
                global_cost = float(cost[i])
                global_position = position[i].copy()

        history["best_cost"][iteration] = global_cost
        cost_scale = float(np.max(np.abs(cost)))
        history["mean_cost"][iteration] = (
            0.0
            if cost_scale == 0.0
            else float(cost_scale * np.mean(cost / cost_scale))
        )
        history["mean_velocity_norm"][iteration] = float(
            np.mean(np.linalg.norm(velocity, axis=1))
        )
        history["swarm_diameter"][iteration] = _swarm_diameter(
            position
        )
        history["best_parameter_drift"][iteration] = float(
            np.linalg.norm(global_position - previous_global)
        )
        history["boundary_hits"][iteration] = float(boundary_hits)
        history["boundary_rejections"][iteration] = float(
            boundary_rejections
        )
        history["inertia"][iteration] = w

        previous_global = global_position.copy()

        if config.dynamics == "nonconstricted":
            current_inertia *= config.inertia_damping

    return PSOResult(
        best_position=global_position,
        best_cost=global_cost,
        history=history,
    )