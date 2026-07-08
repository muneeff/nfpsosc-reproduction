from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


@dataclass(frozen=True)
class PSOConfig:
    iterations: int = 1000
    particles: int = 25
    inertia: float = 1.0
    inertia_damping: float = 0.99
    cognitive: float = 1.0
    social: float = 2.0
    velocity_fraction: float = 0.1
    seed: int = 123
    mode: Literal["legacy", "projected", "constricted"] = "legacy"
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


def particle_swarm_optimize(
    cost_function: Callable[[np.ndarray], float],
    initial_position: np.ndarray,
    lower_bound: np.ndarray | float,
    upper_bound: np.ndarray | float,
    config: PSOConfig,
) -> PSOResult:
    x0 = np.asarray(initial_position, dtype=float).reshape(-1)
    n_var = len(x0)
    lower = np.broadcast_to(np.asarray(lower_bound, dtype=float), (n_var,)).copy()
    upper = np.broadcast_to(np.asarray(upper_bound, dtype=float), (n_var,)).copy()
    if np.any(upper <= lower):
        raise ValueError("Each upper bound must be greater than its lower bound.")
    if config.iterations < 1 or config.particles < 2:
        raise ValueError("PSO requires at least one iteration and two particles.")

    rng = np.random.default_rng(config.seed)
    position = rng.uniform(lower, upper, size=(config.particles, n_var))
    position[0] = np.clip(x0, lower, upper)
    velocity = np.zeros_like(position)
    cost = np.array([float(cost_function(p)) for p in position])
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

    w = config.inertia
    c1 = config.cognitive
    c2 = config.social
    if config.mode == "constricted":
        phi1 = config.constriction_phi1
        phi2 = config.constriction_phi2
        phi = phi1 + phi2
        if phi <= 4.0:
            raise ValueError("Constriction mode requires phi1 + phi2 > 4.")
        chi = 2.0 / (phi - 2.0 + np.sqrt(phi * phi - 4.0 * phi))
        w = chi
        c1 = chi * phi1
        c2 = chi * phi2

    history = {
        "best_cost": np.empty(config.iterations),
        "mean_cost": np.empty(config.iterations),
        "mean_velocity_norm": np.empty(config.iterations),
        "swarm_diameter": np.empty(config.iterations),
        "best_parameter_drift": np.empty(config.iterations),
        "boundary_hits": np.empty(config.iterations),
        "inertia": np.empty(config.iterations),
    }

    previous_global = global_position.copy()
    for iteration in range(config.iterations):
        boundary_hits = 0
        for i in range(config.particles):
            r1 = rng.random(n_var)
            r2 = rng.random(n_var)
            velocity[i] = (
                w * velocity[i]
                + c1 * r1 * (personal_position[i] - position[i])
                + c2 * r2 * (global_position - position[i])
            )
            velocity[i] = np.clip(velocity[i], velocity_min, velocity_max)
            candidate = position[i] + velocity[i]
            outside = (candidate < lower) | (candidate > upper)
            boundary_hits += int(np.count_nonzero(outside))
            velocity[i, outside] *= -1.0
            # All modes project to the finite MATLAB-style box. In research
            # modes, the bounds are meaningful model constraints rather than
            # arbitrary multiplicative factors.
            position[i] = np.clip(candidate, lower, upper)
            cost[i] = float(cost_function(position[i]))
            if not np.isfinite(cost[i]):
                cost[i] = np.finfo(float).max

            if cost[i] < personal_cost[i]:
                personal_cost[i] = cost[i]
                personal_position[i] = position[i].copy()
                if cost[i] < global_cost:
                    global_cost = float(cost[i])
                    global_position = position[i].copy()

        history["best_cost"][iteration] = global_cost
        history["mean_cost"][iteration] = float(np.mean(cost))
        history["mean_velocity_norm"][iteration] = float(
            np.mean(np.linalg.norm(velocity, axis=1))
        )
        history["swarm_diameter"][iteration] = _swarm_diameter(position)
        history["best_parameter_drift"][iteration] = float(
            np.linalg.norm(global_position - previous_global)
        )
        history["boundary_hits"][iteration] = float(boundary_hits)
        history["inertia"][iteration] = w
        previous_global = global_position.copy()
        if config.mode != "constricted":
            w *= config.inertia_damping

    return PSOResult(global_position, global_cost, history)
