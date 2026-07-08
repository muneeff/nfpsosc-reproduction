from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TSKFIS:
    """First-order Takagi-Sugeno-Kang fuzzy inference system."""

    centers: np.ndarray        # (rules, features)
    sigmas: np.ndarray         # (rules, features)
    consequents: np.ndarray    # (rules, features + 1), last column is intercept
    activation_epsilon: float = 1e-12
    sigma_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=float)
        self.sigmas = np.asarray(self.sigmas, dtype=float)
        self.consequents = np.asarray(self.consequents, dtype=float)
        if self.centers.ndim != 2:
            raise ValueError("centers must be a 2D array.")
        if self.sigmas.shape != self.centers.shape:
            raise ValueError("sigmas must have the same shape as centers.")
        expected = (self.n_rules, self.n_features + 1)
        if self.consequents.shape != expected:
            raise ValueError(f"consequents must have shape {expected}.")
        if not all(
            np.all(np.isfinite(a))
            for a in (self.centers, self.sigmas, self.consequents)
        ):
            raise ValueError("Model parameters must be finite.")

    @property
    def n_rules(self) -> int:
        return int(self.centers.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.centers.shape[1])

    def copy(self) -> "TSKFIS":
        return TSKFIS(
            self.centers.copy(),
            self.sigmas.copy(),
            self.consequents.copy(),
            self.activation_epsilon,
            self.sigma_epsilon,
        )

    def _safe_sigmas(self) -> np.ndarray:
        # Negative widths in the legacy multiplicative code are equivalent after
        # squaring. We use absolute values while preventing zero division.
        return np.maximum(np.abs(self.sigmas), self.sigma_epsilon)

    def firing_strengths(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features:
            raise ValueError("Feature count does not match the model.")
        sigma = self._safe_sigmas()
        z = (X[:, None, :] - self.centers[None, :, :]) / sigma[None, :, :]
        return np.exp(-0.5 * np.sum(z * z, axis=2))

    def normalized_weights(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = self.firing_strengths(X)
        sums = np.sum(raw, axis=1, keepdims=True)
        normalized = raw / np.maximum(sums, self.activation_epsilon)
        return normalized, sums[:, 0]

    def rule_outputs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        slopes = self.consequents[:, :-1]
        intercepts = self.consequents[:, -1]
        return X @ slopes.T + intercepts[None, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        weights, _ = self.normalized_weights(X)
        outputs = self.rule_outputs(X)
        return np.sum(weights * outputs, axis=1)

    def fit_consequents(self, X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        weights, _ = self.normalized_weights(X)
        X1 = np.column_stack([X, np.ones(len(X))])
        design = np.concatenate(
            [weights[:, k : k + 1] * X1 for k in range(self.n_rules)], axis=1
        )
        gram = design.T @ design
        rhs = design.T @ y
        coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
        self.consequents = coef.reshape(self.n_rules, self.n_features + 1)

    def to_vector(self) -> np.ndarray:
        # Explicit order: all centers, all sigmas, then all consequents.
        return np.concatenate(
            [self.centers.ravel(), self.sigmas.ravel(), self.consequents.ravel()]
        )

    def from_vector(self, vector: np.ndarray) -> "TSKFIS":
        v = np.asarray(vector, dtype=float).reshape(-1)
        n_antecedent = self.n_rules * self.n_features
        n_consequent = self.n_rules * (self.n_features + 1)
        expected = 2 * n_antecedent + n_consequent
        if len(v) != expected:
            raise ValueError(f"Expected {expected} parameters, received {len(v)}.")
        pos = 0
        centers = v[pos : pos + n_antecedent].reshape(self.centers.shape)
        pos += n_antecedent
        sigmas = v[pos : pos + n_antecedent].reshape(self.sigmas.shape)
        pos += n_antecedent
        consequents = v[pos : pos + n_consequent].reshape(self.consequents.shape)
        return TSKFIS(
            centers,
            sigmas,
            consequents,
            self.activation_epsilon,
            self.sigma_epsilon,
        )

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        """Analytic input Jacobian df/dx for each sample."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw = self.firing_strengths(X)
        total = np.maximum(np.sum(raw, axis=1, keepdims=True), self.activation_epsilon)
        alpha = raw / total
        g = self.rule_outputs(X)
        slopes = self.consequents[:, :-1]
        sigma2 = self._safe_sigmas() ** 2

        # dw[n,k,j]
        dw = raw[:, :, None] * (
            -(X[:, None, :] - self.centers[None, :, :]) / sigma2[None, :, :]
        )
        dtotal = np.sum(dw, axis=1, keepdims=True)
        dalpha = (dw * total[:, :, None] - raw[:, :, None] * dtotal) / (
            total[:, :, None] ** 2
        )
        jac = np.sum(dalpha * g[:, :, None], axis=1) + np.sum(
            alpha[:, :, None] * slopes[None, :, :], axis=1
        )
        return jac
