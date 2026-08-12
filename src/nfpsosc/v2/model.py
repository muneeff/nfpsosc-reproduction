from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


def _as_2d_features(X: np.ndarray, n_features: int) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("X must be a 1D sample or a 2D feature matrix.")
    if arr.shape[1] != n_features:
        raise ValueError(
            f"Expected {n_features} features, received {arr.shape[1]}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("X must contain only finite values.")
    return arr


@dataclass
class TSKModel:
    """First-order TSK model for the frozen V2 PC-NFPSO protocol.

    Antecedents are Gaussian rules parameterized by positive widths.
    Consequents are affine functions, one per rule, with the final
    consequent column representing the intercept.
    """

    centers: np.ndarray
    sigmas: np.ndarray
    consequents: np.ndarray

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=float).copy()
        self.sigmas = np.asarray(self.sigmas, dtype=float).copy()
        self.consequents = np.asarray(self.consequents, dtype=float).copy()

        if self.centers.ndim != 2 or self.centers.shape[0] < 1:
            raise ValueError("centers must be a non-empty 2D array.")
        if self.sigmas.shape != self.centers.shape:
            raise ValueError("sigmas must have the same shape as centers.")
        expected = (self.n_rules, self.n_features + 1)
        if self.consequents.shape != expected:
            raise ValueError(f"consequents must have shape {expected}.")
        if not all(
            np.all(np.isfinite(a))
            for a in (self.centers, self.sigmas, self.consequents)
        ):
            raise ValueError("All TSK parameters must be finite.")
        if np.any(self.sigmas <= 0.0):
            raise ValueError("All Gaussian widths must be strictly positive.")

    @property
    def n_rules(self) -> int:
        return int(self.centers.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.centers.shape[1])

    def copy(self) -> "TSKModel":
        return TSKModel(
            centers=self.centers.copy(),
            sigmas=self.sigmas.copy(),
            consequents=self.consequents.copy(),
        )

    def log_firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """Return log raw Gaussian firing strengths, shape (N, R)."""
        X2 = _as_2d_features(X, self.n_features)
        z = (
            (X2[:, None, :] - self.centers[None, :, :])
            / self.sigmas[None, :, :]
        )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                log_raw = -0.5 * np.sum(z * z, axis=2)
            except FloatingPointError as exc:
                raise ValueError(
                    "Non-finite Gaussian standardized distance encountered."
                ) from exc
        if not np.all(np.isfinite(log_raw)):
            raise ValueError("Non-finite Gaussian log firing strength encountered.")
        return log_raw

    def firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """Return raw Gaussian firing strengths.

        Extreme low-coverage values may underflow to zero; objective code should
        use ``log_total_firing_strength`` rather than summing this array.
        """
        log_raw = self.log_firing_strengths(X)
        with np.errstate(under="ignore"):
            return np.exp(log_raw)

    def log_total_firing_strength(self, X: np.ndarray) -> np.ndarray:
        """Return log(sum_k raw_firing_k) stably for every sample."""
        return logsumexp(self.log_firing_strengths(X), axis=1)

    def normalized_weights(self, X: np.ndarray) -> np.ndarray:
        """Return stable normalized Gaussian rule weights, shape (N, R)."""
        log_raw = self.log_firing_strengths(X)
        log_total = logsumexp(log_raw, axis=1, keepdims=True)
        weights = np.exp(log_raw - log_total)
        if not np.all(np.isfinite(weights)):
            raise ValueError("Non-finite normalized rule weights encountered.")
        return weights

    def rule_outputs(self, X: np.ndarray) -> np.ndarray:
        """Return affine consequent outputs for every sample and rule."""
        X2 = _as_2d_features(X, self.n_features)
        slopes = self.consequents[:, :-1]
        intercepts = self.consequents[:, -1]
        return X2 @ slopes.T + intercepts[None, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X2 = _as_2d_features(X, self.n_features)
        weights = self.normalized_weights(X2)
        outputs = self.rule_outputs(X2)
        pred = np.sum(weights * outputs, axis=1)
        if not np.all(np.isfinite(pred)):
            raise ValueError("Non-finite TSK prediction encountered.")
        return pred

    def consequent_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return the variable-projection TSK consequent design matrix.

        Column order is rule-major:
        [w_1*x_1, ..., w_1*x_D, w_1,
         w_2*x_1, ..., w_2*x_D, w_2, ...].
        """
        X2 = _as_2d_features(X, self.n_features)
        weights = self.normalized_weights(X2)
        X1 = np.column_stack([X2, np.ones(len(X2), dtype=float)])
        return np.concatenate(
            [
                weights[:, k : k + 1] * X1
                for k in range(self.n_rules)
            ],
            axis=1,
        )

    def fit_consequents(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        alpha: float,
    ) -> "TSKModel":
        """Fit all affine consequent coefficients with the frozen V2 ridge solve.

        ``alpha`` is deliberately mandatory: V2 has no hidden ridge default.
        Intercepts are regularized because all consequent coefficients are
        regularized by the frozen protocol.
        """
        X2 = _as_2d_features(X, self.n_features)
        y1 = np.asarray(y, dtype=float).reshape(-1)
        if len(y1) != len(X2):
            raise ValueError("X and y must contain the same number of samples.")
        if not np.all(np.isfinite(y1)):
            raise ValueError("y must contain only finite values.")
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be a finite positive scalar.")

        design = self.consequent_design_matrix(X2)
        gram = design.T @ design
        rhs = design.T @ y1
        system = gram + float(alpha) * np.eye(gram.shape[0], dtype=float)
        try:
            coef = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("V2 ridge consequent solve failed.") from exc

        if not np.all(np.isfinite(coef)):
            raise ValueError("V2 ridge consequent solve produced non-finite values.")
        self.consequents = coef.reshape(self.n_rules, self.n_features + 1)
        return self

    def antecedent_vector(self) -> np.ndarray:
        """Return [all centers, all positive widths] in row-major order."""
        return np.concatenate([self.centers.ravel(), self.sigmas.ravel()])

    def with_antecedents(self, vector: np.ndarray) -> "TSKModel":
        """Return a copy with antecedents replaced by a V2 optimizer vector."""
        v = np.asarray(vector, dtype=float).reshape(-1)
        n = self.n_rules * self.n_features
        if len(v) != 2 * n:
            raise ValueError(
                f"Expected {2 * n} antecedent parameters, received {len(v)}."
            )
        if not np.all(np.isfinite(v)):
            raise ValueError("Antecedent vector must contain only finite values.")

        centers = v[:n].reshape(self.centers.shape)
        sigmas = v[n:].reshape(self.sigmas.shape)
        return TSKModel(
            centers=centers,
            sigmas=sigmas,
            consequents=self.consequents.copy(),
        )

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        """Analytic input Jacobian df/dx, shape (N, D).

        The derivative is computed from the same stable softmax-normalized
        Gaussian weights used by ``predict``; therefore it remains consistent
        in low-coverage regions where raw firing strengths may underflow.
        """
        X2 = _as_2d_features(X, self.n_features)
        weights = self.normalized_weights(X2)
        outputs = self.rule_outputs(X2)
        slopes = self.consequents[:, :-1]

        # q[n,k,j] = d log(raw_firing[n,k]) / d x_j
        q = -(
            X2[:, None, :] - self.centers[None, :, :]
        ) / (self.sigmas[None, :, :] ** 2)

        q_bar = np.sum(weights[:, :, None] * q, axis=1, keepdims=True)
        dweights = weights[:, :, None] * (q - q_bar)

        jac = np.sum(
            dweights * outputs[:, :, None],
            axis=1,
        ) + np.sum(
            weights[:, :, None] * slopes[None, :, :],
            axis=1,
        )

        if not np.all(np.isfinite(jac)):
            raise ValueError("Non-finite analytic TSK Jacobian encountered.")
        return jac
