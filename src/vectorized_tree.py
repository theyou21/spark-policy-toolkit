"""Array-native, vectorized tree inference utilities."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


class VectorizedArrayTree:
    """Vectorized inference for array-native uplift trees.

    The tree is represented as parallel NumPy arrays keyed by:
    `node_type`, `feature_idx`, `thresholds`, `cat_value`, `left_child`,
    `right_child`, `nan_goes_left`, and `leaf_response_rates`.
    Root traversal starts at node index 0.

    Node type encoding:
    - 0: leaf
    - 1: continuous threshold split (`value <= threshold` goes left)
    - 2: categorical equality split (`value == cat_value` goes left)
    """

    LEAF = 0
    CONTINUOUS = 1
    CATEGORICAL = 2

    REQUIRED_KEYS = (
        "node_type",
        "feature_idx",
        "thresholds",
        "cat_value",
        "left_child",
        "right_child",
        "nan_goes_left",
        "leaf_response_rates",
    )

    def __init__(self, tree_arrays: Mapping[str, np.ndarray]) -> None:
        """Initialize from a dictionary of parallel arrays.

        Args:
            tree_arrays: Mapping with required keys defined in `REQUIRED_KEYS`.

        Raises:
            TypeError: If `tree_arrays` is not a mapping.
            ValueError: If array shapes or node schema are invalid.
        """
        if not isinstance(tree_arrays, Mapping):
            raise TypeError("tree_arrays must be a mapping of array fields.")

        missing = [key for key in self.REQUIRED_KEYS if key not in tree_arrays]
        if missing:
            raise ValueError(f"Missing required tree arrays: {missing}")

        self.node_type = np.asarray(tree_arrays["node_type"], dtype=np.int8)
        self.feature_idx = np.asarray(tree_arrays["feature_idx"], dtype=np.int64)
        self.thresholds = np.asarray(tree_arrays["thresholds"], dtype=np.float64)
        self.cat_value = np.asarray(tree_arrays["cat_value"])
        self.left_child = np.asarray(tree_arrays["left_child"], dtype=np.int64)
        self.right_child = np.asarray(tree_arrays["right_child"], dtype=np.int64)
        self.nan_goes_left = np.asarray(tree_arrays["nan_goes_left"], dtype=bool)
        self.leaf_response_rates = np.asarray(
            tree_arrays["leaf_response_rates"], dtype=np.float64
        )

        self._validate_schema()

    @property
    def n_nodes(self) -> int:
        """Total number of nodes in the tree."""
        return int(self.node_type.shape[0])

    @property
    def n_treatments(self) -> int:
        """Number of response-rate entries returned per row."""
        return int(self.leaf_response_rates.shape[1])

    def predict_numpy(
        self, X: np.ndarray, feature_names: list[str] | None = None
    ) -> np.ndarray:
        """Predict leaf response-rate vectors for a NumPy feature matrix.

        Args:
            X: Feature matrix with shape `[n_rows, n_features]`.
            feature_names: Optional feature names aligned with columns in `X`.
                This is accepted for API compatibility and validation only.

        Returns:
            Float64 matrix with shape `[n_rows, n_treatments]`.
        """
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D NumPy array of shape [n_rows, n_features].")

        n_rows, n_features = X_arr.shape

        if feature_names is not None and len(feature_names) != n_features:
            raise ValueError(
                "feature_names length must match X.shape[1] when provided."
            )

        self._validate_feature_indices(n_features=n_features)

        if n_rows == 0:
            return np.empty((0, self.n_treatments), dtype=np.float64)

        current_nodes = np.zeros(n_rows, dtype=np.int64)

        for _ in range(self.n_nodes + 1):
            node_types = self.node_type[current_nodes]
            active_mask = node_types != self.LEAF
            if not np.any(active_mask):
                break

            active_rows = np.flatnonzero(active_mask)
            active_nodes = current_nodes[active_rows]
            active_types = node_types[active_rows]

            continuous_mask = active_types == self.CONTINUOUS
            if np.any(continuous_mask):
                rows = active_rows[continuous_mask]
                nodes = active_nodes[continuous_mask]

                values = X_arr[rows, self.feature_idx[nodes]]
                missing = self._is_missing(values)

                go_left = np.zeros(rows.shape[0], dtype=bool)
                non_missing = ~missing
                if np.any(non_missing):
                    try:
                        compare = values[non_missing] <= self.thresholds[nodes[non_missing]]
                    except TypeError as exc:
                        raise TypeError(
                            "Continuous split comparison failed; ensure features are orderable."
                        ) from exc
                    go_left[non_missing] = np.asarray(compare, dtype=bool)
                go_left[missing] = self.nan_goes_left[nodes[missing]]
                current_nodes[rows] = np.where(
                    go_left, self.left_child[nodes], self.right_child[nodes]
                )

            categorical_mask = active_types == self.CATEGORICAL
            if np.any(categorical_mask):
                rows = active_rows[categorical_mask]
                nodes = active_nodes[categorical_mask]

                values = X_arr[rows, self.feature_idx[nodes]]
                missing = self._is_missing(values)

                go_left = np.zeros(rows.shape[0], dtype=bool)
                non_missing = ~missing
                if np.any(non_missing):
                    equals = values[non_missing] == self.cat_value[nodes[non_missing]]
                    go_left[non_missing] = np.asarray(equals, dtype=bool)
                go_left[missing] = self.nan_goes_left[nodes[missing]]

                current_nodes[rows] = np.where(
                    go_left, self.left_child[nodes], self.right_child[nodes]
                )
        else:
            raise ValueError(
                "Tree traversal exceeded node budget; check for cycles or invalid children."
            )

        if np.any(self.node_type[current_nodes] != self.LEAF):
            raise ValueError("Traversal ended on non-leaf nodes; tree structure is invalid.")

        return self.leaf_response_rates[current_nodes].astype(np.float64, copy=False)

    def predict_pandas(self, pdf: pd.DataFrame) -> np.ndarray:
        """Predict leaf response-rate vectors for a pandas DataFrame.

        Args:
            pdf: Input DataFrame with feature columns.

        Returns:
            Float64 matrix with shape `[len(pdf), n_treatments]`.
        """
        if not isinstance(pdf, pd.DataFrame):
            raise TypeError("pdf must be a pandas DataFrame.")

        return self.predict_numpy(pdf.to_numpy(copy=False), feature_names=list(pdf.columns))

    @staticmethod
    def _is_missing(values: np.ndarray) -> np.ndarray:
        """Return a boolean mask indicating missing values."""
        if values.dtype.kind in {"f", "c"}:
            return np.isnan(values)
        return np.asarray(pd.isna(values), dtype=bool)

    def _validate_schema(self) -> None:
        """Validate tree-array schema consistency."""
        array_fields = (
            ("node_type", self.node_type),
            ("feature_idx", self.feature_idx),
            ("thresholds", self.thresholds),
            ("cat_value", self.cat_value),
            ("left_child", self.left_child),
            ("right_child", self.right_child),
            ("nan_goes_left", self.nan_goes_left),
        )

        if self.node_type.ndim != 1:
            raise ValueError("node_type must be a 1D array.")

        n_nodes = self.node_type.shape[0]
        if n_nodes == 0:
            raise ValueError("Tree must contain at least one node.")

        for name, arr in array_fields[1:]:
            if arr.ndim != 1:
                raise ValueError(f"{name} must be a 1D array.")
            if arr.shape[0] != n_nodes:
                raise ValueError(f"{name} length must equal node_type length ({n_nodes}).")

        if self.leaf_response_rates.ndim != 2:
            raise ValueError("leaf_response_rates must be a 2D array.")
        if self.leaf_response_rates.shape[0] != n_nodes:
            raise ValueError(
                "leaf_response_rates first dimension must equal number of nodes."
            )
        if self.leaf_response_rates.shape[1] == 0:
            raise ValueError("leaf_response_rates must include at least one treatment column.")

        valid_types = {self.LEAF, self.CONTINUOUS, self.CATEGORICAL}
        observed_types = set(np.unique(self.node_type).tolist())
        invalid_types = sorted(observed_types - valid_types)
        if invalid_types:
            raise ValueError(
                f"node_type contains invalid values {invalid_types}; expected 0, 1, or 2."
            )

        split_mask = self.node_type != self.LEAF
        if np.any(split_mask):
            for name, child_arr in (
                ("left_child", self.left_child),
                ("right_child", self.right_child),
            ):
                bad = (child_arr[split_mask] < 0) | (child_arr[split_mask] >= n_nodes)
                if np.any(bad):
                    raise ValueError(
                        f"{name} contains out-of-range child indices for non-leaf nodes."
                    )

    def _validate_feature_indices(self, n_features: int) -> None:
        """Validate split-node feature indices against input feature width."""
        split_mask = self.node_type != self.LEAF
        if not np.any(split_mask):
            return

        split_features = self.feature_idx[split_mask]
        if np.any(split_features < 0):
            raise ValueError("feature_idx must be non-negative for non-leaf nodes.")
        if np.any(split_features >= n_features):
            raise ValueError(
                f"feature_idx references feature >= {n_features}; input does not match tree."
            )
