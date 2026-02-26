import numpy as np
import pandas as pd
import pytest

from src.vectorized_tree import VectorizedArrayTree


def _tree_arrays() -> dict[str, np.ndarray]:
    return {
        "node_type": np.array([1, 0, 2, 0, 0], dtype=np.int8),
        "feature_idx": np.array([0, 0, 1, 0, 0], dtype=np.int64),
        "thresholds": np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "cat_value": np.array([0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float64),
        "left_child": np.array([1, 0, 3, 0, 0], dtype=np.int64),
        "right_child": np.array([2, 0, 4, 0, 0], dtype=np.int64),
        "nan_goes_left": np.array([True, True, False, True, True], dtype=bool),
        "leaf_response_rates": np.array(
            [
                [0.0, 0.0, 0.0],  # internal
                [0.1, 0.2, 0.3],  # leaf
                [0.0, 0.0, 0.0],  # internal
                [0.4, 0.5, 0.6],  # leaf
                [0.7, 0.8, 0.9],  # leaf
            ],
            dtype=np.float64,
        ),
    }


def test_predict_numpy_handles_continuous_categorical_and_nan_routing() -> None:
    tree = VectorizedArrayTree(_tree_arrays())
    X = np.array(
        [
            [0.1, 1.0],  # root left
            [0.9, 3.0],  # root right, categorical left
            [0.9, 2.0],  # root right, categorical right
            [np.nan, 3.0],  # root NaN -> left
            [0.9, np.nan],  # categorical NaN -> right
        ],
        dtype=np.float64,
    )

    pred = tree.predict_numpy(X)
    expected = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.1, 0.2, 0.3],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(pred, expected)
    assert pred.dtype == np.float64


def test_predict_pandas_matches_numpy_path() -> None:
    tree = VectorizedArrayTree(_tree_arrays())
    pdf = pd.DataFrame(
        {
            "f0": [0.1, 0.9, np.nan],
            "f1": [0.0, 3.0, 2.0],
        }
    )

    pred_pd = tree.predict_pandas(pdf)
    pred_np = tree.predict_numpy(pdf.to_numpy(copy=False), feature_names=["f0", "f1"])
    np.testing.assert_allclose(pred_pd, pred_np)


def test_predict_numpy_empty_input_returns_empty_matrix() -> None:
    tree = VectorizedArrayTree(_tree_arrays())
    X = np.empty((0, 2), dtype=np.float64)
    pred = tree.predict_numpy(X)
    assert pred.shape == (0, 3)
    assert pred.dtype == np.float64


def test_init_validates_required_keys() -> None:
    arrays = _tree_arrays()
    arrays.pop("node_type")
    with pytest.raises(ValueError, match="Missing required tree arrays"):
        VectorizedArrayTree(arrays)


def test_predict_validates_feature_width() -> None:
    arrays = _tree_arrays()
    arrays["feature_idx"] = np.array([2, 0, 1, 0, 0], dtype=np.int64)
    tree = VectorizedArrayTree(arrays)
    with pytest.raises(ValueError, match="feature_idx references feature"):
        tree.predict_numpy(np.zeros((4, 2), dtype=np.float64))
