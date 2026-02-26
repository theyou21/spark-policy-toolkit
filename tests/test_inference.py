import numpy as np
import pytest

from src.inference import (
    _build_vectorized_forest,
    _predict_forest_mean_numpy,
    _to_jsonable,
    generate_synthetic_forest,
)


def test_generate_synthetic_forest_shapes_and_routing() -> None:
    forest = generate_synthetic_forest(
        depth=3,
        n_treatments=4,
        n_trees=3,
        n_features=10,
        missing_routing="left",
        seed=1,
    )
    assert len(forest) == 3

    expected_nodes = 2 ** (3 + 1) - 1
    for tree in forest:
        assert tree["node_type"].shape == (expected_nodes,)
        assert tree["feature_idx"].shape == (expected_nodes,)
        assert tree["leaf_response_rates"].shape == (expected_nodes, 4)
        assert np.all(tree["nan_goes_left"])


def test_generate_synthetic_forest_rejects_invalid_missing_routing() -> None:
    with pytest.raises(ValueError, match="missing_routing"):
        generate_synthetic_forest(
            depth=2,
            n_treatments=2,
            n_trees=1,
            missing_routing="invalid",
        )


def test_predict_forest_mean_numpy_averages_tree_outputs() -> None:
    tree_a = {
        "node_type": np.array([0], dtype=np.int8),
        "feature_idx": np.array([0], dtype=np.int64),
        "thresholds": np.array([0.0], dtype=np.float64),
        "cat_value": np.array([0.0], dtype=np.float64),
        "left_child": np.array([0], dtype=np.int64),
        "right_child": np.array([0], dtype=np.int64),
        "nan_goes_left": np.array([True], dtype=bool),
        "leaf_response_rates": np.array([[0.2, 0.8]], dtype=np.float64),
    }
    tree_b = {
        "node_type": np.array([0], dtype=np.int8),
        "feature_idx": np.array([0], dtype=np.int64),
        "thresholds": np.array([0.0], dtype=np.float64),
        "cat_value": np.array([0.0], dtype=np.float64),
        "left_child": np.array([0], dtype=np.int64),
        "right_child": np.array([0], dtype=np.int64),
        "nan_goes_left": np.array([True], dtype=bool),
        "leaf_response_rates": np.array([[0.6, 0.4]], dtype=np.float64),
    }

    models, _ = _build_vectorized_forest([tree_a, tree_b])
    X = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    pred = _predict_forest_mean_numpy(X, models)

    expected = np.array([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]], dtype=np.float64)
    np.testing.assert_allclose(pred, expected)


def test_to_jsonable_converts_numpy_types() -> None:
    payload = {
        "arr": np.array([1, 2, 3], dtype=np.int64),
        "nested": [np.float64(1.25), np.bool_(True)],
    }
    converted = _to_jsonable(payload)
    assert converted == {"arr": [1, 2, 3], "nested": [1.25, True]}
