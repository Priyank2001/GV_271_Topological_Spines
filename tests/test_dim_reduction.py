"""Tests for pipeline/dim_reduction.py. Run from the repo root: python3 -m pytest tests"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.dim_reduction import (ChainReducer, Reducer, SklearnReducer,
                                    available_reducers, make_reducer,
                                    reduced_knn_graph, register_reducer)
from pipeline.extremum_graph import extremum_graph


def ring_in_high_d(n=4000, d=20, seed=0):
    """6-peak ring in the (x0, x1) plane, small noise in the other d-2 axes."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.uniform(-5, 5, (n, 2)), rng.normal(0, 0.1, (n, d - 2))], 1)
    r, th = np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0])
    return X.astype(np.float32), -(r - 2.5) ** 2 + 0.5 * np.cos(6 * th)


@pytest.mark.parametrize("name", ["none", "pca", "random_projection", "isomap", "kernel_pca"])
def test_shape_and_out_of_sample(name):
    X, _ = ring_in_high_d(n=600)
    r = make_reducer(name, n_components=3) if name != "none" else make_reducer(name)
    Y = r.fit_transform(X)
    assert Y.shape == (600, 3 if name != "none" else 20) and Y.dtype == np.float32
    assert r.transform(X[:5]).shape == (5, Y.shape[1])


def test_tsne_fit_on_sample_places_the_rest():
    X, _ = ring_in_high_d(n=1500)
    r = make_reducer("tsne", n_components=2, fit_sample_size=500, perplexity=20)
    Y = r.fit_transform(X)
    assert Y.shape == (1500, 2) and np.isfinite(Y).all()
    assert not r.supports_transform


def test_pca_recovers_the_ring_plane():
    X, _ = ring_in_high_d()
    r = make_reducer("pca", n_components=2)
    r.fit(X)
    assert r.explained_variance_ratio.sum() > 0.95


def test_pca_graph_gives_the_6_cycle():
    X, f = ring_in_high_d()
    g, Y = reduced_knn_graph(X, k=15, method="pca", n_components=2)
    eg = extremum_graph(f, g).simplify(0.04)
    assert Y.shape == (len(X), 2)
    assert eg.num_peaks == 6 and eg.num_saddles == 6


def test_chain_and_string_shorthand():
    X, _ = ring_in_high_d(n=500)
    r = make_reducer("pca>random_projection", pca={"n_components": 5}, n_components=2)
    assert isinstance(r, ChainReducer)
    assert r.steps[0].n_components == 5 and r.steps[1].n_components == 2
    assert r.fit_transform(X).shape == (500, 2)


def test_any_sklearn_estimator_and_custom_registration():
    from sklearn.decomposition import FastICA

    X, _ = ring_in_high_d(n=300)
    assert make_reducer(FastICA(n_components=2, random_state=0)).fit_transform(X).shape == (300, 2)

    @register_reducer("first_axes")
    class FirstAxes(Reducer):
        def _fit_transform(self, X):
            return X[:, :self.n_components]

        def _transform(self, X):
            return X[:, :self.n_components]

    assert "first_axes" in available_reducers()
    assert np.allclose(make_reducer("first_axes", n_components=2).fit_transform(X), X[:, :2])


def test_unknown_name_and_transform_before_fit():
    with pytest.raises(ValueError, match="available"):
        make_reducer("no_such_method")
    with pytest.raises(RuntimeError):
        make_reducer("pca").transform(np.zeros((2, 3)))
