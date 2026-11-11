"""
Dimensionality reduction before building the neighbour graph.

In high dimensions the kNN graph gets lopsided ("hubs", see observations.md)
and joins points that are far apart in the field. One fix is to build the
graph in a lower-dimensional embedding of the points instead. The **field
values are not changed**: only the positions used to decide who is whose
neighbour are.

Every method has the same interface (``Reducer``), and is created by name,
so switching from PCA to t-SNE is a one-string change:

    from pipeline.dim_reduction import make_reducer, reduced_knn_graph

    g, Y = reduced_knn_graph(ds.points, k=12, method="pca", n_components=3)
    g, Y = reduced_knn_graph(ds.points, k=12, method="tsne", n_components=2,
                             fit_sample_size=10_000)
    eg = extremum_graph(ds.values, g)    # same values, new graph

    # or use a reducer directly
    r = make_reducer("isomap", n_components=3, n_neighbors=15)
    Y = r.fit_transform(ds.points)

Built-in methods (``available_reducers()``):

=================== ============ ==============================================
name                out-of-sample notes
=================== ============ ==============================================
``none``            yes          identity, keeps the original space
``pca``             yes          linear; ``n_components`` may be a variance
                                 fraction such as 0.95
``random_projection`` yes        Gaussian random projection, very fast
``kernel_pca``      yes          O(n^2) memory: use ``fit_sample_size``
``isomap``          yes          geodesic distances; use ``fit_sample_size``
``tsne``            no (*)          keeps local neighbourhoods, distorts global
                                 distances; slow above ~20k points
``umap``            yes          needs ``pip install umap-learn``
=================== ============ ==============================================

(*) Methods without their own ``transform`` place the points they were not fit
on by inverse-distance averaging the embedding of their nearest fitted points
(found in the original space). That is what makes ``fit_sample_size`` work for
t-SNE.

Adding a method
---------------
Any object with scikit-learn's ``fit_transform`` (and optionally
``transform``) works without registering::

    reducer = SklearnReducer(sklearn.manifold.LocallyLinearEmbedding(n_components=3))

To make it available by name, subclass and register::

    @register_reducer("lle")
    class LLEReducer(SklearnReducer):
        def __init__(self, n_components=2, n_neighbors=10, **kw):
            from sklearn.manifold import LocallyLinearEmbedding
            super().__init__(LocallyLinearEmbedding(n_components=n_components,
                                                    n_neighbors=n_neighbors), **kw)

Several reducers in a row (the usual "PCA to 50, then t-SNE")::

    r = ChainReducer([make_reducer("pca", n_components=50),
                      make_reducer("tsne", n_components=2)])
    # or the shorthand: make_reducer("pca>tsne", pca={"n_components": 50})

Caveats
-------
* Points that are far apart in the original space can land next to each
  other in the embedding. The graph then links them and the field looks
  rough there, which can add small false peaks (simplify them away) or
  false saddles. Check against the unreduced graph where you can.
* A linear method such as PCA can only help when the field changes mostly
  along a few directions. If the field depends on every coordinate (as the
  ``ext_*d`` datasets do), dropping coordinates mixes up points with very
  different values.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .neighbor_graph import NeighborGraph, knn_graph

__all__ = [
    "Reducer", "IdentityReducer", "SklearnReducer", "ChainReducer",
    "PCAReducer", "RandomProjectionReducer", "KernelPCAReducer",
    "IsomapReducer", "TSNEReducer", "UMAPReducer",
    "register_reducer", "make_reducer", "available_reducers",
    "reduced_knn_graph",
]


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class Reducer(ABC):
    """Maps (N, d) points to (N, n_components) points.

    Subclasses implement ``_fit_transform`` and, if the method can embed new
    points, ``_transform``. The base class adds:

    * ``fit_sample_size``: fit on a random subset of this many points, then
      embed the rest (with ``_transform``, or by nearest-neighbour averaging
      when the method has none). Use it for methods that are slow or need
      O(n^2) memory.
    * ``random_state`` for the subset and for the method itself.
    """

    name: str = "reducer"

    def __init__(self, n_components: Union[int, float] = 2,
                 fit_sample_size: Optional[int] = None,
                 random_state: Optional[int] = 0,
                 extension_neighbors: int = 10):
        self.n_components = n_components
        self.fit_sample_size = fit_sample_size
        self.random_state = random_state
        self.extension_neighbors = extension_neighbors
        self._fit_points: Optional[np.ndarray] = None     # kept for the kNN extension
        self._fit_embedding: Optional[np.ndarray] = None
        self.fitted_ = False

    # ---- to implement ---------------------------------------------------
    @abstractmethod
    def _fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return its embedding."""

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Embed new points. Override when the method supports it."""
        raise NotImplementedError

    @property
    def supports_transform(self) -> bool:
        """True if the method has its own out-of-sample ``_transform``."""
        return type(self)._transform is not Reducer._transform

    # ---- public API -----------------------------------------------------
    def fit(self, X: np.ndarray) -> "Reducer":
        self.fit_transform(X)
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = _as_float(X)
        n = X.shape[0]
        if self.fit_sample_size is None or self.fit_sample_size >= n:
            Y = self._fit_transform(X)
            self._remember(X, Y)
            return _as_float(Y)

        rng = np.random.default_rng(self.random_state)
        sample = np.sort(rng.choice(n, self.fit_sample_size, replace=False))
        Y_sample = _as_float(self._fit_transform(X[sample]))
        self._remember(X[sample], Y_sample)

        rest = np.setdiff1d(np.arange(n), sample, assume_unique=True)
        Y = np.empty((n, Y_sample.shape[1]), dtype=Y_sample.dtype)
        Y[sample] = Y_sample
        Y[rest] = self.transform(X[rest])
        return Y

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(f"{self!r} is not fitted yet")
        X = _as_float(X)
        if self.supports_transform:
            return _as_float(self._transform(X))
        return self._knn_extend(X)

    # ---- helpers --------------------------------------------------------
    def _remember(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.fitted_ = True
        if not self.supports_transform:    # only needed for the kNN extension
            self._fit_points, self._fit_embedding = X, Y

    def _knn_extend(self, X: np.ndarray) -> np.ndarray:
        """Inverse-distance average of the nearest fitted points' embeddings."""
        from sklearn.neighbors import NearestNeighbors

        k = min(self.extension_neighbors, self._fit_points.shape[0])
        nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(self._fit_points)
        dist, idx = nn.kneighbors(X)
        w = 1.0 / np.maximum(dist, 1e-12)
        w /= w.sum(axis=1, keepdims=True)
        return np.einsum("nk,nkc->nc", w, self._fit_embedding[idx]).astype(np.float32)

    def __repr__(self) -> str:
        extra = f", fit_sample_size={self.fit_sample_size}" if self.fit_sample_size else ""
        return f"{type(self).__name__}(n_components={self.n_components}{extra})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Type[Reducer]] = {}


def register_reducer(*names: str) -> Callable[[Type[Reducer]], Type[Reducer]]:
    """Class decorator: make a ``Reducer`` subclass available to ``make_reducer``."""
    def deco(cls: Type[Reducer]) -> Type[Reducer]:
        for n in names:
            _REGISTRY[n.lower()] = cls
        cls.name = names[0]
        return cls
    return deco


def available_reducers() -> List[str]:
    return sorted(_REGISTRY)


ReducerSpec = Union[str, Reducer, Any, None]


def make_reducer(spec: ReducerSpec = "pca", **kwargs) -> Reducer:
    """Create a reducer from a name, a ``Reducer``, or a scikit-learn estimator.

    * ``"pca"``, ``"tsne"``, ... : look up the registry, pass ``kwargs`` on.
    * ``"pca>tsne"`` : a ``ChainReducer``. Per-step options go in a dict
      under the step's name, e.g. ``pca={"n_components": 50}``; other
      ``kwargs`` go to the last step.
    * ``None`` : the identity.
    * a ``Reducer`` : returned as is.
    * an object with ``fit_transform`` : wrapped in ``SklearnReducer``.
    """
    if spec is None:
        return IdentityReducer()
    if isinstance(spec, Reducer):
        return spec
    if isinstance(spec, str):
        if ">" in spec:
            steps = [s.strip() for s in spec.split(">")]
            per_step = {s: kwargs.pop(s, {}) for s in steps}
            per_step[steps[-1]] = {**kwargs, **per_step[steps[-1]]}
            return ChainReducer([make_reducer(s, **per_step[s]) for s in steps])
        key = spec.lower()
        if key not in _REGISTRY:
            raise ValueError(f"unknown reducer {spec!r}; available: {available_reducers()}")
        return _REGISTRY[key](**kwargs)
    if hasattr(spec, "fit_transform"):
        return SklearnReducer(spec, **kwargs)
    raise TypeError(f"cannot make a reducer from {spec!r}")


# ---------------------------------------------------------------------------
# Generic building blocks
# ---------------------------------------------------------------------------
@register_reducer("none", "identity")
class IdentityReducer(Reducer):
    """Keeps the points as they are (the baseline)."""

    def __init__(self, **kwargs):
        kwargs.pop("n_components", None)
        super().__init__(n_components=0, **kwargs)

    def _fit_transform(self, X):
        self.n_components = X.shape[1]
        return X

    def _transform(self, X):
        return X


class SklearnReducer(Reducer):
    """Wraps any estimator with scikit-learn's ``fit_transform`` API.

    Uses the estimator's ``transform`` for new points if it has one,
    otherwise the nearest-neighbour extension of the base class.
    """

    def __init__(self, estimator: Any, **kwargs):
        kwargs.setdefault("n_components", getattr(estimator, "n_components", None))
        super().__init__(**kwargs)
        self.estimator = estimator

    @property
    def supports_transform(self) -> bool:
        return hasattr(self.estimator, "transform")

    def _fit_transform(self, X):
        return self.estimator.fit_transform(X)

    def _transform(self, X):
        return self.estimator.transform(X)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.estimator!r})"


class ChainReducer(Reducer):
    """Applies several reducers one after the other."""

    def __init__(self, steps: Sequence[ReducerSpec], **kwargs):
        self.steps = [make_reducer(s) for s in steps]
        if not self.steps:
            raise ValueError("ChainReducer needs at least one step")
        super().__init__(n_components=self.steps[-1].n_components, **kwargs)

    @property
    def supports_transform(self) -> bool:
        return True       # every step can embed new points (natively or by kNN)

    def _fit_transform(self, X):
        for step in self.steps:
            X = step.fit_transform(X)
        return X

    def _transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X

    def __repr__(self) -> str:
        return " > ".join(repr(s) for s in self.steps)


# ---------------------------------------------------------------------------
# Concrete methods
# ---------------------------------------------------------------------------
@register_reducer("pca")
class PCAReducer(SklearnReducer):
    """Principal component analysis. ``n_components`` in (0, 1) keeps that
    fraction of the variance. ``explained_variance_ratio`` is available after
    fitting."""

    def __init__(self, n_components: Union[int, float] = 2, whiten: bool = False, **kwargs):
        from sklearn.decomposition import PCA

        rs = kwargs.get("random_state", 0)
        super().__init__(PCA(n_components=n_components, whiten=whiten, random_state=rs),
                         n_components=n_components, **kwargs)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.estimator.explained_variance_ratio_


@register_reducer("random_projection", "rp")
class RandomProjectionReducer(SklearnReducer):
    """Gaussian random projection (Johnson-Lindenstrauss)."""

    def __init__(self, n_components: int = 2, **kwargs):
        from sklearn.random_projection import GaussianRandomProjection

        rs = kwargs.get("random_state", 0)
        super().__init__(GaussianRandomProjection(n_components=n_components, random_state=rs),
                         n_components=n_components, **kwargs)


@register_reducer("kernel_pca", "kpca")
class KernelPCAReducer(SklearnReducer):
    """Kernel PCA (RBF by default). Needs an n x n kernel matrix, so set
    ``fit_sample_size`` (default 5000) for large data."""

    def __init__(self, n_components: int = 2, kernel: str = "rbf",
                 gamma: Optional[float] = None, fit_sample_size: Optional[int] = 5000,
                 **kwargs):
        from sklearn.decomposition import KernelPCA

        rs = kwargs.get("random_state", 0)
        super().__init__(KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma,
                                   random_state=rs, n_jobs=-1),
                         n_components=n_components, fit_sample_size=fit_sample_size, **kwargs)


@register_reducer("isomap")
class IsomapReducer(SklearnReducer):
    """Isomap: keeps geodesic (along-the-data) distances. Shortest paths cost
    O(n^2), so ``fit_sample_size`` defaults to 5000."""

    def __init__(self, n_components: int = 2, n_neighbors: int = 10,
                 fit_sample_size: Optional[int] = 5000, **kwargs):
        from sklearn.manifold import Isomap

        super().__init__(Isomap(n_components=n_components, n_neighbors=n_neighbors, n_jobs=-1),
                         n_components=n_components, fit_sample_size=fit_sample_size, **kwargs)


@register_reducer("tsne", "t-sne")
class TSNEReducer(SklearnReducer):
    """t-SNE (scikit-learn, Barnes-Hut). Keeps local neighbourhoods, which is
    what the kNN graph uses, but distorts larger distances. Has no transform:
    points outside ``fit_sample_size`` are placed by the kNN extension.
    Barnes-Hut supports at most 3 components."""

    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 init: str = "pca", fit_sample_size: Optional[int] = 20_000, **kwargs):
        from sklearn.manifold import TSNE

        rs = kwargs.get("random_state", 0)
        super().__init__(TSNE(n_components=n_components, perplexity=perplexity, init=init,
                              learning_rate="auto", random_state=rs, n_jobs=-1),
                         n_components=n_components, fit_sample_size=fit_sample_size, **kwargs)

    @property
    def supports_transform(self) -> bool:
        return False


@register_reducer("umap")
class UMAPReducer(SklearnReducer):
    """UMAP (optional dependency: ``pip install umap-learn``)."""

    def __init__(self, n_components: int = 2, n_neighbors: int = 15,
                 min_dist: float = 0.1, **kwargs):
        try:
            import umap
        except ImportError as e:
            raise ImportError("UMAPReducer needs umap-learn: pip install umap-learn") from e
        rs = kwargs.get("random_state", 0)
        super().__init__(umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                                   min_dist=min_dist, random_state=rs),
                         n_components=n_components, **kwargs)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def reduced_knn_graph(points: np.ndarray, k: int, method: ReducerSpec = "pca",
                      mode: str = "union", device: Optional[str] = None,
                      **reducer_kwargs) -> Tuple[NeighborGraph, np.ndarray]:
    """Reduce ``points`` with ``method``, then build the kNN graph there.

    Returns ``(graph, embedding)``. The vertices are the same points in the
    same order, so the graph goes straight into ``extremum_graph(values, g)``
    with the original values. The embedding is also handy as a 2D/3D
    scatterplot of the data.
    """
    reducer = make_reducer(method, **reducer_kwargs)
    Y = reducer.fit_transform(points)
    if Y.shape[1] > 10 and method not in (None, "none", "identity"):
        warnings.warn(f"embedding still has {Y.shape[1]} dimensions; "
                      "hubness may remain", stacklevel=2)
    return knn_graph(Y, k, mode=mode, device=device), Y


def _as_float(X: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(X), dtype=np.float32)
