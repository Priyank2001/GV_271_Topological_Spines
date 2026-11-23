"""Loss-landscape package (landscape/) and the d-dimensional Freudenthal grid graph."""
import itertools

import numpy as np
import pytest
import torch

from pipeline.neighbor_graph import freudenthal_grid_graph, grid_graph
from pipeline.extremum_graph import extremum_graph
from landscape.cifar import Split, crop
from landscape.evaluator import LossEvaluator
from landscape.models import SmallCNN
from landscape.sampling import Box, grid, sobol, uniform
from landscape.subspace import Subspace, flatten


# ---- Freudenthal grid -------------------------------------------------------------------------
@pytest.mark.parametrize("d,expected", [(1, 2), (2, 6), (3, 14), (4, 30)])
def test_freudenthal_interior_degree(d, expected):
    g = freudenthal_grid_graph((5,) * d)
    centre = np.ravel_multi_index((2,) * d, (5,) * d)
    assert g.degree[centre] == expected


def test_freudenthal_3d_is_ttk_grid_with_y_z_flipped():
    from anyeg.neighborhood import FREUDENTHAL_OFFSETS
    ttk = {(dx, -dy, -dz) for dx, dy, dz in FREUDENTHAL_OFFSETS}
    shape = (5, 5, 5)
    g = freudenthal_grid_graph(shape)
    c = np.ravel_multi_index((2, 2, 2), shape)
    ours = {tuple(np.array(np.unravel_index(n, shape)) - 2) for n in g.neighbors(c)}
    # grid axes are (z, y, x) in C order of (nz, ny, nx); TTK offsets are (dx, dy, dz)
    ours = {(o[2], o[1], o[0]) for o in ours}
    assert ours == ttk


def test_freudenthal_contains_axis_grid():
    a, f = grid_graph((4, 5, 3)), freudenthal_grid_graph((4, 5, 3))
    fe = set(map(tuple, f.edges))
    assert all(tuple(e) in fe for e in a.edges)


def test_freudenthal_diagonal_ridge_is_one_maximum():
    """A ridge along the main diagonal splits on the axis grid but not on the PL grid."""
    n = 12
    x = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3).astype(float)
    f = -np.linalg.norm(x - x.mean(1, keepdims=True), axis=1) + 1e-3 * x.sum(1)
    assert extremum_graph(f, freudenthal_grid_graph((n,) * 3)).num_peaks == 1


# ---- data ---------------------------------------------------------------------------------------
def test_crop_matches_slicing():
    pad = torch.randn(4, 3, 40, 40)
    dy, dx = torch.tensor([0, 8, 3, 5]), torch.tensor([8, 0, 2, 4])
    ref = torch.stack([pad[i, :, dy[i]:dy[i] + 32, dx[i]:dx[i] + 32] for i in range(4)])
    assert torch.equal(crop(pad, dy, dx), ref)


# ---- subspace -----------------------------------------------------------------------------------
def _models(m, seed=0):
    out = []
    for i in range(m):
        torch.manual_seed(seed + i)
        out.append(SmallCNN(width=4).state_dict())
    return out


def test_subspace_through_models_reproduces_them():
    states = _models(4)
    sub = Subspace.through(states)
    assert sub.dim == 3
    assert torch.allclose(sub.basis @ sub.basis.T, torch.eye(3), atol=1e-5)
    theta = sub.params(torch.as_tensor(sub.anchors, dtype=torch.float32))
    for t, s in zip(theta, states):
        assert torch.allclose(t, flatten(s), atol=1e-5)
    assert np.allclose(sub.coords(torch.stack([flatten(s) for s in states])).numpy(), sub.anchors, atol=1e-4)


def test_random_directions_are_orthonormal_and_keep_anchors():
    states = _models(3)
    sub = Subspace.through(states).with_random(2, states[0], seed=1)
    assert sub.dim == 4
    assert torch.allclose(sub.basis @ sub.basis.T, torch.eye(4), atol=1e-5)
    assert np.allclose(sub.anchors[:, 2:], 0)


# ---- evaluator ----------------------------------------------------------------------------------
def test_evaluator_matches_plain_forward():
    states = _models(3)
    torch.manual_seed(5)
    data = Split(torch.randn(37, 3, 32, 32), torch.randint(0, 10, (37,)))
    sub = Subspace.through(states)
    ev = LossEvaluator(SmallCNN(width=4), data, sub, l2=1e-3, device="cpu", param_batch=2, image_batch=10)
    x = np.vstack([sub.anchors, [[0.1, -0.2]]])
    got = ev(x)
    for i, xi in enumerate(x):
        m = SmallCNN(width=4)
        flat = sub.params(torch.as_tensor(xi[None], dtype=torch.float32))[0]
        m.load_state_dict(sub.layout.unflatten(flat))
        with torch.no_grad():
            out = m(data.x)
        ce = torch.nn.functional.cross_entropy(out, data.y).item()
        assert got["ce"][i] == pytest.approx(ce, rel=1e-5)
        assert got["loss"][i] == pytest.approx(ce + 0.5e-3 * (flat.double() ** 2).sum().item(), rel=1e-5)
        assert got["acc"][i] == pytest.approx((out.argmax(1) == data.y).float().mean().item())


# ---- sampling -----------------------------------------------------------------------------------
def test_box_and_samplers():
    box = Box.around(np.array([[0.0, 0.0], [1.0, 2.0]]), margin=0.5)
    assert np.allclose(box.lo, [-0.5, -1.0]) and np.allclose(box.hi, [1.5, 3.0])
    g = grid(box, (3, 4))
    assert g.shape == (12, 2) and np.allclose(g[0], box.lo) and np.allclose(g[-1], box.hi)
    assert np.allclose(g[1], [box.lo[0], box.lo[1] + box.size[1] / 3])      # C order: last axis fastest
    for s in (uniform(box, 64, 0), sobol(box, 64, 0)):
        assert s.shape == (64, 2) and (s >= box.lo).all() and (s <= box.hi).all()


# ---- slices -------------------------------------------------------------------------------------
def test_triple_planes_contain_their_models():
    from landscape.slices import triple_planes
    anchors = np.random.default_rng(0).normal(size=(5, 4))
    planes = triple_planes(anchors)
    assert len(planes) == 10
    for p in planes:
        assert np.allclose(p.basis @ p.basis.T, np.eye(2), atol=1e-10)
        for m in p.models:                                   # each model is on its plane, inside its box
            u = (anchors[m] - p.origin) @ p.basis.T
            assert np.allclose(p.to_subspace(u), anchors[m])
            assert (u >= p.box.lo).all() and (u <= p.box.hi).all()


def test_li_plane_is_filter_normalised_and_centred_on_the_model():
    state = _models(1)[0]
    sub = Subspace.li_plane(state, seed=3)
    assert torch.allclose(sub.params(torch.zeros(1, 2))[0], flatten(state))
    d = sub.layout.unflatten(sub.basis[0])
    for name, v in state.items():
        if v.dim() > 1:
            assert torch.allclose(d[name].flatten(1).norm(dim=1), v.flatten(1).norm(dim=1), rtol=1e-4)
        else:
            assert torch.count_nonzero(d[name]) == 0


# ---- family measures ----------------------------------------------------------------------------
def test_family_measures_on_two_wells():
    """f = -log L with two wells: the centre one (deep, at the origin) and a shallow one at (0.6, 0)."""
    from landscape.family import measures
    n = 61
    u = np.linspace(-1, 1, n)
    x = np.stack(np.meshgrid(u, u, indexing="ij"), -1).reshape(-1, 2)
    loss = 1.0 - 0.8 * np.exp(-np.sum(x ** 2, 1) / 0.05) - 0.4 * np.exp(-np.sum((x - [0.6, 0]) ** 2, 1) / 0.02)
    m = measures(-np.log(loss), x, freudenthal_grid_graph((n, n)), int(np.argmin(np.linalg.norm(x, axis=1))))
    assert m["minima@0.01"] == 2 and m["minima@0.1"] == 2
    assert m["centre_is_global"] and 0.3 < m["basin_frac"] < 1
    assert m["saddles@0.01"] == 1 and m["escape"] > 0.1
    assert 0.2 < m["smd"] < 0.6
