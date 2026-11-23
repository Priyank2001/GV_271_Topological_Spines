"""
Neural-network loss landscapes as high-dimensional scalar fields (Option A of the project plan).

A loss landscape is the training loss L(theta) of a network as a function of its weights.
Around trained weights it is studied in a low-dimensional affine subspace
theta(x) = origin + B x (x in R^d), which makes it an ordinary scalar field on R^d:
the extremum-graph pipeline (``pipeline/``) can then be run on samples of it.

Modules (each has one job; they meet only through the small types below):

    cifar.py      CIFAR-10 from the local pickles (no torchvision needed)
    models.py     the small CNN that is trained and probed
    train.py      train several models (``python3 -m landscape.train``)
    subspace.py   ``Subspace``: affine subspace of weight space, x <-> theta
    evaluator.py  ``LossEvaluator``: loss at many subspace points at once (torch.func.vmap)
    sampling.py   ``Box`` and samplers (grid, uniform, Sobol)
    make_dataset.py  write a landscape as a pipeline dataset (``python3 -m landscape.make_dataset``)
    accuracy.py   step 2: scattered kNN extremum graph vs a dense PL grid (``python3 -m landscape.accuracy``)

Files: trained models and sampled landscapes go to ``data/loss_landscape/`` (ignored by git);
viewer datasets to ``generator/datasets/<name>/``; results to ``comparison/results/``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "loss_landscape"
