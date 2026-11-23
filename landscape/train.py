"""
Train the models whose weights span the landscape subspace.

    python3 -m landscape.train --models 4 --epochs 30            # 4 independent runs (seeds 0..3)
    python3 -m landscape.train --models 4 --spawn-epoch 2        # 4 children of one parent trained 2 epochs

Independent runs (different initialisations) end in different basins of the loss, separated by
high barriers. Children spawned from a shared parent after a few epochs end closer together, with
lower barriers (Frankle et al. 2020, linear mode connectivity), which gives a landscape with more
graded saddles. Each model is saved as ``data/loss_landscape/<run>/model_<i>.pt`` with its
training and test loss / accuracy in ``<run>/models.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from . import WORK, cifar
from .models import SmallCNN


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    augment: bool = True
    width: int = 32


class Trainer:
    """SGD with momentum, cosine learning-rate schedule, on GPU-resident CIFAR-10."""

    def __init__(self, cfg: TrainConfig, train: cifar.Split, test: cifar.Split, device: str):
        self.cfg, self.device = cfg, device
        self.train, self.test = train.to(device), test.to(device)

    def run(self, model: torch.nn.Module, seed: int, epochs: int, start_epoch: int = 0,
            total_epochs: int | None = None) -> torch.nn.Module:
        """Train ``model`` for ``epochs`` epochs; the schedule spans ``total_epochs``."""
        cfg, total = self.cfg, total_epochs or epochs
        model.to(self.device).train()
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda e: 0.5 * (1 + torch.cos(torch.tensor((start_epoch + e) / total * torch.pi))).item())
        g = torch.Generator(device=self.device).manual_seed(seed)
        n = len(self.train)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g, device=self.device)
            for i in range(0, n, cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                x = self.train.x[idx]
                if cfg.augment:
                    x = cifar.augment(x, g)
                loss = F.cross_entropy(model(x), self.train.y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
        return model.eval()

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, split: cifar.Split) -> dict:
        model.eval()
        loss = correct = 0.0
        for i in range(0, len(split), 2000):
            out = model(split.x[i:i + 2000])
            loss += F.cross_entropy(out, split.y[i:i + 2000], reduction="sum").item()
            correct += (out.argmax(1) == split.y[i:i + 2000]).sum().item()
        return {"loss": loss / len(split), "acc": correct / len(split)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--models", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--spawn-epoch", type=int, default=0,
                    help="0: independent runs; e > 0: train one parent e epochs, then the children")
    ap.add_argument("--run", default=None, help="output folder name (default from the options)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = TrainConfig(epochs=args.epochs)
    run = args.run or (f"cnn_m{args.models}_e{args.epochs}" + (f"_spawn{args.spawn_epoch}" if args.spawn_epoch else ""))
    out = WORK / run
    out.mkdir(parents=True, exist_ok=True)
    train, test = cifar.load()
    trainer = Trainer(cfg, train, test, args.device)
    arch = {"name": "SmallCNN", "width": cfg.width}

    parent = None
    if args.spawn_epoch:
        torch.manual_seed(1000)
        parent = trainer.run(SmallCNN(cfg.width), seed=1000, epochs=args.spawn_epoch, total_epochs=cfg.epochs)

    records = []
    for i in range(args.models):
        t0 = time.time()
        torch.manual_seed(i)
        if parent is None:
            model = trainer.run(SmallCNN(cfg.width), seed=i, epochs=cfg.epochs)
        else:
            model = trainer.run(copy.deepcopy(parent), seed=i, epochs=cfg.epochs - args.spawn_epoch,
                                start_epoch=args.spawn_epoch, total_epochs=cfg.epochs)
        rec = {"file": f"model_{i}.pt", "seed": i, "train": trainer.evaluate(model, trainer.train),
               "test": trainer.evaluate(model, trainer.test), "seconds": round(time.time() - t0, 1)}
        torch.save({"arch": arch, "state_dict": model.state_dict()}, out / rec["file"])
        records.append(rec)
        print(f"model {i}: train loss {rec['train']['loss']:.3f} acc {rec['train']['acc']:.3f} | "
              f"test loss {rec['test']['loss']:.3f} acc {rec['test']['acc']:.3f}  ({rec['seconds']}s)", flush=True)
    (out / "models.json").write_text(json.dumps(
        {"run": run, "arch": arch, "config": asdict(cfg), "spawn_epoch": args.spawn_epoch,
         "models": records}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
