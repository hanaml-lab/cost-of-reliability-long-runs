"""BELAY on the CPU toy setting: fit on shadow traces, plan, and compare policies on held-out toy jobs.

    python -m belay.demo [--config configs/toy.yaml]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from .toy import run_toy

NAMES = {"none": "no checks", "every": "critic after every step", "periodic": "critic every 4 steps",
         "indep": "independent-step program", "belay": "BELAY", "belay+replan": "BELAY, re-planning after flags"}


def config_path(name: str) -> Path:
    """``name`` from the working directory, else from the checkout this package is installed from."""
    path = Path(name)
    if not path.exists() and (Path(__file__).resolve().parents[2] / name).exists():
        return Path(__file__).resolve().parents[2] / name
    return path


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/toy.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(config_path(args.config).read_text())
    t0 = time.time()
    out = run_toy(seed=cfg["seed"], lam0=cfg["price"], trials=cfg["trials"], jobs=cfg["jobs"])
    print(f"{cfg['jobs'] // 2} held-out toy jobs, {cfg['trials']} simulated runs each; price {cfg['price']} per run of tokens")
    print(f"{'policy':<32}{'success':>8}{'tokens':>9}")
    for key, row in out.items():
        print(f"{NAMES[key]:<32}{row['success']:>8.3f}{row['tokens']:>8.2f}x")
    print(f"done in {time.time() - t0:.1f}s")
    return out


if __name__ == "__main__":
    main()
