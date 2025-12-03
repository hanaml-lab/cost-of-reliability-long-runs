"""Run the CPU toy setting end to end and print success and tokens for each policy.

    python scripts/run_toy.py [--seed 0] [--price 0.02]
"""
import argparse

from belay.toy import run_toy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--price", type=float, default=0.02, help="lam0: success points per unchecked run of tokens")
    args = ap.parse_args()
    out = run_toy(seed=args.seed, lam0=args.price)
    print(f"{'policy':<13} {'success':>8} {'tokens':>8}")
    for name, row in out.items():
        print(f"{name:<13} {row['success']:>8.3f} {row['tokens']:>7.2f}x")


if __name__ == "__main__":
    main()
