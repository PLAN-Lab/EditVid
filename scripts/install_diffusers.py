#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

COMMIT = "769a1f3a120dd2a483b0e99bd6b13464b8ee62fb"
UPSTREAM = "https://github.com/huggingface/diffusers.git"


def run(*args, cwd=None):
    print("+", " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    vendor = root / ".vendor" / "diffusers"

    if not vendor.exists():
        vendor.parent.mkdir(parents=True, exist_ok=True)
        run("git", "init", vendor)
        run("git", "remote", "add", "origin", UPSTREAM, cwd=vendor)
        run("git", "fetch", "--depth", "1", "origin", COMMIT, cwd=vendor)
        run("git", "checkout", "--detach", "FETCH_HEAD", cwd=vendor)
    elif not (vendor / ".git").exists():
        raise RuntimeError(f"{vendor} is not a Git checkout")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
    if head != COMMIT:
        raise RuntimeError(f"Expected Diffusers {COMMIT}, found {head}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=vendor, text=True).strip():
        raise RuntimeError(f"Diffusers checkout is modified: {vendor}")

    if not args.no_install:
        run(sys.executable, "-m", "pip", "install", "--no-deps", "-e", vendor)
        run(sys.executable, "-m", "pip", "install", "--no-deps", "-e", root)
    print("Clean Diffusers prepared at", vendor)
    print("Pinned upstream commit:", COMMIT)


if __name__ == "__main__":
    main()
