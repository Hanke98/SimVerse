#!/usr/bin/env python3
"""
Generate dense SPD block matrices and save to .bin.

Default output format is raw matrix payload in row-major:
  [block0 (n*n), block1 (n*n), ...]

Optional --with-header writes a small header before payload:
  magic(8 bytes) = b"SPDBIN01"
  uint32 version = 1
  uint32 dtype_code (1=float32, 2=float64)
  uint32 block_size
  uint32 num_blocks
  uint32 reserved = 0
  payload ...
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


MAGIC = b"SPDBIN01"
VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SPD matrices to .bin")
    parser.add_argument("--output", "-o", required=True, help="Output .bin path")
    parser.add_argument("--block-size", "-n", type=int, required=True, help="Matrix size n")
    parser.add_argument("--num-blocks", "-b", type=int, default=1, help="Number of blocks")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Element type",
    )
    parser.add_argument("--seed", type=int, default=20260320, help="Random seed")
    parser.add_argument(
        "--diag-min",
        type=float,
        default=5.0,
        help="Lower-triangular diagonal min",
    )
    parser.add_argument(
        "--diag-max",
        type=float,
        default=20.0,
        help="Lower-triangular diagonal max",
    )
    parser.add_argument(
        "--offdiag-scale",
        type=float,
        default=0.2,
        help="Off-diagonal scale for L",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Extra epsilon for A += eps * I",
    )
    parser.add_argument(
        "--with-header",
        action="store_true",
        help="Write metadata header before payload",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    n = args.block_size
    nb = args.num_blocks
    if n <= 0 or nb <= 0:
        raise ValueError("block-size and num-blocks must be > 0")

    dtype = np.float32 if args.dtype == "float32" else np.float64
    dtype_code = 1 if dtype is np.float32 else 2
    rng = np.random.default_rng(args.seed)

    mats = np.empty((nb, n, n), dtype=dtype)
    eye = np.eye(n, dtype=dtype)

    for k in range(nb):
        # Build a random lower-triangular matrix L with strictly positive diagonal.
        L = np.zeros((n, n), dtype=dtype)
        diag = rng.uniform(args.diag_min, args.diag_max, size=n).astype(dtype, copy=False)
        offdiag = rng.uniform(-args.offdiag_scale, args.offdiag_scale, size=(n, n)).astype(dtype, copy=False)
        li, lj = np.tril_indices(n, k=-1)
        L[li, lj] = offdiag[li, lj]
        np.fill_diagonal(L, diag)

        # SPD: A = L * L^T + eps * I
        A = L @ L.T
        if args.eps > 0.0:
            A = A + dtype(args.eps) * eye
        mats[k] = A

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        if args.with_header:
            f.write(MAGIC)
            f.write(struct.pack("<IIIII", VERSION, dtype_code, n, nb, 0))
        f.write(mats.tobytes(order="C"))

    bytes_written = out_path.stat().st_size
    print(
        f"[gen_spd_bin] wrote {out_path} | blocks={nb}, n={n}, dtype={args.dtype}, "
        f"header={args.with_header}, bytes={bytes_written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

