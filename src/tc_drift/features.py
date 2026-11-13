"""Spline features for any number of assets (William). Phase 1: signature only.

In plain terms: the same additive-plus-pairwise cubic B-spline map as
src/features.py, for three assets instead of nine. It lives in a new file
because src/features.py hard-codes nine assets and is fingerprinted in every
function-approximation training manifest (src/train.py FA_CODE_FILES), so
editing it would flag every FA run as "training code changed". It will reuse
src.features.bspline_basis, which imports without changing that file.

Leakage boundary: reads only the weights it is given, never data.
"""

import numpy as np


def spline_features(W: np.ndarray, n_basis: int = 5, degree: int = 3) -> np.ndarray:
    """(m, n) long-only weights summing to one -> (m, k) nonnegative features:
    the n_basis basis functions of each weight, then every product of basis
    functions for each pair of assets, with all-zero columns dropped."""
    raise NotImplementedError("Phase 2 (William)")
