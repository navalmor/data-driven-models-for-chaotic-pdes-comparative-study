import itertools

import numpy as np
from scipy.integrate import odeint
from scipy.special import binom


# Preserved validated logic from the converted PyTorch utilities.
def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


def _poly_terms(X, poly_order):
    m, n = X.shape
    for degree in range(1, poly_order + 1):
        for combo in itertools.combinations_with_replacement(range(n), degree):
            term = X[:, combo[0]]
            for idx in combo[1:]:
                term = term * X[:, idx]
            yield term


def sindy_library(X, poly_order, include_sine=False):
    X = np.asarray(X)
    m, n = X.shape

    terms = [np.ones(m)]
    terms.extend(_poly_terms(X, poly_order))

    if include_sine:
        terms.extend(np.sin(X[:, i]) for i in range(n))

    return np.column_stack(terms)


def sindy_library_order2(X, dX, poly_order, include_sine=False):
    X_combined = np.concatenate((X, dX), axis=1)
    return sindy_library(X_combined, poly_order, include_sine)


def sindy_fit(RHS, LHS, coefficient_threshold):
    RHS = np.asarray(RHS)
    LHS = np.asarray(LHS)

    Xi = np.linalg.lstsq(RHS, LHS, rcond=None)[0]

    for _ in range(10):
        small = np.abs(Xi) < coefficient_threshold
        Xi[small] = 0

        for i in range(LHS.shape[1]):
            big = ~small[:, i]
            if not np.any(big):
                continue

            RHS_big = RHS[:, big]
            Xi[big, i] = np.linalg.lstsq(RHS_big, LHS[:, i], rcond=None)[0]

    return Xi


def _build_library_function(n, poly_order, include_sine):
    combos = []
    for degree in range(1, poly_order + 1):
        combos.extend(itertools.combinations_with_replacement(range(n), degree))

    def compute(x):
        terms = [1.0]

        for combo in combos:
            val = x[combo[0]]
            for idx in combo[1:]:
                val *= x[idx]
            terms.append(val)

        if include_sine:
            terms.extend(np.sin(x))

        return np.array(terms)

    return compute


def sindy_simulate(x0, t, Xi, poly_order, include_sine):
    x0 = np.asarray(x0)
    n = x0.size

    lib_fn = _build_library_function(n, poly_order, include_sine)

    def f(x, t):
        Theta = lib_fn(x)
        return Theta @ Xi

    return odeint(f, x0, t)


def sindy_simulate_order2(x0, dx0, t, Xi, poly_order, include_sine):
    x0 = np.asarray(x0)
    dx0 = np.asarray(dx0)

    n = x0.size
    l = Xi.shape[0]

    Xi_order1 = np.zeros((l, 2 * n))

    for i in range(n):
        Xi_order1[2 * (i + 1), i] = 1.0
        Xi_order1[:, i + n] = Xi[:, i]

    return sindy_simulate(
        np.concatenate((x0, dx0)),
        t,
        Xi_order1,
        poly_order,
        include_sine,
    )
