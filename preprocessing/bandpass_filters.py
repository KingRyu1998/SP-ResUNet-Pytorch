# -*- coding: utf-8 -*-
'''
@Time    : 2022/9/21 7:49
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : Sprog_CBAM_Unet
@File    : bandpass_filters.py
@Language: Python3
'''

import numpy as np

def filter_gaussian(
        shape,
        n,
        gauss_scale=0.5,
        d=1.0,
        normalize=True,
        include_mean=True,
        return_weight_funcs=False
):
    if n < 3:
        raise ValueError("n must be greater than 2")

    try:
        height, width = shape
    except TypeError:
        height, width = (shape, shape)

    max_length = max(width, height)

    rx = np.s_[: int(width / 2) + 1]

    if (height % 2) == 1:
        ry = np.s_[-int(height / 2): int(height / 2) + 1]
    else:
        ry = np.s_[-int(height / 2): int(height / 2)]

    y_grid, x_grid = np.ogrid[ry, rx]
    dy = int(height / 2) if height % 2 == 0 else int(height / 2) + 1

    r_2d = np.roll(np.sqrt(x_grid * x_grid + y_grid * y_grid), dy, axis=0)

    r_max = int(max_length / 2) + 1
    r_1d = np.arange(r_max)

    wfs, central_wavenumbers = _gaussweights_1d(
        max_length,
        n,
        gauss_scale=gauss_scale,
    )

    weights_1d = np.empty((n, r_max))
    weights_2d = np.empty((n, height, int(width / 2) + 1))

    for i, wf in enumerate(wfs):
        weights_1d[i, :] = wf(r_1d)
        weights_2d[i, :, :] = wf(r_2d)

    if normalize:
        weights_1d_sum = np.sum(weights_1d, axis=0)
        weights_2d_sum = np.sum(weights_2d, axis=0)
        for k in range(weights_2d.shape[0]):
            weights_1d[k, :] /= weights_1d_sum
            weights_2d[k, :, :] /= weights_2d_sum

    for i in range(len(wfs)):
        if i == 0 and include_mean:
            weights_1d[i, 0] = 1.0
            weights_2d[i, 0, 0] = 1.0
        else:
            weights_1d[i, 0] = 0.0
            weights_2d[i, 0, 0] = 0.0

    out = {"weights_1d": weights_1d, "weights_2d": weights_2d}
    out["shape"] = shape

    central_wavenumbers = np.array(central_wavenumbers)
    out["central_wavenumbers"] = central_wavenumbers

    # Compute frequencies
    central_freqs = 1.0 * central_wavenumbers / max_length
    central_freqs[0] = 1.0 / max_length
    central_freqs[-1] = 0.5  # Nyquist freq
    central_freqs = 1.0 * d * central_freqs
    out["central_freqs"] = central_freqs

    if return_weight_funcs:
        out["weight_funcs"] = wfs

    return out


def _gaussweights_1d(l, n, gauss_scale=0.5):
    q = pow(0.5 * l, 1.0 / n)
    r = [(pow(q, k - 1), pow(q, k)) for k in range(1, n + 1)]
    r = [0.5 * (r_[0] + r_[1]) for r_ in r]

    def log_e(x):
        if len(np.shape(x)) > 0:
            res = np.empty(x.shape)
            res[x == 0] = 0.0
            res[x > 0] = np.log(x[x > 0]) / np.log(q)
        else:
            if x == 0.0:
                res = 0.0
            else:
                res = np.log(x) / np.log(q)

        return res

    class GaussFunc:
        def __init__(self, c, s):
            self.c = c
            self.s = s

        def __call__(self, x):
            x = log_e(x) - self.c
            return np.exp(-(x ** 2.0) / (2.0 * self.s ** 2.0))

    weight_funcs = []
    central_wavenumbers = []

    for i, ri in enumerate(r):
        rc = log_e(ri)
        weight_funcs.append(GaussFunc(rc, gauss_scale))
        central_wavenumbers.append(ri)

    return weight_funcs, central_wavenumbers
