import numpy as np
from numpy.typing import NDArray
from scipy.stats import kstest, kstwo, entropy, norm
from pandas import DataFrame


def calculate_entropy(sample: NDArray, bins, n, N):
    intervals = np.empty((n, bins + 1))
    emp_frequencies = np.empty((n, bins))

    for i in range(n):
        emp_frequencies[i], intervals[i] = np.histogram(sample[:, i], bins=bins, density=True)

    thr_frequencies = np.empty((n, bins))

    for i in range(n):
        thr_frequencies[i][0] = norm(0, 1).cdf(intervals[i][1])

    for i in range(n):
        for j in range(1, emp_frequencies[i].shape[0] - 1):
            thr_frequencies[i][j] = norm(0, 1).cdf(intervals[i][j + 1]) - norm(0, 1).cdf(intervals[i][j])

    for i in range(n):
        thr_frequencies[i][-1] = 1 - norm(0, 1).cdf(intervals[i][-2])

    val_entropy = [entropy(pk=thr_frequencies[i], qk=emp_frequencies[i]) for i in range(n)]

    return val_entropy


def calculate_statistic(sample: NDArray, n, N, bins):
    df = DataFrame([kstest(sample[:, i], 'norm', args=(0, 1)) for i in range(n)])
    df['cr_value'] = kstwo(N).isf(0.05)
    df['cr_pv'] = 0.05
    df['KL'] = calculate_entropy(sample, bins, n, N)
    df = df[['KL', 'statistic', 'cr_value', 'pvalue', 'cr_pv']]
    df.columns = ['KL', 'DN', 'кр DN', 'p-value', 'кр p-value']
    return df.round(3)
