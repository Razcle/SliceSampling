""" Simple implementation of slice sampling """
import numpy as np
from math import floor


def step_out(density, y, x, w, m):
    """Expands the slice to find an interval I = [L, R]
    that captures as much of p(x|y) as possible
        density - function prop to p(x)
        y - auxiliary varibale
        x - current sample
        w - guess for average width
        m - maximum number of steps out
    returns (L, R)"""

    U = np.random.rand()
    L = x - U * w
    R = L + w
    V = np.random.rand()
    J = floor(m * V)
    K = m - 1 - J

    while J > 0 and y < density(L):
        L -= w
        J -= 1

    while K > 0 and y < density(R):
        R += w
        K -= 1

    return L, R


def step_in(density, y, x, L, R):
    """ Draws a sample form the interval [L, R] that
    is below the density, iteratively shrinking the interval as
    needed"""
    x_prop = np.random.uniform(L, R)
    if y < density(x_prop):
        return x_prop
    else:
        if x_prop < x:
            return step_in(density, y, x, x_prop, R)
        else:
            return step_in(density, y, x, L, x_prop)


def get_cond(density, x, d):
    """return a function to calculate the condtional
    density given the joint"""
    x = x.copy()

    def cond(x_d):
        x[d] = x_d
        return density(x)

    return cond


def slice_sample(density, x_init, num_samples, w, m):
    D = x_init.size
    x_new = x_init
    samples = [x_new]

    for i in range(num_samples):
        for d in range(D):
            cond = get_cond(density, x_new, d)
            y = np.random.uniform(0, cond(x_new[d]))
            L, R, = step_out(cond, y, x_new[d], w, m)
            x_new_d = step_in(cond, y, x_new[d], L, R)
            x_new = x_new.copy()
            x_new[d] = x_new_d
            samples.append(x_new)
    return np.array(samples)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def normal(x):
        return np.exp(-np.sum((x)**2)/(0.5))

    samples = slice_sample(normal, np.zeros(2), 10000, 10.0, 4)

    plt.figure()
    plt.hist2d(samples[:, 0], samples[:, 1], bins=100)
    plt.show()