"""
  Author: ZAFATI Eliass
          2021 
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt


def compute_critical_omh(beta, ratio, l=1):
    t = np.cos(2 * np.pi * l / ratio)
    k = 2 * (1 - t)
    return np.sqrt(k / (1 - beta * k))


def compute_root(x, gamma, beta, xi):
    gamp = gamma + 2 * xi / x
    betp = beta + 2 * gamma * xi / x
    fact = np.power(x, 2) / (1 + betp * np.power(x, 2))
    re = 0.5 * (-(gamp + 0.5) * fact + 2)
    im = 0.5 * np.sqrt(4 * fact - np.power(fact, 2) * np.power(gamp + 0.5, 2))
    return np.complex(real=re, imag=im)


def compute_trigo_val(x, gamma, ratio):
    sum = 0
    for i in range(1, ratio):
        sum += np.power(x, i)
    return np.power(x, ratio) + 1 / gamma * sum


def compute_eig_val(x, xi, gamma, beta, ratio):
    """ add comments """
    h2 = 1
    h1 = h2 / ratio
    nu = h1 * gamma * np.reciprocal((1 + 2 * gamma * xi * x + beta * np.power(x, 2)))
    root = compute_root(x, gamma, beta, xi)
    e = np.imag(compute_trigo_val(root, gamma, ratio))
    e /= (ratio * np.imag(root))
    return nu * e


def plot_eta_e_curves(xi, gamma, beta):
    fig, ax = plt.subplots()
    plt.grid()
    ratios = [1, 2, 5, 10]
    linestyles = [':', '-', '--', '-.']
    colors = ['red', 'black', 'green', 'blue']
    interval = np.linspace(0, 2, num=100)
    res = np.zeros(interval.shape)
    for ratio, linestyle, color in zip(ratios, linestyles, colors):
        i = 0
        for x in interval:
            res[i] = compute_eig_val(x, xi, gamma, beta, ratio)
            i += 1
        ax.plot(interval, res, color=color, linestyle=linestyle, label=r'$m = $' + str(ratio))
    plt.xlabel(r'$\omega_{1i}h_1$')
    plt.ylabel(r'$\eta_{im}e_{im}(z_{im})$')
    plt.legend()
    plt.show()


def plot_critical_omh(beta=0):
    """add comments"""
    m_list = range(3, 100)
    res = []
    i = 0
    for ratio in m_list:
        for k in range(1, 1 + int(ratio / 2)):
            res.append(compute_critical_omh(beta, ratio, k))
            i += 1
    interval = np.array(res)
    res = np.ones((i,))
    fig, ax = plt.subplots()
    ax.plot(interval, res, 'o')
    plt.xlabel(r'$\omega_{1i}h_1$')
    plt.show()
    
