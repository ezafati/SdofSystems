"""
  Author: ZAFATI Eliass
          2021 
"""

from sdof import *
import sys


def main():
    """compute numerically det(SC)"""
    fig, ax = plt.subplots()
    plt.grid()
    gamma = 0.5
    beta = 0.25
    ratios = [5, 8, 10]
    lxi = np.linspace(0, 1, num=30)
    res = np.zeros(lxi.shape)

    # for post-treatement
    linestyles = [':', '-', '--', '-.']
    colors = ['red', 'black', 'green', 'blue']

    p1 = Sdofs(2)
    p2 = Sdofs(1)
    for ratio, linestyle, color in zip(ratios, linestyles, colors):
        i = 0
        for xi in lxi:
            # compute  omega*h which cancels e_mi
            x = compute_critical_omh(beta, ratio)

            # stiffness
            k = x ** 2 / p1.h ** 2
            p1.set_mat_prop([1, 0.01], [k, 10], [xi, xi])
            p1.update_newmark(gamma=gamma, beta=beta)

            p1.build_M_N()
            p1.ldofs = [0, 1]
            p1.build_L_B()

            p1.compute_A_global(ratio)
            p1.compute_global_L_B(ratio)
            H1 = p1.GB @ np.linalg.inv(p1.A) @ p1.GL.T

            # stiffness

            p2.set_mat_prop([1], [1], [0])
            p2.update_newmark(gamma=gamma, beta=beta)

            p2.build_M_N()
            p2.ldofs = [0, 1]
            p2.build_L_B()
            p2.update_L_B({0: 0})

            p2.compute_A_global(ratio)
            p2.compute_global_L_B(ratio)
            H2 = p2.GB @ np.linalg.inv(p2.A) @ p2.GL.T

            H = H1 + H2

            det_H = Sdofs.compute_determinant(H)
            res[i] = det_H
            i += 1
        ax.plot(lxi, res, color=color, linestyle=linestyle, label=r'$m = $' + str(ratio))
    plt.legend()
    plt.xlabel(r'$\xi_{12}$')
    plt.ylabel(r'$\det(SC)$')
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
    # plot_eta_e_curves(gamma=0.8, xi=0., beta=1)
    # plot_critical_omh()

