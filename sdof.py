"""
  Author: ZAFATI Eliass
          2021
"""

from utils import *


class Sdofs:
    """ SDOF CLASS """

    def __init__(self, size=None, ldofs=None):
        self.ldofs = ldofs
        self._size = size
        self._m = None
        self._k = None
        self._c = None

        # default values for Newmark parameters
        self.gamma = 0.5
        self.beta = 0.
        self.h = 0.01

        self.L = None
        self.B = None
        self.map_dof_row = {}

    @property
    def size(self):
        return self._size

    def set_mat_prop(self, m: List, k: List, c: List):
        """add comments"""
        size = self.size
        if not m and k:
            raise Exception("Fatal error")
        assert len(m) == size and len(k) == size
        self._m = np.zeros((size, size), np.float32)
        self._k = np.zeros((size, size), np.float32)
        np.fill_diagonal(self._m, m)
        np.fill_diagonal(self._k, k)
        # not mandatory
        self._c = np.zeros([])
        if c:
            assert len(c) == self.size
            self._c = np.zeros((size, size), np.float32)
            np.fill_diagonal(self._c, c)

    def update_newmark(self, gamma=None, beta=None, h=None):
        """add comments """
        if gamma and isinstance(gamma, float):
            self.gamma = gamma
        if beta and isinstance(gamma, float):
            self.beta = beta
        if h and isinstance(gamma, float):
            self.h = h
        self.check_stability()

    def check_stability(self):
        """ to add later! be carefull
        numerical stability should be satisfied !!"""
        pass

    def build_M_N(self):
        """add comments """
        size = self.size
        M = np.zeros((3 * size, 3 * size), np.float64)
        N = np.zeros((3 * size, 3 * size), np.float64)

        # M FIRST
        # add mass and stiffness matrix
        M[0:size, 0:size] = self._m
        M[0:size, 2 * size:3 * size] = self._k
        if self._c.any():
            M[0:size, size:2 * size] = self._c

        # OTHER COMPONENTS
        M[size:2 * size, 0:size] = - self.gamma * self.h * np.identity(size)
        M[size:2 * size, size:2 * size] = np.identity(size)
        M[2 * size:3 * size, 0:size] = - self.beta * np.power(self.h, 2) * np.identity(size)
        M[2 * size:3 * size, 2 * size:3 * size] = np.identity(size)

        # MATRIX N
        N[size:2 * size, 0:size] = -(1 - self.gamma) * self.h * np.identity(size)
        N[size:2 * size, size:2 * size] = -np.identity(size)
        N[2 * size:3 * size, 0:size] = - (0.5 - self.beta) * np.power(self.h, 2) * np.identity(size)
        N[2 * size:3 * size, size:2 * size] = -self.h * np.identity(size)
        N[2 * size:3 * size, 2 * size:3 * size] = -np.identity(size)

        #
        self.M = M
        self.N = N

    def set_imposed_dofs(self, ldofs):
        """add comments"""
        self.ldofs = ldofs

    def build_L_B(self):
        if self.ldofs:
            ndofs = len(self.ldofs)
            self.L = np.zeros((ndofs, 3 * self.size), np.float64)
            for i, dof in enumerate(self.ldofs):
                self.L[i, dof] = 1
                self.map_dof_row[dof] = i
            self.B = np.zeros((ndofs, 3 * self.size), np.float64)
            for i, dof in enumerate(self.ldofs):
                self.B[i, self.size + dof] = 1
                self.map_dof_row[dof] = i

    def update_L_B(self, dval):
        """add comments"""
        if self.B is None or self.L is None:
            self.build_L_B()
        for dof, val in dval.items():
            self.L[self.map_dof_row[dof], dof] = val
        for dof, val in dval.items():
            self.B[self.map_dof_row[dof], self.size + dof] = val

    def compute_schur_part(self, ratio):
        invm = np.linalg.inv(self.M)
        ninvm = self.N @ invm
        mat = ninvm
        res = np.identity(3 * self.size)
        for p in range(1, ratio):
            # res += ((-1) ** p) * ((ratio - p) / ratio) * np.linalg.matrix_power(ninvm, p)
            res += ((-1) ** p) * ((ratio - p) / ratio) * mat
            mat = mat @ ninvm
        return self.B @ invm @ res @ self.L.T

    def compute_A_global(self, ratio):
        size = self.size
        ncol = 3 * size * ratio
        nrow = ncol
        mat = np.zeros((nrow, ncol), np.float64)
        for p in range(ratio):
            mat[p * 3 * size:(p + 1) * 3 * size, p * 3 * size:(p + 1) * 3 * size] = self.M
            if p > 0:
                mat[p * 3 * size:(p + 1) * 3 * size, (p - 1) * 3 * size:p * 3 * size] = self.N
        self.A = mat
        return True

    def compute_global_L_B(self, ratio):
        """add comments """
        ndofs = len(self.ldofs)
        self.GL = np.zeros((ndofs, 3 * self.size * ratio), np.float64)
        for p in range(ratio):
            self.GL[:, 3 * p * self.size:3 * (p + 1) * self.size] = (p + 1) / ratio * self.L
        self.GB = np.zeros((ndofs, 3 * self.size * ratio), np.float64)
        self.GB[:, 3 * (ratio - 1) * self.size:3 * ratio * self.size] = self.B

    @staticmethod
    def compute_determinant(mat):
        return np.linalg.det(mat)

"""
class PHSystem:
    def __init__(self, lsdofs=None):
        if isinstance(lsdofs, list):
            assert len(lsdofs) == 2, "Only two subdomains are accepted"
            lsdofs.sort(key=lambda p: p.h)
            self.lsdofs = lsdofs
            h1 = self.lsdofs[0].h
            h2 = self.lsdofs[1].h
            ratio = int(h2 / h1)
            self.lsdofs[0].update_newmark(h=h2 / ratio)
            size1 = self.lsdofs[0].size
            size2 = self.lsdofs[1].size
            sizeL = self.lsdofs[0].L.shape
            size = 3 * ratio * size1 + 3 * size2 + sizeL[0]

    def build_full_system(self):
        pass
"""

"""
gamma = 0.5
beta = 0.
ratio = 2
xi = 0.
x = compute_critical_omh(beta, ratio)
y = compute_root(x, gamma, beta, xi)
z = compute_trigo_val(y, gamma, ratio)
print(z)
# plot_eta_e_curves(xi, gamma, beta)
mass = 1
p1 = Sdofs(2)
k = x ** 2 / p1.h ** 2
p1.set_mat_prop([1, 1], [k, 1], [xi, 0])
p1.update_newmark(gamma=gamma)
p1.build_M_N()
p1.ldofs = [0, 1]
p1.build_L_B()
p1.update_L_B({1: 0})
res = p1.compute_schur_part(ratio)
bol = p1.compute_A_global(ratio)
p1.compute_global_L_B(ratio)
H = p1.GB @ np.linalg.inv(p1.A) @ p1.GL.T
print(res)
"""
