from operator import itemgetter
import numpy as np
import copy
import time
import matplotlib.pyplot as plt


def main():
    # fourier's law
    #
    # Q = -kA dT/dx


    # Discretization setup
    #
    # c = cell
    # m = Mesh-point
    # A = Area
    # T = Temp (solution)
    #
    #                       n_cell = 3
    #           <----------User Defined---------->
    # |   c0    |    c1    |    c2    |    c3    |    c4    |
    # m0        m1         m2         m3         m4         m0
    # |  x_c0   |   x_c1   |   x_c2   |   x_c3   |   x_c4   |
    #      <--dx0---> <--dx1---> <--dx2---> <--dx3--->
    #
    # A0        A1         A2         A3         A4         A5
    #     T0    |    T1    |    T2    |    T3    |    T4    |


    # -------------------------------------------------
    # Problem definition
    # -------------------------------------------------

    # mesh definition
    n_cells = 1000
    mesh_points = ((np.cos(np.linspace(np.pi, 0, n_cells+1)) + 1 )/2 )

    # values at mesh-points
    def rad(x):
        left = 1
        right = 0.1
        return left - x*(left-right)
    radius = rad(mesh_points)

    def cond(x):
        left = 1
        right = 1
        return left - x*(left-right)
    conductivity = cond(mesh_points)

    def cap(x):
        left = 1
        right = 1
        return left - x*(left-right)
    capacity = cap(mesh_points)

    def rho(x):
        left = 1
        right = 1
        return left - x*(left-right)
    rho = rho(mesh_points)

    # Boundary definition
    BCs = [
        ['dirichlet', 100],
        # ['neumann', 0.01],
        ['dirichlet', 20],
        # ['neumann', 0.1],
    ]

    # Numerical method
    max_n = 1e4
    conv_tol = 1e-12

    problem = 'steady'
    # problem = 'pseudo-transient'


    # -------------------------------------------------
    # Solving
    # -------------------------------------------------

    # init mesh
    mesh = Mesh(
        mesh_points,
        radius,
        conductivity,
        capacity,
        rho
    )

    # init BCs
    boundary_conditions = BoundaryConditions(
        BCs[0][0], BCs[0][1],
        BCs[1][0], BCs[1][1],
    )

    # init system
    system = System(
        mesh,
        boundary_conditions,
        problem=problem,
    )

    system.assemble_A()
    system.assemble_B()


    # setup solver and solve
    solvers = [
        # SolverNumpy(copy.deepcopy(system), max_n, conv_tol),
        # SolverJacobi(copy.deepcopy(system), max_n, conv_tol),
        # SolverGaussSeidel(copy.deepcopy(system), max_n, conv_tol),
        # SolverThomas(copy.deepcopy(system), max_n, conv_tol),
        SolverNewton(copy.deepcopy(system), max_n, conv_tol, jac_type='FD'),
        SolverNewton(copy.deepcopy(system), max_n, conv_tol, jac_type='FD_col'),
    ]

    for solver in solvers:
        solver.solve()


    # -------------------------------------------------
    # Post Processing
    # -------------------------------------------------

    # plot result
    fig, axs = plt.subplots(3, 1)

    system.plot_system(axs[0])

    # if possible, plot analytic solution
    if boundary_conditions.left.bc_type == 'dirichlet' and \
       boundary_conditions.right.bc_type == 'dirichlet' and \
       np.all(conductivity == conductivity[0]) and \
       np.all(radius == radius[0]):
        plot_analytic(axs[1], BCs, mesh.p_x)

    # Residual + Solution
    for solver in solvers:
        solver.mesh.plot_T(axs[1], label=solver.name)
        solver.plot_R_hists(axs[2])
    axs[1].legend()
    mesh.plot_mesh(axs[1])
    axs[2].legend()

    # clean up figure
    fig.set_figwidth(10)
    fig.set_figheight(10)
    plt.tight_layout()
    plt.show()




class Mesh:
    def __init__(self, p_x, p_radius, p_conductivity, p_capacity, p_rho):
        # prefix p means a value at a grid point
        # prefix c means a value at the cell center

        #  point quantities
        self.p_x = self.add_halo_points(p_x)

        self.p_radius = self.add_halo(p_radius)
        self.p_area = self.p_radius**2 * np.pi

        self.p_conductivity = self.add_halo(p_conductivity)

        self.p_n = len(self.p_x)
        self.p_n_non_halo = self.p_n -2


        # cell quantities
        self.c_n = self.p_n - 1
        self.c_n_non_halo = self.c_n - 2

        self.c_x = np.diff(self.p_x)/2 + self.p_x[:-1]
        self.c_dx = np.diff(self.c_x)
        self.c_width = self.p_x[1:] - self.p_x[:-1]

        c_area = self.interp_p2c(self.p_area)
        self.c_volume = c_area * self.c_width
        self.c_rho = self.interp_p2c(self.add_halo(p_rho))
        self.c_capacity = self.interp_p2c(self.add_halo(p_capacity))
        self.c_conductivity = self.interp_p2c(self.add_halo(p_conductivity))


        # average quantities
        self.total_volume = np.sum(self.c_volume)
        self.av_rho = np.dot(self.c_rho, self.c_volume) / self.total_volume
        self.av_capacity = np.dot(self.c_capacity, self.c_volume) / self.total_volume
        self.av_conductivity = np.dot(self.c_conductivity, self.c_volume) / self.total_volume


        #index of non-halo cells
        self.c_ind = np.arange(self.c_n)[1:-1]
        self.p_ind = np.arange(self.c_n+1)[1:-1]

        # init Solution vector
        self.c_T = np.zeros(self.c_n)

    def plot_mesh(self, ax):
        # plot mesh
        mi, ma = ax.get_ylim()
        y = np.linspace(mi, ma, 2, endpoint=True)
        for n in self.p_ind:
            x = np.ones(2) * self.p_x[n]
            ax.plot(x, y, '--', color='lightgray', zorder=0)

    def plot_T(self, ax, label=None):
        # plot temp
        p = ax.plot(
            self.c_x[self.c_ind], self.c_T[self.c_ind],
            '.-', label=label,
        )
        color = p[-1].get_color()

        # plot value at left boundary
        x = np.array([self.p_x[1], self.c_x[1] ])
        y = np.array([(self.c_T[0] + self.c_T[1]) / 2, self.c_T[1]])
        ax.plot(x, y, '--', color=color)

        # plot value at right boundary
        x = np.array([self.c_x[-2], self.p_x[-2]])
        y = np.array([self.c_T[-2], (self.c_T[-1] + self.c_T[-2]) / 2])
        ax.plot(x, y, '--', color=color)

        ax.set_title('Temp distribution')
        ax.set_xlabel('x-position [m]')
        ax.set_ylabel('Temperature [°K]')


    @staticmethod
    def add_halo_points(mesh):
        """
        Adds a halo cell at the beginning and end by mirroring the first and last
        cell.

        This function is intended for meshes
        """

        mesh_h = np.zeros(len(mesh)+2)
        mesh_h[1:-1] = mesh
        mesh_h[0] = mesh[0] - (mesh[1] - mesh[0])
        mesh_h[-1] = mesh[-1] + mesh[-1] - mesh[-2]

        return mesh_h

    @staticmethod
    def add_halo(vec):
        """
        Adds a halo cell at the beginning and end by copying the first and last
        cell.

        This function is intended for scalar values
        """

        vec_h = np.zeros(len(vec)+2)
        vec_h[1:-1] = vec
        vec_h[0] = vec[0]
        vec_h[-1] = vec[-1]

        return  vec_h

    @staticmethod
    def interp_p2c(vec):
        """Interpolates from a point quantity to a cell quantity"""
        ret_vec = (vec[1:] + vec[:-1]) / 2
        return ret_vec


class BoundaryCondition:
    def __init__(self, bc_type, value):
        self.bc_type = bc_type
        self.value = value

class BoundaryConditions:
    def __init__(self, left_type, left_value, right_type, right_value):
        self.left = BoundaryCondition(left_type, left_value)
        self.right = BoundaryCondition(right_type, right_value)



class System:
    def __init__(self, mesh: Mesh, boundary_conditions: BoundaryConditions, problem='steady'):

        self.mesh = mesh
        self.BCs = boundary_conditions

        # residual vector
        self.R = np.zeros(self.mesh.c_n_non_halo)

        # steady = no transient part
        # pseudo-transient = steady solution, but using transient part
        # transient = only transient solution
        self.problem = problem

        self.dt = 1
        if self.problem == 'pseudo-transient':
            self.calc_optimal_timestep()

    def calc_optimal_timestep(self):
        L0 = np.max(self.mesh.p_x[self.mesh.p_ind]) - np.min(self.mesh.p_x[self.mesh.p_ind])

        alpha = self.mesh.av_conductivity / (self.mesh.av_capacity * self.mesh.av_rho)
        U0 = alpha / L0
        self.dt = L0 / U0

        print(f'Optimal timestep for pseudo-transient is: {self.dt}')


    def assemble_A(self):
        self.A_steady = np.zeros((self.mesh.c_n_non_halo, self.mesh.c_n_non_halo))

        self.A_unsteady = np.zeros_like(self.A_steady)

        for i in self.mesh.c_ind:
            # prepare values for neighboring cells
            left_cell, right_cell = 0, 0

            left_cell   = self.mesh.p_conductivity[i] * self.mesh.p_area[i] / \
                self.mesh.c_dx[i-1]
            right_cell  = self.mesh.p_conductivity[i+1] * self.mesh.p_area[i+1] / \
                self.mesh.c_dx[i]

            # unsteady part
            self.A_unsteady[i-1, i-1] += self.mesh.c_rho[i] * \
                self.mesh.c_capacity[i] * self.mesh.c_volume[i] / self.dt

            # assemble equation for current cell
            if i-1 > 0:
                self.A_steady[i-1, i-2]   += left_cell
            self.A_steady[i-1, i-1]         -= left_cell + right_cell
            if i-1 < self.mesh.c_n_non_halo-1:
                self.A_steady[i-1, i]   += right_cell


    def assemble_B(self):
        # steady sources
        self.B_steady = np.zeros(self.mesh.c_n_non_halo)

        # unsteady part
        self.B_unsteady = np.zeros_like(self.B_steady)


    def update_B_unsteady(self):
        self.B_unsteady = self.mesh.c_rho[self.mesh.c_ind] * \
            self.mesh.c_capacity[self.mesh.c_ind] * \
            self.mesh.c_volume[self.mesh.c_ind] * \
            self.mesh.c_T[self.mesh.c_ind] / self.dt

    @property
    def A(self):
        if self.problem == 'pseudo-transient':
            return self.A_steady + self.A_unsteady
        # steady
        return self.A_steady

    @property
    def B(self):
        if self.problem == 'pseudo-transient':
            return self.B_steady + self.B_unsteady

        #steady
        return self.B_steady


    def set_BC_values(self):
        # calc temp in halo cells
        # left side
        if self.BCs.left.bc_type == 'dirichlet':
            dT_c_dx = (self.mesh.c_T[2] - self.mesh.c_T[1]) / self.mesh.c_dx[1]

            # set temp in halo cells
            self.mesh.c_T[0] = self.BCs.left.value - dT_c_dx * self.mesh.c_dx[0]/2

        elif self.BCs.left.bc_type == 'neumann':
            self.mesh.c_T[0] = -2*self.BCs.left.value + self.mesh.c_T[1]


        # right side
        if self.BCs.right.bc_type == 'dirichlet':
            dT_c_dx = (self.mesh.c_T[-2] - self.mesh.c_T[-3]) / self.mesh.c_dx[-2]

            # set temp in halo cells
            self.mesh.c_T[-1]= self.BCs.right.value + dT_c_dx * self.mesh.c_dx[-1]/2

        elif self.BCs.right.bc_type == 'neumann':
            self.mesh.c_T[-1] = +2*self.BCs.right.value + self.mesh.c_T[-2]

    def apply_BCs(self):
        self.B_steady[0] = - self.mesh.p_conductivity[1] * self.mesh.p_area[1] / \
                    self.mesh.c_dx[0] * self.mesh.c_T[0]
        self.B_steady[-1] = - self.mesh.p_conductivity[-2] * self.mesh.p_area[-2] / \
                    self.mesh.c_dx[-1] * self.mesh.c_T[-1]


    def calc_res(self):
        self.R = self._calc_res(
            self.A_steady,
            self.mesh.c_T[self.mesh.c_ind],
            self.B_steady
        )

    def _calc_res(self, A, T, B):
        # calc residuals
        R = np.dot(A, T) - B

        # scale by element width
        R = np.multiply(R, self.mesh.c_width[1:-1])

        return R

    def calc_res_norm(self):
        R_norm = np.linalg.norm(self.R)

        return R_norm


    def calc_res_jac_fd(self):
        """Calculate the Jacobian of the residuals in a finite-difference way"""

        h = 1e-7

        JR = np.zeros((self.mesh.c_n_non_halo, self.mesh.c_n_non_halo))
        for i in range(self.mesh.c_n_non_halo):
            T = self.mesh.c_T[self.mesh.c_ind]
            T[i] += h

            R_step = self._calc_res(
                self.A_steady,
                T,
                self.B_steady
            )

            JR_col = (R_step - self.R) / h
            JR[:,i] = JR_col

        return JR

    def calc_res_jac_fd_col(self):
        """Calculate the Jacobian of the residuals in a finite-difference way.
        But it uses coloring for more efficiency"""

        stencil_size = 3
        h = 1e-7

        JR = np.zeros((self.mesh.c_n_non_halo, self.mesh.c_n_non_halo))
        for i in range(stencil_size):
            # figure out coloring indexes
            ind_color = np.arange(i, self.mesh.c_n_non_halo, stencil_size)

            # apply FD step
            T = self.mesh.c_T[self.mesh.c_ind]
            T[ind_color] += h

            # perform f(x)
            R_step = self._calc_res(
                self.A_steady,
                T,
                self.B_steady
            )
            JR_col = (R_step - self.R) / h

            # fill jacobian
            for ind in ind_color:
                i_start = max(ind - 1, 0)
                i_end = min(ind+2, self.mesh.c_n_non_halo)
                JR[i_start:i_end,ind] = JR_col[i_start:i_end]

        return JR


    def plot_system(self, ax):
        # plot rod sizes
        p = ax.plot(
            self.mesh.p_x[self.mesh.p_ind],
            self.mesh.p_radius[self.mesh.p_ind]/2,
            'k'
        )
        ax.plot(
            self.mesh.p_x[self.mesh.p_ind],
            -self.mesh.p_radius[self.mesh.p_ind]/2,
            color=p[0].get_color()
        )

        # plot p_conductivity
        ax_2 = ax.twinx()
        ax_2.plot(
            self.mesh.p_x[self.mesh.p_ind],
            self.mesh.p_conductivity[self.mesh.p_ind]
        )

        # # plot Boundarc conditions

        for i in range(2):
            # left side
            BC = self.BCs.left

            # right side
            if i == 1:
                BC = self.BCs.right

            text = ''
            if BC.bc_type == 'dirichlet':
                text = f'$T = {BC.value} °K$'
            elif BC.bc_type == 'neumann':
                text = '$\\frac{dT}{c_dx} = ' + str(BC.value) + \
                    '°K/m$'

            # left side
            x = np.min(self.mesh.p_x[self.mesh.p_ind]) - 0.01
            va = 'bottom'

            # right side
            if i == 1:
                x = np.max(self.mesh.p_x[self.mesh.p_ind][-1]) + 0.01
                va = 'top'

            # plot text
            ax.text(
                x, 0, text, rotation=90,
                rotation_mode='anchor',
                horizontalalignment='center',
                verticalalignment=va
            )

        # plot mesh
        self.mesh.plot_mesh(ax)

        ax.set_title('Problem and Mesh definition')
        ax.set_xlabel('x-position [m]')
        ax.set_ylabel('y-position [m]')
        ax_2.set_ylabel('p_conductivity')



class Solver:
    name = 'Solver'
    def __init__(self, system: System, max_n=100, conv_tol=1e-4):

        self.system = system
        self.mesh = system.mesh

        self.R_ref_norm = None
        self.R_norm_hist = list()

        self.max_n = max_n
        self.conv_tol = conv_tol

        self.rel_conv = 1
        self.n = 0

    def solve(self):

        t0 = time.time_ns()

        # apply boundary conditions
        self.system.set_BC_values()
        self.system.apply_BCs()

        # calculate initial residuals
        self.system.calc_res()
        self.R_ref_norm = self.system.calc_res_norm()

        # solve for T
        self.R_norm_hist.append(self.R_ref_norm)
        while True:

            # actually solve
            self.solve_step()

            # update boundary conditions
            self.system.set_BC_values()
            self.system.apply_BCs()

            # update unsteady Source terms
            if self.system.problem == 'pseudo-transient':
                self.system.update_B_unsteady()

            # Residuals and relative convergence
            self.system.calc_res()
            R_norm = self.system.calc_res_norm()
            self.rel_conv = R_norm / self.R_ref_norm
            self.R_norm_hist.append(R_norm)

            self.n += 1

            # cancel loop
            if self.n > self.max_n:
                print(f'{self.name}: Iteration limit reached.')
                break

            if self.rel_conv < self.conv_tol:
                print(f'{self.name}: Converged after {self.n} iterations.')
                break

            if self.rel_conv > 1e10:
                print(f'{self.name}: Numerical method diverged.')
                break

        tt = (time.time_ns() - t0) / 1e9
        print(f'{self.name}: Finished after {tt} seconds.')

    def solve_step(self):
        raise NotImplementedError

    def plot_R_hists(self, ax):
        ax.plot(self.R_norm_hist, label=self.name)
        ax.set_yscale('log')

        ax.set_title('Residual')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMS Heat Flux')


class SolverNumpy(Solver):
    name = 'numpy.linalg.solve'
    def solve_step(self):
        self.mesh.c_T[self.mesh.c_ind] = np.linalg.solve(
            self.system.A, self.system.B
        )

class SolverJacobi(Solver):
    name = 'Jacobi'
    def solve_step(self):
        T_new = np.zeros_like(self.mesh.c_T[self.mesh.c_ind])

        for i in range(self.mesh.c_n_non_halo):
            sum_AT = 0
            for j in range(self.mesh.c_n_non_halo):
                if j == i:
                    continue
                sum_AT += self.system.A[i, j]*self.mesh.c_T[self.mesh.c_ind][j]

            T_new[i] = 1/self.system.A[i,i] * (self.system.B[i] - sum_AT)

        self.mesh.c_T[self.mesh.c_ind] = T_new

class SolverGaussSeidel(Solver):
    name = 'Gauss-Seidel'
    def solve_step(self):

        T = self.mesh.c_T[self.mesh.c_ind]

        for i in range(self.mesh.c_n_non_halo):
            sum_AT = 0
            for j in range(self.mesh.c_n_non_halo):
                if j == i:
                    continue
                sum_AT += self.system.A[i, j]*T[j]

            T[i] = 1/self.system.A[i,i] * \
                (self.system.B[i] - sum_AT)

        self.mesh.c_T[self.mesh.c_ind] = T

class SolverThomas(Solver):
    name = 'Thomas'
    def solve_step(self):

        # copy matrices
        AA = copy.copy(self.system.A)
        BB = copy.copy(self.system.B)
        T = self.mesh.c_T[self.mesh.c_ind]

        # forward elimination phase
        for k in range(1, self.mesh.c_n_non_halo):
            # ak = k, k-1
            # bk-1 = k-1, k-1
            # ck-1 = k-1, k

            m = AA[k, k-1]/AA[k-1, k-1]

            AA[k, k] -= m*AA[k-1, k]
            BB[k] -= m*BB[k-1]

        # backwards substitution
        T[-1] = BB[-1] / AA[-1, -1]

        for k in range(self.mesh.c_n_non_halo-2, -1, -1):
            # ck = k, k+1

            T[k] = (BB[k] - AA[k, k+1] * T[k+1]) / AA[k, k]

        self.mesh.c_T[self.mesh.c_ind] = T

class SolverNewton(Solver):
    def __init__(self, system: System, max_n=100, conv_tol=0.0001, jac_type='FD'):
        super().__init__(system, max_n, conv_tol)

        self.jac_type = jac_type
        self.name = f'Newton_{jac_type}'

    def solve_step(self):

        # JR = np.zeros((self.mesh.c_n_non_halo, self.mesh.c_n_non_halo))
        if self.jac_type == 'FD':
            JR = self.system.calc_res_jac_fd()
        elif self.jac_type == 'FD_col':
            JR = self.system.calc_res_jac_fd_col()

        # JR_inv = JR
        JR_inv = np.linalg.inv(JR)

        self.mesh.c_T[self.mesh.c_ind] = self.mesh.c_T[self.mesh.c_ind] - \
             np.dot(JR_inv, self.system.R)


def plot_analytic(ax, BCs, mesh_h):

    T_l = BCs[0][1]
    T_r = BCs[1][1]

    l = mesh_h[-2] - mesh_h[1]
    x = np.linspace(0, l, 100)
    T = (T_r - T_l) / l * x + T_l

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.plot(x+mesh_h[1], T, '--', color=colors[1], label='Analytic')




if __name__ == '__main__':
    main()
