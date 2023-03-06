import numpy as np
import copy
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

    # i_h = 2


    # -------------------------------------------------
    # Problem definition
    # -------------------------------------------------

    # mesh definition
    n_cells = 5

    mesh = (np.cos(np.linspace(np.pi, 0, n_cells+1)) + 1 )/2
    # mesh = np.linspace(0, 1, n_cells+1)

    # values at mesh-points
    def c(x):
        return 0.5 - x*(0.5-0.1)
    # conductivity = c(mesh)
    conductivity = np.ones(n_cells+1) * 1

    def r(x):
        return 0.5 - x*(0.5-0.1)
    # radius = r(mesh)
    radius = np.ones(n_cells+1) * 1

    # Boundary definition
    BCs = [
        ['dirichlet', 100],
        # ['neumann', 0.01],
        ['dirichlet', 50],
        # ['neumann', 0.1],
    ]

    # Numerical method
    # solver = 'numpy.linalg.solve'
    # solver = 'jacobi'
    # solver = 'gauss-seidel'
    solver = 'thomas'
    max_n = 1e4
    rel_conv_tolerance = 1e-12



    # -------------------------------------------------
    # Solving
    # -------------------------------------------------

    # add halo cells
    mesh_h = add_halo_mesh(mesh)
    conductivity_h = add_halo(conductivity)
    radius_h = add_halo(radius)

    area_h = radius_h**2 * np.pi

    x_cell_h = np.diff(mesh_h)/2 + mesh_h[:-1]
    dx_h = np.diff(x_cell_h)


    # Assemble matrices and vectors
    A = assemble_A(n_cells, conductivity_h, area_h, dx_h)
    B = np.zeros(n_cells)
    T_h = np.zeros(n_cells+2)


    # actually solve system
    T_h, R_norm_hist = solve(
        BCs, A, B, T_h,
        mesh_h, dx_h, conductivity_h, area_h,
        solver, max_n, rel_conv_tolerance
    )


    # -------------------------------------------------
    # Post Processing
    # -------------------------------------------------

    # plot result
    fig, axs = plt.subplots(3, 1)

    plot_problem(axs[0], mesh_h, conductivity_h, radius_h, BCs)
    # if possilbe, plot analytic solution
    if BCs[0][0] == 'dirichlet' and \
       BCs[1][0] == 'dirichlet' and \
       np.all(conductivity == conductivity[0]) and \
       np.all(radius == radius[0]):
        plot_analytic(axs[1], BCs, mesh_h)

    plot_temp(axs[1], T_h, x_cell_h, mesh_h)
    plot_res(axs[2], R_norm_hist)

    axs[1].legend()

    fig.suptitle(f'Solver: {solver}')
    fig.set_figwidth(10)
    fig.set_figheight(10)
    plt.tight_layout()
    plt.show()


def solve(
        BCs, A, B, T_h,
        mesh_h, dx_h, conductivity_h, area_h,
        solver, max_n, rel_conv_tolerance
):
    # calc initial norm (we need to set BCs for this)
    T_h = set_T_halo(BCs, T_h, dx_h)
    B_BCs = apply_BCs(B, T_h, conductivity_h, area_h, dx_h)
    R_initial = calc_res(A, B_BCs, T_h[1:-1], mesh_h[1:-1])
    R_norm_initial = calc_res_norm(R_initial)

    # solve for T
    rel_conv = 1
    n = 0
    R_norm_hist = [R_norm_initial]
    while True:

        # solve (ignore halo cells)
        if solver == 'jacobi':
            T_h[1:-1] = jacobi_step(A, B_BCs, T_h[1:-1])
        elif solver == 'gauss-seidel':
            T_h[1:-1] = gauss_seidel_step(A, B_BCs, T_h[1:-1])
        elif solver == 'thomas':
            T_h[1:-1] = thomas_step(A, B_BCs, T_h[1:-1])


        elif solver == 'numpy.linalg.solve':
            T_h[1:-1] = np.linalg.solve(A, B_BCs)

        # update boundary conditions
        T_h = set_T_halo(BCs, T_h, dx_h)
        B_BCs = apply_BCs(B, T_h, conductivity_h, area_h, dx_h)

        # Residuals and relative convergence
        R = calc_res(A, B_BCs, T_h[1:-1], mesh_h[1:-1])
        R_norm = calc_res_norm(R)
        rel_conv = R_norm / R_norm_initial
        R_norm_hist.append(R_norm)

        n += 1

        # cancel loop
        if n > max_n:
            print('Iteration limit reached.')
            break

        if rel_conv < rel_conv_tolerance:
            print(f'Converged after {n} iterations.')
            break

        if rel_conv > 1e10:
            print('Numerical method diverged.')
            break


    return T_h, R_norm_hist

def jacobi_step(A, B, T):
    T_new = np.zeros_like(T)

    for i in range(len(A)):
        sum_AT = 0
        for j in range(len(A)):
            if j == i:
                continue
            sum_AT += A[i, j]*T[j]

        T_new[i] = 1/A[i,i] * (B[i] - sum_AT)

    return T_new

def gauss_seidel_step(A, B, T):

    for i in range(len(A)):
        sum_AT = 0
        for j in range(len(A)):
            if j == i:
                continue
            sum_AT += A[i, j]*T[j]

        T[i] = 1/A[i,i] * (B[i] - sum_AT)

    return T

def thomas_step(A, B, T):

    # copy matrices
    AA = copy.copy(A)
    BB = copy.copy(B)

    # forward elimination phase
    for k in range(1, len(T)):
        # ak = k, k-1
        # bk-1 = k-1, k-1
        # ck-1 = k-1, k

        m = AA[k, k-1]/AA[k-1, k-1]

        AA[k, k] -= m*AA[k-1, k]
        BB[k] -= m*BB[k-1]

    # backwards substitution
    T[-1] = BB[-1] / AA[-1, -1]

    for k in range(len(T)-2, -1, -1):
        # ck = k, k+1

        T[k] = (BB[k] - AA[k, k+1]*T[k+1]) / AA[k, k]

    return T


def calc_res(A, B_BCs, T, mesh):

    # calc residuals
    R = np.dot(A, T) - B_BCs

    # scale by element width
    width = mesh[1:] - mesh[:-1]
    R_scaled = np.divide(R, width)

    return R_scaled

def calc_res_norm(R_scaled):
    R_norm = np.linalg.norm(R_scaled)

    return R_norm

def assemble_A(n_cells, conductivity_h, area_h, dx_h):
    A = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        i_h = i + 1

        # prepare values for neighboring cells
        left_cell, right_cell = 0, 0

        left_cell   = conductivity_h[i_h] * area_h[i_h] / dx_h[i]
        right_cell  = conductivity_h[i_h+1] * area_h[i_h+1] / dx_h[i+1]

        # assemble equation for current cell
        if i > 0:
            A[i, i-1]   += left_cell
        A[i, i]         -= left_cell + right_cell
        if i < n_cells-1:
            A[i, i+1]   += right_cell

    return A

def set_T_halo(BCs, T_h, dx_h):

    left, right = 0, 1
    name, value = 0, 1

    # calc temp in halo cells
    # left side
    if BCs[left][name] == 'dirichlet':
        dT_dx = (T_h[2] - T_h[1]) / dx_h[1]

        # set temp in halo cells
        T_h[0] = BCs[left][value] - dT_dx * dx_h[0]/2

    elif BCs[left][name] == 'neumann':
        T_h[0] = -2*BCs[left][value] + T_h[1]


    # right side
    if BCs[right][name] == 'dirichlet':
        dT_dx = (T_h[-2] - T_h[-3]) / dx_h[-2]

        # set temp in halo cells
        T_h[-1]= BCs[right][value] + dT_dx * dx_h[-1]/2

    elif BCs[right][name] == 'neumann':
        T_h[-1] = +2*BCs[right][value] + T_h[-2]

    return T_h

def apply_BCs(B, T_h, conductivity_h, area_h, dx_h):
    B_BCs = copy.copy(B)
    B_BCs[0] = - conductivity_h[1] * area_h[1] / dx_h[0] * T_h[0]
    B_BCs[-1] = - conductivity_h[-2] * area_h[-2] / dx_h[-1] * T_h[-1]

    return B_BCs


def add_halo_mesh(mesh):
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


def plot_problem(ax, mesh_h, conductivity_h, radius_h, BCs):


    # plot rod sizes
    p = ax.plot(mesh_h[1:-1], radius_h[1:-1]/2, 'k')
    ax.plot(mesh_h[1:-1], -radius_h[1:-1]/2, color=p[0].get_color())

    # plot conductivity
    ax_2 = ax.twinx()
    ax_2.plot(mesh_h[1:-1], conductivity_h[1:-1])

    # plot Boundarc conditions
    left, right = 0, 1
    name, value = 0, 1

    for i in [left, right]:
        text = ''
        if BCs[i][name] == 'dirichlet':
            text = f'$T = {BCs[i][value]} °K$'
        elif BCs[i][name] == 'neumann':
            text = '$\\frac{dT}{dx} = ' + str(BCs[i][value]) + \
                   '°K/m$'

        # left side
        x = 0 - 0.01
        va = 'bottom'

        # right side
        if i == right:
            x = mesh_h[-2] + 0.01
            va = 'top'

        # plot text
        ax.text(
            x, 0, text, rotation=90,
            rotation_mode='anchor',
            horizontalalignment='center',
            verticalalignment=va
        )


    # plot mesh
    plot_mesh_lines(ax, mesh_h)

    ax.set_title('Problem and Mesh definition')
    ax.set_xlabel('x-position [m]')
    ax.set_ylabel('y-position [m]')
    ax_2.set_ylabel('conductivity')

def plot_temp(ax, T_h, x_cell_h, mesh_h):
    # plot temp
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    p = ax.plot(x_cell_h[1:-1], T_h[1:-1], '.-', color=colors[0], label='Numeric')
    color = p[-1].get_color()

    # plot value at left boundary
    x = np.array([mesh_h[1], x_cell_h[1] ])
    y = np.array([(T_h[0] + T_h[1]) / 2, T_h[1]])
    ax.plot(x, y, '.--', color=color)

    # plot value at right boundary
    x = np.array([x_cell_h[-2], mesh_h[-2]])
    y = np.array([T_h[-2], (T_h[-1] + T_h[-2]) / 2])
    ax.plot(x, y, '.--', color=color)

    plot_mesh_lines(ax, mesh_h)

    ax.set_title('Temp distribution')
    ax.set_xlabel('x-position [m]')
    ax.set_ylabel('Temperature [°K]')

def plot_res(ax, R_norm_hist):
    ax.plot(R_norm_hist)
    ax.set_yscale('log')

    ax.set_title('Residual')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('RMS L2 norm')

def plot_mesh_lines(ax, mesh_h):
    # plot mesh
    mi, ma = ax.get_ylim()
    y = np.linspace(mi, ma, 2, endpoint=True)
    for n in range(1, len(mesh_h)-1):
        x = np.ones(2) * mesh_h[n]
        ax.plot(x, y, '--', color='lightgray')

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
