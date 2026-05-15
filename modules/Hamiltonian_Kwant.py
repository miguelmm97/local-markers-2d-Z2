# %% modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.integrate import quad
import scipy.linalg as la
from functools import partial
from scipy.sparse import diags
from scipy.linalg import sqrtm
from numpy.linalg import eigh

# Kwant
import kwant
import tinyarray as ta
from kwant.kpm import jackson_kernel

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# %% Logging setup
loger_kwant = logging.getLogger('kwant')
loger_kwant.setLevel(logging.DEBUG)

stream_handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    '%(black)s%(asctime) -5s| %(blue)s%(name) -10s %(black)s| %(cyan)s %(funcName) '
    '-40s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'TRACE': 'black',
        'DEBUG': 'purple',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
stream_handler.setFormatter(formatter)
loger_kwant.addHandler(stream_handler)

# %% Module

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z



#%% BHZ Hamiltonian
def displacement2D(x1, y1, x2, y2):

    v = np.zeros((2,))
    v[0] = (x2 - x1)
    v[1] = (y2 - y1)

    # Norm of the vector between sites 2 and 1
    r = np.sqrt(v[0] ** 2 + v[1] ** 2)

    # Phi angle of the vector between sites 2 and 1 (angle in the XY plane)
    if v[0] == 0:                                    # Pathological case, separated to not divide by 0
        if v[1] > 0:
            phi = pi / 2                             # Hopping in y
        else:
            phi = 3 * pi / 2                         # Hopping in -y
    else:
        if v[1] > 0:
            phi = np.arctan2(v[1], v[0])             # 1st and 2nd quadrants
        else:
            phi = 2 * pi + np.arctan2(v[1], v[0])    # 3rd and 4th quadrants

    return r, phi

def displacement2D_kwant(site0, site1):
    x1, y1 = site0.pos[0], site0.pos[1]
    x2, y2 = site1.pos[0], site1.pos[1]

    v = np.zeros((2,))
    v[0] = (x2 - x1)
    v[1] = (y2 - y1)

    # Norm of the vector between sites 2 and 1
    r = np.sqrt(v[0] ** 2 + v[1] ** 2)

    # Phi angle of the vector between sites 2 and 1 (angle in the XY plane)
    if v[0] == 0:                                    # Pathological case, separated to not divide by 0
        if v[1] > 0:
            phi = pi / 2                             # Hopping in y
        else:
            phi = 3 * pi / 2                         # Hopping in -y
    else:
        if v[1] > 0:
            phi = np.arctan2(v[1], v[0])             # 1st and 2nd quadrants
        else:
            phi = 2 * pi + np.arctan2(v[1], v[0])    # 3rd and 4th quadrants

    return r, phi

def hopping(t, lamb, d, phi, cutoff_dist):
    f_cutoff = np.heaviside(cutoff_dist - d, 1) * np.exp(-d + 1)

    # U(1)-spin hoppings
    hopp_x = np.cos(phi) * (np.kron(tau_z, sigma_0) - 1j * np.kron(tau_x, sigma_z))
    hopp_y = np.sin(phi) * (np.kron(tau_z, sigma_0) - 1j * np.kron(tau_y, sigma_0))

    # U(1)-spin breaking
    mixing_x = 1j * 0.5 * lamb * np.cos(phi) * np.kron(tau_z, sigma_y)
    mixing_y = -1j * 0.5 * lamb * np.sin(phi) * np.kron(tau_z, sigma_x)

    return f_cutoff * (2 * t * (hopp_x + hopp_y) + mixing_x + mixing_y)

def spectrum(H, Nsp=None):
    """
    Input:
    H -> nd.array: Hamiltonian of the nanowire
    Nsp -> int: filling fraction

    Output:
    energy -> np.ndarray: sorted eigenvalues of H
    eigenstates -> np.ndarray: corresponding eigenvectors of H
    P -> np.ndarray: One particle density matrix (projector onto the filled bands)
    """
    if Nsp is None:
        Nsp = int(len(H) / 2)

    # Spectrum
    energy, eigenstates = np.linalg.eigh(H)
    idx = energy.argsort()
    energy = energy[idx]
    eigenstates = eigenstates[:, idx]

    # OPDM
    U = np.zeros((len(H), len(H)), dtype=np.complex128)
    U[:, 0: Nsp] = eigenstates[:, 0: Nsp]
    P = U @ np.conj(np.transpose(U))

    return energy, eigenstates, P

class FullyAmorphousWire_ScatteringRegion(kwant.builder.SiteFamily):
    def __init__(self, norbs, lattice, name=None):

        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError("The norbs parameter must be an integer > 0.")
            norbs = int(norbs)

        # Class fields
        loger_kwant.trace('Initialising cross section as a SiteFamily...')
        self.norbs = norbs
        self.coords = np.array([lattice.x, lattice.y]).T
        self.Nsites = lattice.Nsites
        self.Nx = lattice.Nx
        self.Ny = lattice.Ny
        self.name = name
        self.canonical_repr = "1" if name is None else name

    def pos(self, tag):
        return self.coords[tag, :][0, :]

    def normalize_tag(self, tag):
        return ta.array(tag)

    def __hash__(self):
        return 1

def BHZ_Hamiltonian_Kwant(lattice_tree, param_dict):

    # Load parameters into the builder namespace
    try:
        m = param_dict['m']
        t = param_dict['t']
        lamb = param_dict['lamb']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Create SiteFamily from the amorphous lattice
    latt = FullyAmorphousWire_ScatteringRegion(norbs=4, lattice=lattice_tree, name='bbh_model')

    # Hopping and onsite functions
    def onsite_potential(site):
        return m * np.kron(tau_z, sigma_0)

    def hopp(site1, site0):
        d, phi = displacement2D_kwant(site1, site0)
        return hopping(t, lamb, d, phi, lattice_tree.r)

    # Initialise kwant system
    loger_kwant.trace('Creating kwant scattering region...')
    syst = kwant.Builder()
    syst[(latt(i) for i in range(latt.Nsites))] = onsite_potential

    # Populate hoppings
    for i in range(latt.Nsites):
        for n in lattice_tree.neighbours[i]:
            loger_kwant.trace(f'Defining hopping from site {i} to {n}.')
            syst[(latt(n), latt(i))] = hopp

    return syst


#%% OPDM and density
def OPDM(eigenvectors, filling=0.5):

    loger_kwant.info('Calculating OPDM...')
    dim = eigenvectors.shape[0]
    Nsp = int(dim * filling)
    U = np.zeros((dim, dim), dtype=np.complex128)
    U[:, 0: Nsp] = eigenvectors[:, 0: Nsp]
    rho = U @ np.conj(np.transpose(U))
    return rho

def reduced_OPDM(rho, site_indices):

    Nred = len(site_indices) * 4
    rho_red = np.zeros((Nred, Nred), dtype=np.complex128)
    for i, site1 in enumerate(site_indices):
        for j, site2 in enumerate(site_indices):
            block1 = site1 * 4
            block2 = site2 * 4
            rho_red[i * 4: i * 4 + 4, j * 4: j * 4 + 4] = rho[block1: block1 + 4, block2: block2 + 4]

    return rho_red

def local_DoS(state, Nsites):

    local_DoS = np.zeros((Nsites, ), dtype=np.complex128)
    for i in range(Nsites):
        psi_i = state[i * 4: i * 4 + 4]
        local_DoS[i] = psi_i.T.conj() @ psi_i

    if np.sum(np.imag(local_DoS)) < 1e-10:
        local_DoS = np.real(local_DoS)
    else:
        raise TypeError('DoS is complex.')

    return local_DoS

def occupied_zero_energy_DoS(rho_eigvecs, H, Nsites, tol=1e-5, filling=0.5):
    zero_energy_Dos = np.zeros((Nsites, ), dtype=np.float64)
    for i in range(int(H.shape[0] * filling)):
        if np.allclose(np.abs(H @ rho_eigvecs[:, i]), np.zeros((H.shape[0], ), dtype=np.float64), atol=tol):
            zero_energy_Dos += local_DoS(rho_eigvecs[:, i], Nsites)
    return zero_energy_Dos



#%% Local marker ED
def local_marker(x, y, rho, S, spin_mixing=False, shift_per_site=False, Nx=None, Ny=None):
    """
    Input:
    x, y -> np.ndarray: coordinates of the lattice sites
    P -> np.ndarray: One particle density matrix (projector onto the filled bands)
    S -> np.ndarray: onsite auxiliary (kronecker product of onsite chiral symmetry)
    Spin-mixing -> bool: whether spin is conserved or not
    shift_per_site -> bool: Whether the marker is calculated at each site by shifting its position to the centre
    Nx -> int: number of sites along x
    Ny -> int: number of sites along y

    Output:
    local_marker -> np.ndarray: local z2 marker at each site
    """

    if spin_mixing:
        # vals, vecs = eigh(S @ rho + rho @ S - S)
        # PS = vecs @ np.diag(np.sign(vals)) @ vecs.T.conj()
        # C = - 1 / (32 * pi)
        I = np.eye(rho.shape[0], dtype=np.complex128)
        Q = I - 2 * rho
        Qp = 2 * (Q + S @ Q @ S)
        vals, vecs = eigh(Qp)
        Qp = vecs @ np.diag(np.sign(vals)) @ vecs.T.conj()
        rho = (I - Qp) / 2
        PS = rho @ S
        C = pi
    else:
        PS = rho @ S
        C = pi

    local_marker = np.zeros((len(x),))
    if shift_per_site:
        if Nx is None or Ny is None:
            loger_kwant.error('To shift sites Nx and Ny must be specified')
        for i in range(len(x)):
            loger_kwant.trace(f'Calculating marker at site: {i}/{len(x)-1}')
            X, Y = np.repeat(x - 0.5 * (Nx - 1), 4), np.repeat(y - 0.5 * (Ny - 1), 4)
            X = np.reshape(X, (len(X), 1))
            Y = np.reshape(Y, (len(Y), 1))
            XPS = X * PS
            YPS = Y * PS
            M = PS @ XPS @ YPS - PS @ YPS @ XPS
            idx = 4 * i
            local_marker[i] = C * np.trace(np.imag(M[idx: idx + 4, idx: idx + 4]))
    else:
        X, Y = np.repeat(x, 4), np.repeat(y, 4)
        X = np.reshape(X, (len(X), 1))
        Y = np.reshape(Y, (len(Y), 1))
        XPS = X * PS
        YPS = Y * PS
        M = PS @ XPS @ YPS - PS @ YPS @ XPS
        for i in range(len(x)):
            idx = 4 * i
            local_marker[i] = C * np.trace(np.imag(M[idx: idx + 4, idx: idx + 4]))

    return local_marker

def bulk_avg_marker(site_pos, local_marker, Nx, Ny, cutoff_x=0.25, cutoff_y=0.25):
    bool_x_right = site_pos[:, 0] < (Nx - 1) * (1 - cutoff_x)
    bool_x_left = site_pos[:, 0] > (Nx - 1) * cutoff_x
    bool_y_right = site_pos[:, 1] < (Ny - 1) * (1 - cutoff_y)
    bool_y_left = site_pos[:, 1] > (Ny - 1) * cutoff_y
    bool_site = bool_x_right & bool_x_left & bool_y_right & bool_y_left
    return np.sum(local_marker[bool_site]) / np.sum(bool_site)


#%% Local marker KPM
def kpm_vector_generator(H, state, max_moments):
    """
    Input:
    H -> np.ndarray: Hamiltonian
    state -> np.ndarray: state to which we apply the kpm expansion
    max_moments -> int: number of moments to use in the KPM expansion

    Output:
    np.ndarray: yields the state after each order of the expansion
    """

    # 0th moment in the expansion: Just the quantum state to which we are applying the operator
    alpha = state
    n = 0
    yield alpha

    # 1st moment in the expansion: Applying the Hamiltonian
    n += 1
    alpha_prev = alpha.copy()
    alpha = H @ alpha
    yield alpha

    # nth moments of the expansion: Follows by the recurrence of the Chebyshev polynomials
    n += 1
    while n < max_moments:
        alpha_save = alpha.copy()
        alpha = 2 * H @ alpha - alpha_prev
        alpha_prev = alpha_save
        yield alpha
        n += 1

def OPDM_KPM(state, num_moments, H, Ef=0, bounds=None):
    """
    Input:
    state -> np.ndarray: state to which we apply the kpm expansion
    num_moments -> int: number of moments to use in the KPM expansion
    H -> np.ndarray: Hamiltonian
    Ef -> float: Fermi energy at which to calculate the filled band projector (OPDM)
    bounds -> tuple of floats: Bounds for the spectrum, estimated if not provided

    Output:
    P_vec -> np.ndarray: OPDM applied onto the input state
    """

    # Rescaling of H and energies for the Kernel Polynomial Expansion
    num_moments = num_moments
    H_rescaled, (a, b) = kwant.kpm._rescale(H, 0.05, None, bounds)
    phi_f = np.arccos((Ef - b) / a)

    # Calculation of the coefficients in the expansion using the Jackson Kernel
    g = jackson_kernel(np.ones(num_moments))
    g[0] = 0
    m = np.arange(num_moments)
    m[0] = 1
    coefs = -2 * g * (np.sin(m * phi_f) / (m * np.pi))

    # Calculation of the OPDM (projector) applied onto vector as described in PRR 2, 013229 (2020)
    P_vec = (1 - phi_f/np.pi) * state + sum(c * vec for c, vec
                             in zip(coefs, kpm_vector_generator(H_rescaled, state, num_moments)))
    return P_vec

def local_marker_KPM_bulk(syst, S, Nx, Ny, Ef=0., num_moments=1000, num_vecs=5, bounds=None, cutoff=0.2):
    """
    Input:
    syst -> kwant.builder.Builder: Kwant closed system
    S -> scipy.sparse.csr_matrix: Auxiliary operator
    Nx, Ny -> int: Number of sites in each direction
    Ef -> float: Fermi energy at which to calculate the OPDM
    num_moments -> int: number of moments to use on the KPM expansion
    num_vecs -> int: number of random vectors to use in the stochastic trace evaluation
    bounds -> tuple of floats: Bounds for the spectrum, estimated if not provided
    cutoff -> float: fraction of the system we average over, taking the origin at the centre

    Output:
    np.complex128: Average bulk marker
    """

    # Region where we calculate the local marker
    project_to_region = partial(bulk_state, syst, lx=cutoff * Nx, ly=cutoff * Ny, Nx=Nx, Ny=Ny)

    # Operators involved in the calculation of the local marker
    H = syst.hamiltonian_submatrix(params=dict(flux=0., mu=0.), sparse=True).tocsr()
    P = partial(OPDM_KPM, num_moments=num_moments, H=H, Ef=Ef, bounds=bounds)
    [X, Y] = position_operator_OBC(syst, Nx, Ny)[0]

    # Calculation using the stochastic trace + KPM algorithm
    M = 0.
    for i in range(num_vecs):

        # Random initial state supported in the region that we trace over
        loger_kwant.info(f'Random vector {i}/ {num_vecs - 1}')
        state, Nsites = project_to_region(state=np.exp(2j * np.pi * np.random.random((H.shape[0]))))

        # Calculation of the invariant
        P_psi = P(state)
        if spin_mixing:
            I = np.eye(rho.shape[0], dtype=np.complex128)
            Q = I - 2 * rho
            Qp = 2 * (Q + S @ Q @ S)
            vals, vecs = eigh(Qp)
            if np.min(np.abs(vals)) < 1e-8:
                loger_kwant.warning('Qp has near-zero eigenvalues; marker may be ill-defined')
            Qp = vecs @ np.diag(np.sign(vals)) @ vecs.T.conj()
            rho = (I - Qp) / 2
        else:
            pass

        M = PS @ XPS @ YPS - PS @ YPS @ XPS

        P_psi = P(state)
        SP
        SP_psi = S @ P_psi
        PXP_psi, PYP_psi, PZP_psi = P(X @ P_psi),  P(Y @ P_psi),  P(Z @ P_psi)
        PXSP_psi, PYSP_psi, PZSP_psi = P(X @ SP_psi), P(Y @ SP_psi),  P(Z @ SP_psi)
        M +=  (Y @ PXSP_psi).T.conj() @ PZP_psi + (X @ PZSP_psi).T.conj() @ PYP_psi + (Z @ PYSP_psi).T.conj() @ PXP_psi
        M += -(Z @ PXSP_psi).T.conj() @ PYP_psi - (Y @ PZSP_psi).T.conj() @ PXP_psi - (X @ PYSP_psi).T.conj() @ PZP_psi

    return (8 * pi / 3) * np.imag(M) / (num_vecs * Nsites)

def bulk_state(syst, lx, ly, Nx, Ny, state):
    """
    Input:
    syst -> kwant.builder.Builder: Kwant closed system (no leads) for the nanowire
    lx, ly -> float: cutoff distances delimiting the bulk
    Nx, Ny -> int: Number of sites in each direction
    state -> np.ndarray: state to be restricted to the bulk sites

    Output:
    state -> np.ndarray: state with only support in the bulk sites
    Nsites -> int: number sites in the considered bulk
    """

    # Selecting a region on the bulk
    pos = np.array([s.pos for s in syst.sites])
    x_pos, y_pos = pos[:, 0] - 0.5 * (Nx-1), pos[:, 1] - 0.5 * (Ny-1)
    cond1 = np.abs(x_pos) < lx
    cond2 = np.abs(y_pos) < ly
    cond = cond1 * cond2
    Nsites = len(cond[cond])
    cond = np.repeat(cond, 4)

    # Weighted state on the bulk region
    state[~cond] = 0.
    return state, Nsites

def position_operator_OBC(syst, Nx, Ny):
    """
    Input:
      syst -> kwant.builder.Builder: Kwant closed system
      Nx, Ny -> int: Number of sites in each direction

    Output:
    operators -> list of np.ndarrays: Position operators
    pos -> np.ndarray: position of the sites
    """
    operators = []
    norbs = syst.sites[0].family.norbs
    pos = np.array([s.pos for s in syst.sites])
    for c in range(pos.shape[1]):
        if c==0:
            N = Nx
        else:
            N = Ny
        operators.append(diags(np.repeat(pos[:, c] - 0.5 * (N - 1), norbs), format='csr'))
    return operators, pos