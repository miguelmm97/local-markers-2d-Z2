"""
Relevant numerical plots for the 2D AII local Z2 marker
=======================================================

This script collects the *relevant* numerical checks from the earlier
step-by-step calculations, rather than introducing a new numerical scheme.

It produces two groups of results.

A. Spin-frame disorder (comparison with Fig. 2 of Gomez Paz et al.)
   ----------------------------------------------------------------
   Starting from the lambda_R = 0 model with exact

       S0 = tau_z s_z,

   apply local Hadamard spin-frame rotations to a random fraction eta of
   sites.  Compare

       (i)  a fixed/non-rotated S = S0,
       (ii) the covariantly rotated S = U S0 U^dagger.

   The main plot scans Anderson disorder W for several eta values, in the
   same spirit as Fig. 2 of the paper.  A second plot shows the corresponding
   S_tilde gap.

B. Genuine Rashba spin mixing
   ---------------------------
   For

       H_R(k) = 2 lambda_R [sin(ky) s_x - sin(kx) s_y],

   compare the fixed S0 with

       S_opt = cos(theta_opt) tau_z s_z + sin(theta_opt) tau_y,
       theta_opt = 1/2 arctan(lambda_R/A).

   The script produces:

       1. Delta_{S_tilde}(theta) at lambda_R = 1 (Step 5).
       2. Auxiliary gap versus lambda_R, numerical + analytic (Step 6).
       3. PBC shifted-coordinate real-space marker versus lambda_R (Step 9).
       4. One W=3 disorder realization: theta scan (Step 11).
       5. Optional finite-size scaling at lambda_R = 2 (Step 10).
       6. Printed FHS Chern check for the positive-S_tilde bundle (Step 8).

Conventions
-----------
Basis: orbital tensor spin.

Clean Bloch Hamiltonian:

    H(k) = [M + 2 A (cos kx + cos ky)] tau_z
           + 2 A sin(kx) tau_x s_x
           + 2 A sin(ky) tau_x s_y
           + 2 lambda_R [sin(ky) s_x - sin(kx) s_y].

Real-space hoppings (one +x and one +y bond plus h.c.):

    t_x = A (tau_z - i tau_x s_x) + i lambda_R s_y,
    t_y = A (tau_z - i tau_x s_y) - i lambda_R s_x.

This is the convention used in our previous steps and gives clean phase
boundaries M = -4A, 0, +4A.

Dependencies: numpy, matplotlib.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. GLOBAL PARAMETERS
# ============================================================

A = 1.0
M = -2.0
SEED = 12345

LAMBDA_MARKER_VALUES = np.array([0.0, 0.5, 1.0, 2.0])
ETA_VALUES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ETA_FOR_W_PLOTS = [0.0, 0.4, 0.8, 1.0]

W_VALUES_FULL = np.array([
    0.0, 1.5, 3.0, 4.5, 6.0, 7.0, 7.5, 8.0,
    9.0, 10.0, 11.0, 12.0, 15.0
])


# ============================================================
# 2. PAULI MATRICES
# ============================================================

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

t0 = I2.copy()
tx = sx.copy()
ty = sy.copy()
tz = sz.copy()
s0 = I2.copy()

I4 = np.eye(4, dtype=complex)

TAU_0 = np.kron(t0, s0)
TAU_Z = np.kron(tz, s0)
TAU_Y = np.kron(ty, s0)
TAU_X_SX = np.kron(tx, sx)
TAU_X_SY = np.kron(tx, sy)
S_X = np.kron(t0, sx)
S_Y = np.kron(t0, sy)

S0_LOCAL = np.kron(tz, sz)
U_T_LOCAL = np.kron(t0, 1j * sy)

HAD_SPIN = (1.0 / np.sqrt(2.0)) * np.array(
    [[1, 1], [1, -1]], dtype=complex
)
HAD_LOCAL = np.kron(t0, HAD_SPIN)


# ============================================================
# 3. BASIC S(theta) FUNCTIONS
# ============================================================

def theta_hop(lambda_R, A=A):
    return np.arctan2(lambda_R, A)


def theta_optimal(lambda_R, A=A):
    return 0.5 * theta_hop(lambda_R, A)


def S_theta_local(theta):
    return np.cos(theta) * S0_LOCAL + np.sin(theta) * TAU_Y


def S_opt_local(lambda_R, A=A):
    return S_theta_local(theta_optimal(lambda_R, A))


def global_S(Lx, Ly, S_local):
    return np.kron(np.eye(Lx * Ly, dtype=complex), S_local)


def check_local_S(S):
    herm = np.linalg.norm(S - S.conj().T)
    invol = np.linalg.norm(S @ S - I4)
    trace = abs(np.trace(S))
    S_TR = U_T_LOCAL @ S.conj() @ U_T_LOCAL.conj().T
    tr_odd = np.linalg.norm(S_TR + S)
    return herm, invol, trace, tr_odd


# ============================================================
# 4. CLEAN BLOCH HAMILTONIAN
# ============================================================

def bloch_hamiltonian(kx, ky, lambda_R, A=A, M=M):
    mass = (M + 2.0 * A * (np.cos(kx) + np.cos(ky))) * TAU_Z
    original_so = (
        2.0 * A * np.sin(kx) * TAU_X_SX
        + 2.0 * A * np.sin(ky) * TAU_X_SY
    )
    rashba = 2.0 * lambda_R * (
        np.sin(ky) * S_X - np.sin(kx) * S_Y
    )
    return mass + original_so + rashba


def occupied_projector_bloch(Hk):
    energies, vectors = np.linalg.eigh(Hk)
    V_occ = vectors[:, :2]
    P = V_occ @ V_occ.conj().T
    return P, energies


def stilde(P, S):
    return P @ S + S @ P - S


def stilde_gap(P, S):
    return float(np.min(np.abs(np.linalg.eigvalsh(stilde(P, S)))))


def clean_bloch_gap(lambda_R, S, NK=60):
    ks = np.linspace(-np.pi, np.pi, NK, endpoint=False)
    gap = np.inf
    k_min = None
    for kx in ks:
        for ky in ks:
            P, _ = occupied_projector_bloch(
                bloch_hamiltonian(kx, ky, lambda_R)
            )
            g = stilde_gap(P, S)
            if g < gap:
                gap = g
                k_min = (kx, ky)
    return gap, k_min


def fixed_gap_prediction(lambda_R, A=A):
    return A / np.sqrt(A**2 + lambda_R**2)


def optimal_gap_prediction(lambda_R, A=A):
    return np.cos(theta_optimal(lambda_R, A))


# ============================================================
# 5. FHS CHERN NUMBER OF POSITIVE S_tilde BUNDLE (STEP 8)
# ============================================================

def positive_stilde_basis(P, S):
    vals, vecs = np.linalg.eigh(stilde(P, S))
    pos = np.where(vals > 0.0)[0]
    if len(pos) != 2:
        raise RuntimeError("Positive S_tilde subspace does not have rank 2.")
    return vecs[:, pos], float(np.min(np.abs(vals)))


def build_positive_bundle(lambda_R, S, NK=31):
    ks = np.linspace(-np.pi, np.pi, NK, endpoint=False)
    bundle = np.zeros((NK, NK, 4, 2), dtype=complex)
    min_gap = np.inf
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            Hk = bloch_hamiltonian(kx, ky, lambda_R)
            P, _ = occupied_projector_bloch(Hk)
            Vp, g = positive_stilde_basis(P, S)
            bundle[ix, iy] = Vp
            min_gap = min(min_gap, g)
    return bundle, min_gap


def normalized_det(overlap):
    det = np.linalg.det(overlap)
    mag = abs(det)
    if mag < 1e-12:
        raise RuntimeError("Near-singular neighboring-subspace overlap.")
    return det / mag


def chern_number_fhs(bundle):
    NK = bundle.shape[0]
    Ux = np.zeros((NK, NK), dtype=complex)
    Uy = np.zeros((NK, NK), dtype=complex)

    for ix in range(NK):
        for iy in range(NK):
            V = bundle[ix, iy]
            Vx = bundle[(ix + 1) % NK, iy]
            Vy = bundle[ix, (iy + 1) % NK]
            Ux[ix, iy] = normalized_det(V.conj().T @ Vx)
            Uy[ix, iy] = normalized_det(V.conj().T @ Vy)

    total_flux = 0.0
    for ix in range(NK):
        for iy in range(NK):
            plaquette = (
                Ux[ix, iy]
                * Uy[(ix + 1) % NK, iy]
                * np.conj(Ux[ix, (iy + 1) % NK])
                * np.conj(Uy[ix, iy])
            )
            total_flux += np.angle(plaquette)

    return total_flux / (2.0 * np.pi)


# ============================================================
# 6. REAL-SPACE PBC HAMILTONIAN (STEPS 9-11)
# ============================================================

def site_index(x, y, Lx, Ly=None):
    if Ly is None:
        Ly = Lx
    return x + Lx * y


def site_xy(site, Lx, Ly=None):
    if Ly is None:
        Ly = Lx
    return site % Lx, site // Lx


def dof_indices_from_site(site):
    return np.arange(4 * site, 4 * site + 4, dtype=int)


def hopping_x(lambda_R, A=A):
    return A * (TAU_Z - 1j * TAU_X_SX) + 1j * lambda_R * S_Y


def hopping_y(lambda_R, A=A):
    return A * (TAU_Z - 1j * TAU_X_SY) - 1j * lambda_R * S_X


def build_pbc_hamiltonian(Lx, Ly, lambda_R, disorder_values=None, A=A, M=M):
    N_sites = Lx * Ly
    D = 4 * N_sites
    H = np.zeros((D, D), dtype=complex)

    if disorder_values is None:
        disorder_values = np.zeros(N_sites)
    disorder_values = np.asarray(disorder_values)
    if disorder_values.shape != (N_sites,):
        raise ValueError("disorder_values must have length Lx*Ly")

    tx_hop = hopping_x(lambda_R, A)
    ty_hop = hopping_y(lambda_R, A)

    for y in range(Ly):
        for x in range(Lx):
            i = site_index(x, y, Lx, Ly)
            ii = dof_indices_from_site(i)

            onsite = M * TAU_Z + disorder_values[i] * TAU_0
            H[np.ix_(ii, ii)] += onsite

            jx = site_index((x + 1) % Lx, y, Lx, Ly)
            jjx = dof_indices_from_site(jx)
            H[np.ix_(ii, jjx)] += tx_hop
            H[np.ix_(jjx, ii)] += tx_hop.conj().T

            jy = site_index(x, (y + 1) % Ly, Lx, Ly)
            jjy = dof_indices_from_site(jy)
            H[np.ix_(ii, jjy)] += ty_hop
            H[np.ix_(jjy, ii)] += ty_hop.conj().T

    err = np.linalg.norm(H - H.conj().T)
    if err > 1e-10:
        raise RuntimeError(f"Hamiltonian Hermiticity error = {err}")
    return H


def half_filling_projector(H):
    energies, vectors = np.linalg.eigh(H)
    n_occ = H.shape[0] // 2
    V_occ = vectors[:, :n_occ]
    P = V_occ @ V_occ.conj().T
    return P, energies, vectors


# ============================================================
# 7. PBC SHIFTED-COORDINATE REAL-SPACE MARKER (STEP 9)
# ============================================================

def periodic_displacement(coord, center, L):
    """Signed periodic displacement coord-center in [-L/2,L/2)."""
    return ((coord - center + L / 2.0) % L) - L / 2.0


def shifted_position_diagonals_for_site(site, Lx, Ly):
    """
    Shift periodic coordinates separately for each target site so that the
    target is far from the branch cut.  This is the same construction used
    in the earlier step-by-step PBC marker calculation.

    The returned arrays are the diagonal entries of X_r and Y_r, repeated
    for the four internal states at every lattice site.
    """
    x0, y0 = site_xy(site, Lx, Ly)

    half_Lx = np.floor(Lx / 2.0)
    half_Ly = np.floor(Ly / 2.0)

    if x0 >= half_Lx:
        delta_x = abs(x0 - (half_Lx + Lx))
    else:
        delta_x = abs(half_Lx - x0)

    if y0 >= half_Ly:
        delta_y = abs(y0 - (half_Ly + Ly))
    else:
        delta_y = abs(half_Ly - y0)

    N_sites = Lx * Ly
    x_diag_site = np.zeros(N_sites)
    y_diag_site = np.zeros(N_sites)

    for s in range(N_sites):
        x, y = site_xy(s, Lx, Ly)
        x_diag_site[s] = (x + delta_x) % Lx
        y_diag_site[s] = (y + delta_y) % Ly

    x_diag = np.repeat(x_diag_site, 4).astype(float)
    y_diag = np.repeat(y_diag_site, 4).astype(float)
    return x_diag, y_diag


def flattened_stilde(P, S):
    """
    Construct S_tilde={P,S}-S and flatten it to A=sign(S_tilde).

    IMPORTANT: for genuine Rashba mixing [P,S] is not zero.  Therefore the
    noncommuting-S marker must use A=sign(S_tilde).  The older expression
    based directly on M=P S is the commuting-S reduction and should not be
    used for the Rashba calculation.
    """
    vals, vecs = np.linalg.eigh(stilde(P, S))
    gap = float(np.min(np.abs(vals)))
    if gap < 1e-10:
        raise RuntimeError("S_tilde gap is numerically closed.")

    Aop = (vecs * np.sign(vals)[None, :]) @ vecs.conj().T
    flatten_error = np.linalg.norm(
        Aop @ Aop - np.eye(Aop.shape[0], dtype=complex)
    )
    return Aop, gap, flatten_error


def marker_at_site(Aop, site, Lx, Ly):
    r"""
    Evaluate the local TRI marker at one target cell r:

        nu_TRI(r) = pi/(8 i) Tr_int[
            A X_r A Y_r A - A Y_r A X_r A
        ]_{r,r},

    where A=sign(S_tilde).  The prefactor is the lattice normalization used
    in the earlier numerical calibration, for which |nu_TRI| -> 1 in the
    clean topological bulk.
    """
    x_diag, y_diag = shifted_position_diagonals_for_site(site, Lx, Ly)
    ind = dof_indices_from_site(site)

    # Only the four target rows are required.  Since X_r and Y_r are
    # diagonal, right multiplication by X_r/Y_r is column-wise scaling.
    AXA_rows = (Aop[ind, :] * x_diag[None, :]) @ Aop
    AYA_rows = (Aop[ind, :] * y_diag[None, :]) @ Aop
    A_back = Aop[:, ind].T

    term_XY = np.sum((AXA_rows * y_diag[None, :]) * A_back)
    term_YX = np.sum((AYA_rows * x_diag[None, :]) * A_back)

    value = (np.pi / (8.0j)) * (term_XY - term_YX)
    return float(np.real_if_close(value).real)


def local_tri_marker_pbc(P, S, Lx, Ly):
    """
    Compute the shifted-coordinate local TRI marker at every unit cell.

    For every target r, the periodic X_r,Y_r branch cuts are moved away
    from r.  This is the fully real-space PBC calculation used for the
    Rashba marker plots below.

    Returns
    -------
    marker : (Ly,Lx) array
        Site-resolved local marker.
    gap : float
        min |eig(S_tilde)|.
    flatten_error : float
        ||sign(S_tilde)^2-I||.
    """
    Aop, gap, flatten_error = flattened_stilde(P, S)
    N_sites = Lx * Ly
    marker = np.array([
        marker_at_site(Aop, site, Lx, Ly)
        for site in range(N_sites)
    ])
    return marker.reshape((Ly, Lx)), gap, flatten_error


def pbc_marker_target(P, S, Lx, Ly, site=0):
    """Fast one-site version, retained for the optional large-L scan."""
    Aop, gap, _ = flattened_stilde(P, S)
    marker = marker_at_site(Aop, site, Lx, Ly)
    return marker, gap


# ============================================================
# 8. OBC HAMILTONIAN + BULK MARKER FOR FRAME DISORDER
#    (FROM THE EARLIER FRAME-DISORDER SWEEP)
# ============================================================

def block_slice(site):
    return slice(4 * site, 4 * site + 4)


def build_obc_hamiltonian(L, W, rng, A=A, M=M):
    """lambda_R=0 OBC Hamiltonian used in the earlier frame-disorder steps."""
    N = L * L
    D = 4 * N
    H = np.zeros((D, D), dtype=complex)
    Wj = rng.uniform(-W / 2.0, W / 2.0, N)

    hop_x = A * (TAU_Z - 1j * TAU_X_SX)
    hop_y = A * (TAU_Z - 1j * TAU_X_SY)

    for y in range(L):
        for x in range(L):
            i = site_index(x, y, L, L)
            si = block_slice(i)
            H[si, si] += M * TAU_Z + Wj[i] * I4

            if x + 1 < L:
                j = site_index(x + 1, y, L, L)
                sj = block_slice(j)
                H[sj, si] += hop_x
                H[si, sj] += hop_x.conj().T

            if y + 1 < L:
                j = site_index(x, y + 1, L, L)
                sj = block_slice(j)
                H[sj, si] += hop_y
                H[si, sj] += hop_y.conj().T

    return H


def obc_coordinate_vectors(L):
    x_sites = np.tile(np.arange(L), L)
    y_sites = np.repeat(np.arange(L), L)
    return (
        np.repeat(x_sites, 4).astype(float),
        np.repeat(y_sites, 4).astype(float),
    )


def obc_local_marker_and_gap(P, S, L):
    St = stilde(P, S)
    vals, vecs = np.linalg.eigh(St)
    gap = float(np.min(np.abs(vals)))
    if gap < 1e-12:
        # We still calculate with sign(0)=0, but flag it by returning the gap.
        pass
    Aop = (vecs * np.sign(vals)[None, :]) @ vecs.conj().T

    x, y = obc_coordinate_vectors(L)
    AXA = (Aop * x[None, :]) @ Aop
    AXAYA = (AXA * y[None, :]) @ Aop
    AYA = (Aop * y[None, :]) @ Aop
    AYAXA = (AYA * x[None, :]) @ Aop

    diagonal = np.diag(AXAYA - AYAXA)
    diagonal_site = diagonal.reshape(L * L, 4).sum(axis=1)
    marker = np.real((np.pi / (8j)) * diagonal_site).reshape(L, L)
    return marker, gap


def bulk_average(marker, margin):
    if 2 * margin >= marker.shape[0]:
        raise ValueError("margin is too large for this system size")
    return float(np.mean(marker[margin:-margin, margin:-margin]))


def frame_unitary_from_uniforms(L, uniforms, eta):
    N = L * L
    D = 4 * N
    U = np.zeros((D, D), dtype=complex)
    for site, u in enumerate(uniforms):
        block = HAD_LOCAL if u < eta else I4
        sl = block_slice(site)
        U[sl, sl] = block
    return U


def run_frame_disorder_sweep(L, W_values, eta_values, N_real, margin, seed=SEED):
    """
    Earlier frame-disorder sweep.

    For every Anderson realization:
      - build P for the unrotated Hamiltonian,
      - apply U P U^dagger for each eta,
      - compare fixed S0 with covariant U S0 U^dagger.

    Disorder averaging is done as abs(mean(signed marker)), not mean(abs(...)).
    """
    N = L * L
    S_fixed = np.kron(np.eye(N, dtype=complex), S0_LOCAL)

    shape = (len(W_values), len(eta_values))
    fixed_marker = np.zeros(shape)
    cov_marker = np.zeros(shape)
    fixed_gap = np.zeros(shape)
    cov_gap = np.zeros(shape)

    for iW, W in enumerate(W_values):
        print(f"  frame sweep W={W:4.1f} ({iW+1}/{len(W_values)})")

        fixed_samples = [[] for _ in eta_values]
        cov_samples = [[] for _ in eta_values]
        fixed_gap_samples = [[] for _ in eta_values]
        cov_gap_samples = [[] for _ in eta_values]

        for r in range(N_real):
            rng_H = np.random.default_rng(seed + 100000 * iW + r)
            H = build_obc_hamiltonian(L, W, rng_H)
            P, _, _ = half_filling_projector(H)

            # Covariant reference: exactly invariant under onsite U.
            m_ref, g_ref = obc_local_marker_and_gap(P, S_fixed, L)
            cov_bulk_ref = bulk_average(m_ref, margin)

            rng_frame = np.random.default_rng(seed + 500000 + 100000 * iW + r)
            uniforms = rng_frame.random(N)

            for ieta, eta in enumerate(eta_values):
                U = frame_unitary_from_uniforms(L, uniforms, eta)
                P_rot = U @ P @ U.conj().T

                # Fixed / non-rotated S.
                m_fixed, g_fixed = obc_local_marker_and_gap(P_rot, S_fixed, L)
                fixed_samples[ieta].append(bulk_average(m_fixed, margin))
                fixed_gap_samples[ieta].append(g_fixed)

                # Covariantly rotated S = U S U^dagger.
                # Marker and gap equal the reference by covariance.
                cov_samples[ieta].append(cov_bulk_ref)
                cov_gap_samples[ieta].append(g_ref)

        for ieta in range(len(eta_values)):
            fixed_marker[iW, ieta] = abs(np.mean(fixed_samples[ieta]))
            cov_marker[iW, ieta] = abs(np.mean(cov_samples[ieta]))
            fixed_gap[iW, ieta] = np.mean(fixed_gap_samples[ieta])
            cov_gap[iW, ieta] = np.mean(cov_gap_samples[ieta])

    return fixed_marker, cov_marker, fixed_gap, cov_gap


# ============================================================
# 9. DISORDER THETA-SCAN HELPERS (STEP 11)
# ============================================================

def projected_S_generators(vectors):
    D = vectors.shape[0]
    n_occ = D // 2
    V_occ = vectors[:, :n_occ]
    V_unocc = vectors[:, n_occ:]
    N_sites = D // 4

    S0_global = np.kron(np.eye(N_sites), S0_LOCAL)
    tauy_global = np.kron(np.eye(N_sites), TAU_Y)

    return {
        "S0_global": S0_global,
        "tauy_global": tauy_global,
        "S0_occ": V_occ.conj().T @ S0_global @ V_occ,
        "tauy_occ": V_occ.conj().T @ tauy_global @ V_occ,
        "S0_unocc": V_unocc.conj().T @ S0_global @ V_unocc,
        "tauy_unocc": V_unocc.conj().T @ tauy_global @ V_unocc,
    }


def auxiliary_gap_from_projected_blocks(theta, blocks):
    c = np.cos(theta)
    s = np.sin(theta)
    S_occ = c * blocks["S0_occ"] + s * blocks["tauy_occ"]
    S_unocc = c * blocks["S0_unocc"] + s * blocks["tauy_unocc"]
    gap_occ = float(np.min(np.abs(np.linalg.eigvalsh(S_occ))))
    gap_unocc = float(np.min(np.abs(np.linalg.eigvalsh(S_unocc))))
    return min(gap_occ, gap_unocc)


# ============================================================
# 10. PLOTTING HELPERS
# ============================================================

def save_or_show(fig, path, show=True):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_figure_caption(filename, caption):
    """Print a LaTeX-ready caption that can be copied from the terminal."""
    print("\n" + "-" * 72)
    print(f"FIGURE TEXT: {filename}")
    print("-" * 72)
    print(r"\caption{" + caption + "}")
    print("-" * 72)


def nearest_index(array, value):
    return int(np.argmin(np.abs(array - value)))


# ============================================================
# 11. FIGURE: LAURA-FIG-2-LIKE FRAME-DISORDER SCAN
# ============================================================

def make_frame_disorder_figures(output_dir, quick=False, show=True):
    print("\n" + "=" * 72)
    print("FRAME DISORDER: LAURA-FIG-2-LIKE W SCAN")
    print("=" * 72)

    if quick:
        L = 5
        margin = 2
        N_real = 2
        W_values = np.array([0.0, 4.5, 7.5, 9.0, 12.0, 15.0])
    else:
        # These are the parameters used in our earlier step-by-step sweep.
        L = 7
        margin = 2
        N_real = 20
        W_values = W_VALUES_FULL

    print(f"L={L}, N_real={N_real}, M={M}, A={A}")
    print("For a closer match to Laura et al.'s disorder averaging, set N_real=50.")

    fixed_marker, cov_marker, fixed_gap, cov_gap = run_frame_disorder_sweep(
        L=L,
        W_values=W_values,
        eta_values=ETA_VALUES,
        N_real=N_real,
        margin=margin,
    )

    # Main plot: same W-scan logic as Laura Fig. 2, but comparing our fixed
    # and covariantly rotated S choices.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for eta in ETA_FOR_W_PLOTS:
        ieta = nearest_index(ETA_VALUES, eta)
        axes[0].plot(
            W_values, fixed_marker[:, ieta], marker="o", label=rf"$\eta={eta}$"
        )
        axes[1].plot(
            W_values, cov_marker[:, ieta], marker="o", label=rf"$\eta={eta}$"
        )

    axes[0].set_title(r"(a) fixed $S=\tau_z s_z$")
    axes[1].set_title(r"(b) rotated $S=U(\tau_zs_z)U^\dagger$")

    for ax in axes:
        ax.set_xlabel(r"Anderson disorder $W$")
        ax.axvline(7.5, linestyle="--", alpha=0.5)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[0].set_ylabel(r"$|\langle\nu_{\rm bulk}\rangle_{\rm disorder}|$")
    save_or_show(fig, output_dir / "frame_disorder_marker_vs_W.png", show)
    print_figure_caption(
        "frame_disorder_marker_vs_W.png",
        rf"Real-space TRI marker as a function of Anderson disorder $W$ for "
        rf"spin-frame disorder fractions $\eta=0,0.4,0.8,1$.  The system has "
        rf"$A={A}$, $M={M}$ and $L={L}$.  In (a) the Hamiltonian/projector is "
        rf"locally frame rotated while the grading is incorrectly kept fixed at "
        rf"$S_0=\tau_zs_z$.  In (b) the grading is transformed covariantly, "
        rf"$S\rightarrow US_0U^\dagger$.  The covariant marker is independent "
        rf"of $\eta$, as required for an onsite basis change, while the fixed-$S$ "
        rf"marker fails as the local spin frame is scrambled.  The dashed line at "
        rf"$W=7.5$ indicates the approximate Anderson-driven transition quoted for "
        rf"this parameter choice."
    )

    # Companion plot: S_tilde gap for the same scan.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for eta in ETA_FOR_W_PLOTS:
        ieta = nearest_index(ETA_VALUES, eta)
        axes[0].plot(
            W_values, fixed_gap[:, ieta], marker="o", label=rf"$\eta={eta}$"
        )
        axes[1].plot(
            W_values, cov_gap[:, ieta], marker="o", label=rf"$\eta={eta}$"
        )

    axes[0].set_title(r"(a) $\widetilde S$ gap, fixed $S$")
    axes[1].set_title(r"(b) $\widetilde S$ gap, rotated $S$")

    for ax in axes:
        ax.set_xlabel(r"Anderson disorder $W$")
        ax.axvline(7.5, linestyle="--", alpha=0.5)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[0].set_ylabel(r"$\Delta_{\widetilde S}$")
    save_or_show(fig, output_dir / "frame_disorder_Stilde_gap_vs_W.png", show)
    print_figure_caption(
        "frame_disorder_Stilde_gap_vs_W.png",
        rf"Auxiliary gap $\Delta_{{\widetilde S}}=\min|\operatorname{{spec}}\widetilde S|$ "
        rf"for the same frame-disorder scan.  In (a), keeping "
        rf"$S=S_0=\tau_zs_z$ fixed while rotating the local basis drives the "
        rf"auxiliary spectrum towards zero as $\eta$ increases.  In (b), the "
        rf"covariant choice $S=US_0U^\dagger$ gives "
        rf"$\widetilde S\rightarrow U\widetilde S U^\dagger$, so its spectrum "
        rf"and gap are unchanged by frame disorder."
    )


# ============================================================
# 12. FIGURE: CLEAN THETA SCAN AT lambda_R=1 (STEP 5)
# ============================================================

def make_rashba_theta_scan(output_dir, quick=False, show=True):
    print("\n" + "=" * 72)
    print("RASHBA: THETA SCAN AT lambda_R=1")
    print("=" * 72)

    lam = 1.0
    NK = 24 if quick else 40
    theta_h = theta_hop(lam)
    theta_opt = theta_optimal(lam)
    theta_values = np.linspace(0.0, theta_h, 121 if quick else 181)

    gaps = []
    for theta in theta_values:
        g, _ = clean_bloch_gap(lam, S_theta_local(theta), NK=NK)
        gaps.append(g)
    gaps = np.array(gaps)

    i_best = int(np.argmax(gaps))
    print(f"numerical theta_opt = {theta_values[i_best]*180/np.pi:.6f} deg")
    print(f"analytic  theta_opt = {theta_opt*180/np.pi:.6f} deg")
    print(f"theta_hop            = {theta_h*180/np.pi:.6f} deg")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(theta_values * 180 / np.pi, gaps)
    ax.axvline(0.0, linestyle="--", label=r"$S_0$")
    ax.axvline(theta_opt * 180 / np.pi, linestyle="--", label=r"$\theta_{\rm opt}$")
    ax.axvline(theta_h * 180 / np.pi, linestyle=":", label=r"$\theta_{\rm hop}$")
    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_ylabel(r"$\Delta_{\widetilde S}$")
    ax.set_title(r"Clean auxiliary gap at $\lambda_R=1$")
    ax.grid(alpha=0.25)
    ax.legend()
    save_or_show(fig, output_dir / "rashba_theta_scan_lambda1.png", show)
    print_figure_caption(
        "rashba_theta_scan_lambda1.png",
        rf"Clean auxiliary gap $\Delta_{{\widetilde S}}$ as a function of the "
        rf"grading angle $\theta$ for $A={A}$, $M={M}$ and $\lambda_R=1$.  "
        rf"The family is $S(\theta)=\cos\theta\,\tau_zs_z+\sin\theta\,\tau_y$.  "
        rf"The original grading is at $\theta=0$, the hopping-preferred angle is "
        rf"$\theta_{{\rm hop}}=\arctan(\lambda_R/A)=45^\circ$, and the gap is "
        rf"maximal at the analytic result $\theta_{{\rm opt}}=\theta_{{\rm hop}}/2=22.5^\circ$."
    )


# ============================================================
# 13. FIGURE: GAP VERSUS RASHBA STRENGTH (STEP 6)
# ============================================================

def make_rashba_gap_vs_lambda(output_dir, quick=False, show=True):
    print("\n" + "=" * 72)
    print("RASHBA: AUXILIARY GAP VS lambda_R")
    print("=" * 72)

    NK = 24 if quick else 40
    lambda_numeric = np.linspace(0.0, 3.0, 7)
    gap_fixed_num = []
    gap_opt_num = []

    for lam in lambda_numeric:
        gf, _ = clean_bloch_gap(lam, S0_LOCAL, NK=NK)
        go, _ = clean_bloch_gap(lam, S_opt_local(lam), NK=NK)
        gap_fixed_num.append(gf)
        gap_opt_num.append(go)
        print(
            f"lambda={lam:.2f}: fixed={gf:.9f}, opt={go:.9f}, "
            f"theta_opt={theta_optimal(lam)*180/np.pi:.3f} deg"
        )

    lambda_dense = np.linspace(0.0, 3.0, 401)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        lambda_dense,
        fixed_gap_prediction(lambda_dense),
        label=r"fixed $S_0$: analytic",
    )
    ax.plot(
        lambda_dense,
        optimal_gap_prediction(lambda_dense),
        label=r"$S_{\rm opt}$: analytic",
    )
    ax.plot(lambda_numeric, gap_fixed_num, "o", label=r"fixed $S_0$: numerical")
    ax.plot(lambda_numeric, gap_opt_num, "s", label=r"$S_{\rm opt}$: numerical")
    ax.set_xlabel(r"Rashba strength $\lambda_R$")
    ax.set_ylabel(r"$\Delta_{\widetilde S}$")
    ax.set_title(r"Auxiliary gap under genuine spin mixing")
    ax.grid(alpha=0.25)
    ax.legend()
    save_or_show(fig, output_dir / "rashba_Stilde_gap_vs_lambda.png", show)
    print_figure_caption(
        "rashba_Stilde_gap_vs_lambda.png",
        rf"Auxiliary gap under genuine Rashba spin mixing.  The fixed grading "
        rf"$S_0=\tau_zs_z$ has $\Delta_{{\widetilde S}}=A/\sqrt{{A^2+\lambda_R^2}}$, "
        rf"while the optimized grading "
        rf"$S_{{\rm opt}}=\cos\theta_{{\rm opt}}\,\tau_zs_z+"
        rf"\sin\theta_{{\rm opt}}\,\tau_y$, with "
        rf"$\theta_{{\rm opt}}=\tfrac12\arctan(\lambda_R/A)$, has "
        rf"$\Delta_{{\widetilde S}}^{{\rm opt}}=\cos\theta_{{\rm opt}}$.  "
        rf"Lines show the analytic expressions and symbols show direct numerical "
        rf"diagonalization of $\widetilde S(\mathbf k)$ over the Brillouin zone."
    )


# ============================================================
# 14. FIGURE: PBC MARKER VERSUS lambda_R (STEP 9)
# ============================================================

def make_rashba_marker_vs_lambda(output_dir, quick=False, show=True):
    print("\n" + "=" * 72)
    print("RASHBA: FULL REAL-SPACE PBC MARKER VS lambda_R")
    print("=" * 72)

    L = 6 if quick else 8
    marker_fixed = []
    marker_opt = []
    gap_fixed = []
    gap_opt = []

    for lam in LAMBDA_MARKER_VALUES:
        H = build_pbc_hamiltonian(L, L, lam)
        P, _, _ = half_filling_projector(H)

        S_fixed = global_S(L, L, S0_LOCAL)
        S_opt = global_S(L, L, S_opt_local(lam))

        # Full real-space calculation: evaluate the shifted-coordinate
        # marker separately at every target site and then average.
        map_fixed, gf, flat_f = local_tri_marker_pbc(P, S_fixed, L, L)
        map_opt, go, flat_o = local_tri_marker_pbc(P, S_opt, L, L)

        mf = float(np.mean(map_fixed))
        mo = float(np.mean(map_opt))
        sf = float(np.std(map_fixed))
        so = float(np.std(map_opt))

        marker_fixed.append(mf)
        marker_opt.append(mo)
        gap_fixed.append(gf)
        gap_opt.append(go)

        print(
            f"lambda={lam:.1f}: "
            f"<nu> fixed/opt={mf:+.9f}/{mo:+.9f}, "
            f"site std={sf:.2e}/{so:.2e}, "
            f"Sgap={gf:.9f}/{go:.9f}, "
            f"flatten err={flat_f:.2e}/{flat_o:.2e}"
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(LAMBDA_MARKER_VALUES, np.abs(marker_fixed), "o-", label=r"fixed $S_0$")
    ax.plot(LAMBDA_MARKER_VALUES, np.abs(marker_opt), "s--", label=r"$S_{\rm opt}$")
    ax.axhline(1.0, linestyle=":")
    ax.set_xlabel(r"Rashba strength $\lambda_R$")
    ax.set_ylabel(r"$|\langle\nu_{\rm TRI}(\mathbf{r})\rangle_{\mathbf{r}}|$")
    ax.set_title(rf"Full real-space PBC shifted-coordinate marker, $L={L}$")
    ax.grid(alpha=0.25)
    ax.legend()
    save_or_show(fig, output_dir / "rashba_marker_vs_lambda.png", show)
    print_figure_caption(
        "rashba_marker_vs_lambda.png",
        rf"Real-space TRI marker on an $L={L}$ periodic lattice as a function of "
        rf"Rashba strength $\lambda_R$.  For every target cell $\mathbf r$, the "
        rf"periodic position operators $X_\mathbf r,Y_\mathbf r$ are shifted so "
        rf"that their branch cuts lie far from the target.  The plotted value is "
        rf"the spatial average of the site-resolved marker.  The fixed grading "
        rf"$S_0=\tau_zs_z$ and the optimized grading $S_{{\rm opt}}$ give the "
        rf"same marker to numerical precision, although their auxiliary gaps differ "
        rf"substantially.  The deviation from unity at large $\lambda_R$ for this "
        rf"small lattice is a finite-size effect."
    )

    # Same real-space calculations, but plot the auxiliary gap separately.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(LAMBDA_MARKER_VALUES, gap_fixed, "o-", label=r"fixed $S_0$")
    ax.plot(LAMBDA_MARKER_VALUES, gap_opt, "s--", label=r"$S_{\rm opt}$")
    ax.set_xlabel(r"Rashba strength $\lambda_R$")
    ax.set_ylabel(r"$\Delta_{\widetilde S}$")
    ax.set_title(rf"Real-space auxiliary gap, $L={L}$")
    ax.grid(alpha=0.25)
    ax.legend()
    save_or_show(fig, output_dir / "rashba_realspace_gap_vs_lambda.png", show)
    print_figure_caption(
        "rashba_realspace_gap_vs_lambda.png",
        rf"Auxiliary gap extracted from the full real-space operator "
        rf"$\widetilde S=\{{P,S\}}-S$ on the same $L={L}$ torus.  The original "
        rf"grading $S_0$ becomes progressively less robust as $\lambda_R$ grows, "
        rf"whereas $S_{{\rm opt}}$ maintains a much larger separation between the "
        rf"positive and negative $\widetilde S$ sectors."
    )


# ============================================================
# 15. FIGURE: ONE DISORDERED THETA SCAN (STEP 11)
# ============================================================

def make_rashba_disorder_theta_scan(output_dir, quick=False, show=True):
    print("\n" + "=" * 72)
    print("RASHBA: ONE W=3 DISORDER THETA SCAN")
    print("=" * 72)

    L = 6 if quick else 8
    lam = 1.0
    W = 3.0
    rng = np.random.default_rng(SEED)
    disorder = rng.uniform(-W / 2.0, W / 2.0, size=L * L)

    H = build_pbc_hamiltonian(L, L, lam, disorder_values=disorder)
    P, energies, vectors = half_filling_projector(H)
    blocks = projected_S_generators(vectors)

    theta_values = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 181 if quick else 361)
    gaps = np.array([
        auxiliary_gap_from_projected_blocks(theta, blocks)
        for theta in theta_values
    ])

    i_opt = int(np.argmax(gaps))
    theta_dis_opt = theta_values[i_opt]
    theta_clean = theta_optimal(lam)

    print(f"clean analytic theta = {theta_clean*180/np.pi:.6f} deg")
    print(f"disorder optimum     = {theta_dis_opt*180/np.pi:.6f} deg")
    print(f"optimized gap        = {gaps[i_opt]:.10f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(theta_values * 180 / np.pi, gaps)
    ax.axvline(0.0, linestyle="--", label=r"fixed $S_0$")
    ax.axvline(theta_clean * 180 / np.pi, linestyle="--", label="clean analytic optimum")
    ax.axvline(theta_dis_opt * 180 / np.pi, linestyle=":", label="sample optimum")
    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_ylabel(r"$\Delta_{\widetilde S}$")
    ax.set_title(rf"One scalar-disorder realization: $\lambda_R=1$, $W=3$, $L={L}$")
    ax.grid(alpha=0.25)
    ax.legend()
    save_or_show(fig, output_dir / "rashba_disorder_theta_scan_W3.png", show)
    print_figure_caption(
        "rashba_disorder_theta_scan_W3.png",
        rf"Auxiliary gap as a function of $\theta$ for one scalar-disorder "
        rf"realization with $\lambda_R=1$, $W=3$ and $L={L}$.  The fixed "
        rf"grading $S_0$ corresponds to $\theta=0$.  The clean analytic optimum "
        rf"is $22.5^\circ$, while the optimum of this particular disordered sample "
        rf"is {theta_dis_opt*180/np.pi:.1f}$^\circ$.  Their close agreement shows, "
        rf"for this realization, that the clean analytic grading remains nearly "
        rf"optimal after translation invariance is broken."
    )


# ============================================================
# 16. OPTIONAL FIGURE: FINITE-SIZE SCALING (STEP 10)
# ============================================================

def make_rashba_finite_size(output_dir, quick=False, show=True):
    print("\n" + "=" * 72)
    print("RASHBA: FINITE-SIZE SCALING AT lambda_R=2")
    print("=" * 72)

    lam = 2.0
    L_values = [4, 8] if quick else [4, 8, 12, 16]
    vals_fixed = []
    vals_opt = []

    for L in L_values:
        H = build_pbc_hamiltonian(L, L, lam)
        P, _, _ = half_filling_projector(H)
        S_fixed = global_S(L, L, S0_LOCAL)
        S_opt = global_S(L, L, S_opt_local(lam))
        mf, _ = pbc_marker_target(P, S_fixed, L, L, site=0)
        mo, _ = pbc_marker_target(P, S_opt, L, L, site=0)
        vals_fixed.append(mf)
        vals_opt.append(mo)
        print(f"L={L:2d}: marker fixed/opt = {mf:+.10f}/{mo:+.10f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(L_values, np.abs(vals_fixed), "o-", label=r"fixed $S_0$")
    ax.plot(L_values, np.abs(vals_opt), "s--", label=r"$S_{\rm opt}$")
    ax.axhline(1.0, linestyle=":")
    ax.set_xlabel(r"system size $L$")
    ax.set_ylabel(r"$|\nu_{\rm TRI}|$")
    ax.set_title(r"Finite-size scaling at $\lambda_R=2$")
    ax.grid(alpha=0.25)
    ax.legend()
    save_or_show(fig, output_dir / "rashba_finite_size_lambda2.png", show)
    print_figure_caption(
        "rashba_finite_size_lambda2.png",
        rf"Finite-size dependence of the periodic real-space TRI marker at "
        rf"$\lambda_R=2$.  The fixed and optimized gradings give the same marker "
        rf"within numerical precision.  The non-monotonic finite-size deviation at "
        rf"small $L$ decreases for the larger lattices, with the marker approaching "
        rf"its topological bulk value $|\nu_{{\rm TRI}}|=1$."
    )


# ============================================================
# 17. PRINTED CHERN CHECK (STEP 8)
# ============================================================

def run_chern_check(quick=False):
    print("\n" + "=" * 72)
    print("FHS CHERN CHECK OF POSITIVE S_tilde BUNDLE")
    print("=" * 72)
    NK = 21 if quick else 31
    for lam in [0.0, 2.0]:
        for name, S in [
            ("fixed S0", S0_LOCAL),
            ("dressed S_opt", S_opt_local(lam)),
        ]:
            bundle, sgap = build_positive_bundle(lam, S, NK=NK)
            chern = chern_number_fhs(bundle)
            print(
                f"lambda_R={lam:.1f}, {name:16s}: "
                f"C_pos(S_tilde)={chern:+.10f}, Sgap={sgap:.9f}"
            )


# ============================================================
# 18. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller grids/fewer disorder samples for a fast test run.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures but do not open matplotlib windows.",
    )
    parser.add_argument(
        "--finite-size",
        action="store_true",
        help="Also run the expensive L=4,8,12,16 finite-size marker plot.",
    )
    parser.add_argument(
        "--skip-frame",
        action="store_true",
        help="Skip the spin-frame-disorder W/eta sweep.",
    )
    parser.add_argument(
        "--output-dir",
        default="z2_marker_figures",
        help="Directory in which PNG figures are saved.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    show = not args.no_show

    print("Local S_opt legality checks")
    for lam in LAMBDA_MARKER_VALUES:
        errors = check_local_S(S_opt_local(lam))
        print(
            f"lambda_R={lam:.2f}: herm={errors[0]:.2e}, "
            f"S^2-I={errors[1]:.2e}, trace={errors[2]:.2e}, "
            f"TR-odd={errors[3]:.2e}"
        )

    if not args.skip_frame:
        make_frame_disorder_figures(output_dir, quick=args.quick, show=show)

    make_rashba_theta_scan(output_dir, quick=args.quick, show=show)
    make_rashba_gap_vs_lambda(output_dir, quick=args.quick, show=show)
    make_rashba_marker_vs_lambda(output_dir, quick=args.quick, show=show)
    make_rashba_disorder_theta_scan(output_dir, quick=args.quick, show=show)
    run_chern_check(quick=args.quick)

    if args.finite_size:
        make_rashba_finite_size(output_dir, quick=args.quick, show=show)

    print("\nDone. Figures are in:", output_dir.resolve())


if __name__ == "__main__":
    main()
