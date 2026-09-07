# Maths
import numpy as np
from numpy import pi
from numpy.linalg import eigh

# Kwant
import kwant

# Logging
import logging

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns

# Loading Julia's notes script as a module (the hyphen in the filename means it
# cannot be reached with a normal "import" statement)
import importlib.util
from pathlib import Path

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.amorphous_rashba import rashba_syst_Kwant
from modules.OPDM import spectrum, local_DoS
from modules.S_optimisation import rashba_bhz_S_tilde
from modules.logging_config import setup_logging

#%%
# ============================================================
# Logging setup
# ============================================================

setup_logging()
loger_main = logging.getLogger(__name__)

#%%
# ============================================================
# Load Julia's notes script
# ============================================================

julia_path = Path(__file__).resolve().parent.parent / 'base-code' / 'analytics-Julia' / 'Julias-notes-plots.py'
julia_spec = importlib.util.spec_from_file_location('julia_notes', julia_path)
julia_notes = importlib.util.module_from_spec(julia_spec)
julia_spec.loader.exec_module(julia_notes)

#%%
# ============================================================
# Open-boundary counterpart of Julia's PBC Hamiltonian
# ============================================================

def build_obc_hamiltonian_rashba(Lx, Ly, lambda_R, disorder_values, A, M):
    N_sites = Lx * Ly
    D = 4 * N_sites
    H_obc = np.zeros((D, D), dtype=complex)

    tx_hop = julia_notes.hopping_x(lambda_R, A)
    ty_hop = julia_notes.hopping_y(lambda_R, A)

    for y in range(Ly):
        for x in range(Lx):
            i = julia_notes.site_index(x, y, Lx, Ly)
            ii = julia_notes.dof_indices_from_site(i)
            H_obc[np.ix_(ii, ii)] += M * julia_notes.TAU_Z + disorder_values[i] * julia_notes.TAU_0

            if x + 1 < Lx:
                j = julia_notes.site_index(x + 1, y, Lx, Ly)
                jj = julia_notes.dof_indices_from_site(j)
                H_obc[np.ix_(ii, jj)] += tx_hop
                H_obc[np.ix_(jj, ii)] += tx_hop.conj().T

            if y + 1 < Ly:
                j = julia_notes.site_index(x, y + 1, Lx, Ly)
                jj = julia_notes.dof_indices_from_site(j)
                H_obc[np.ix_(ii, jj)] += ty_hop
                H_obc[np.ix_(jj, ii)] += ty_hop.conj().T

    return H_obc

#%%
# ============================================================
# Code parameters
# ============================================================
# Kept small so this runs in seconds under pytest. Run this file directly
# (python integration_test_PBC_vs_OBC.py) for the full-resolution version with
# plots -- see the "if __name__" guard at the end of the Figures section

M                 = -2.
W                 = 0.
A                 = 1.
lambR             = 2
width             = 0.1
r                 = 1.3
Nx                = 8
Ny                = 8
crystalline       = True
params_dict = {'M': M, 'W': W, 'A': A, 'lambR': lambR}
dim_Hext = Nx * Ny
dim_Hint = 4
seed     = 12345
Ntheta_scan = 41
Nmin = 8
Nmax = 10

# Resolution of each theta scan
Ntheta_OBC = 41
Ntheta_PBC = 121
Nk         = 20
theta_hop = julia_notes.theta_hop(lambR, A=A)
theta_opt = julia_notes.theta_optimal(lambR, A=A)


#%%
# ============================================================
# Main: gap vs theta for the five methods (PBC bloch, PBC lattice, PBC kwant,
# OBC lattice, OBC kwant), plus the S_tilde spectrum at each method's own
# optimal angle
# ============================================================

# --- OBC kwant
loger_main.info('Building OBC lattice with kwant: ...')
lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice.seed = seed
lattice.build_lattice(crystalline=crystalline)
lattice.generate_onsite_disorder(K_onsite=0.5 * W)
rashba_model = rashba_syst_Kwant(lattice, params_dict).finalized()
site_pos = np.array([site.pos for site in rashba_model.id_by_site])
H = rashba_model.hamiltonian_submatrix()
eps, eigenvectors, rho = spectrum(H)
loger_main.info('Building OBC lattice with kwant: Done')

theta_vec_OBC = np.linspace(-pi / 2, pi / 2, Ntheta_OBC)
gap_OBC_kwant = np.zeros(theta_vec_OBC.shape)
for i, theta in enumerate(theta_vec_OBC):
    loger_main.trace(f'S_tilde gap at angle {i}/{Ntheta_OBC} for OBC kwant')
    _, gap_OBC_kwant[i], _ = rashba_bhz_S_tilde(rho, theta, dim_Hext)
theta_OBC_kwant_opt = theta_vec_OBC[np.argmax(gap_OBC_kwant)]

S_opt_OBC_kwant = np.kron(np.eye(dim_Hext), julia_notes.S_theta_local(theta_OBC_kwant_opt))
S_tilde_OBC_kwant = rho @ S_opt_OBC_kwant + S_opt_OBC_kwant @ rho - S_opt_OBC_kwant
vals_OBC_kwant, vecs_OBC_kwant = eigh(S_tilde_OBC_kwant)
min_eigvec = vecs_OBC_kwant[:, np.argmin(np.abs(vals_OBC_kwant))]
DoS_min = local_DoS(min_eigvec, dim_Hext)


# --- PBC kwant
loger_main.info('Building PBC lattice with kwant: ...')
lattice_pbc_kwant = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice_pbc_kwant.seed = seed
lattice_pbc_kwant.boundary = 'Closed'
lattice_pbc_kwant.build_lattice(crystalline=crystalline)
lattice_pbc_kwant.generate_onsite_disorder(K_onsite=0.5 * W)
rashba_model_pbc_kwant = rashba_syst_Kwant(lattice_pbc_kwant, params_dict).finalized()
H_pbc_kwant = rashba_model_pbc_kwant.hamiltonian_submatrix()
_, _, rho_pbc_kwant = spectrum(H_pbc_kwant)
loger_main.info('Building PBC lattice with kwant: Done')

gap_PBC_kwant = np.zeros(theta_vec_OBC.shape)
for i, theta in enumerate(theta_vec_OBC):
    loger_main.trace(f'S_tilde gap at angle {i}/{Ntheta_OBC} for PBC kwant')
    _, gap_PBC_kwant[i], _ = rashba_bhz_S_tilde(rho_pbc_kwant, theta, dim_Hext)
theta_PBC_kwant_opt = theta_vec_OBC[np.argmax(gap_PBC_kwant)]

S_opt_PBC_kwant = np.kron(np.eye(dim_Hext), julia_notes.S_theta_local(theta_PBC_kwant_opt))
S_tilde_PBC_kwant = rho_pbc_kwant @ S_opt_PBC_kwant + S_opt_PBC_kwant @ rho_pbc_kwant - S_opt_PBC_kwant
vals_PBC_kwant = eigh(S_tilde_PBC_kwant)[0]


# --- PBC bloch
loger_main.info('Computing the PBC S_tilde gap (Bloch): ...')
theta_vec_PBC = np.linspace(-pi / 2, pi / 2, Ntheta_PBC)
gap_PBC_bloch = np.zeros(theta_vec_PBC.shape)
for i, theta in enumerate(theta_vec_PBC):
    S = julia_notes.S_theta_local(theta)
    gap_PBC_bloch[i], _ = julia_notes.clean_bloch_gap(lambR, S, NK=Nk)
theta_PBC_bloch_opt = theta_vec_PBC[np.argmax(gap_PBC_bloch)]
loger_main.info('Computing the PBC S_tilde gap (Bloch): Done')

S_opt_bloch_local = julia_notes.S_theta_local(theta_PBC_bloch_opt)
ks = np.linspace(-pi, pi, Nk, endpoint=False)
vals_PBC_bloch = []
for kx in ks:
    for ky in ks:
        P_k, _ = julia_notes.occupied_projector_bloch(julia_notes.bloch_hamiltonian(kx, ky, lambR))
        vals_PBC_bloch.append(np.linalg.eigvalsh(julia_notes.stilde(P_k, S_opt_bloch_local)))
vals_PBC_bloch = np.concatenate(vals_PBC_bloch)


# --- PBC lattice
loger_main.info('Computing the PBC S_tilde gap (lattice): ...')
rng_pbc = np.random.default_rng(julia_notes.SEED)
disorder_pbc = rng_pbc.uniform(-W / 2., W / 2., size=Nx * Ny)
H_pbc = julia_notes.build_pbc_hamiltonian(Nx, Nx, lambR, disorder_values=disorder_pbc)
P_pbc, energies_pbc, vectors_pbc = julia_notes.half_filling_projector(H_pbc)
blocks_pbc = julia_notes.projected_S_generators(vectors_pbc)
gap_PBC_lattice = np.array([julia_notes.auxiliary_gap_from_projected_blocks(theta, blocks_pbc)
                             for theta in theta_vec_PBC])
theta_PBC_lattice_opt = theta_vec_PBC[np.argmax(gap_PBC_lattice)]
loger_main.info('Computing the PBC S_tilde gap (lattice): Done')

S_opt_PBC_lattice = julia_notes.global_S(Nx, Ny, julia_notes.S_theta_local(theta_PBC_lattice_opt))
S_tilde_PBC_lattice = P_pbc @ S_opt_PBC_lattice + S_opt_PBC_lattice @ P_pbc - S_opt_PBC_lattice
vals_PBC_lattice = eigh(S_tilde_PBC_lattice)[0]


# --- OBC lattice
loger_main.info('Computing the OBC S_tilde gap (lattice): ...')
H_obc_lattice = build_obc_hamiltonian_rashba(Nx, Ny, lambR, disorder_pbc, A, M)
P_obc_lattice, energies_obc_lattice, vectors_obc_lattice = julia_notes.half_filling_projector(H_obc_lattice)
blocks_obc_lattice = julia_notes.projected_S_generators(vectors_obc_lattice)
gap_OBC_lattice = np.array([julia_notes.auxiliary_gap_from_projected_blocks(theta, blocks_obc_lattice)
                             for theta in theta_vec_PBC])
theta_OBC_lattice_opt = theta_vec_PBC[np.argmax(gap_OBC_lattice)]
loger_main.info('Computing the OBC S_tilde gap (lattice): Done')

S_opt_OBC_lattice = julia_notes.global_S(Nx, Ny, julia_notes.S_theta_local(theta_OBC_lattice_opt))
S_tilde_OBC_lattice = P_obc_lattice @ S_opt_OBC_lattice + S_opt_OBC_lattice @ P_obc_lattice - S_opt_OBC_lattice
vals_OBC_lattice = eigh(S_tilde_OBC_lattice)[0]


loger_main.info(f'theta_opt analytic     = {theta_opt * 180 / pi:.3f} deg')
loger_main.info(f'theta_opt PBC bloch    = {theta_PBC_bloch_opt * 180 / pi:.3f} deg')
loger_main.info(f'theta_opt PBC lattice  = {theta_PBC_lattice_opt * 180 / pi:.3f} deg')
loger_main.info(f'theta_opt PBC kwant    = {theta_PBC_kwant_opt * 180 / pi:.3f} deg')
loger_main.info(f'theta_opt OBC lattice  = {theta_OBC_lattice_opt * 180 / pi:.3f} deg')
loger_main.info(f'theta_opt OBC kwant    = {theta_OBC_kwant_opt * 180 / pi:.3f} deg')



#%%
# ============================================================
# Main: finite-size scaling of the OBC optimal angle
# ============================================================

Nx_scan = np.arange(Nmin, Nmax, 2)
theta_vec_scan = np.linspace(-pi / 2, pi / 2, Ntheta_scan)

theta_OBC_kwant_scan = np.zeros(Nx_scan.shape)
theta_OBC_lattice_scan = np.zeros(Nx_scan.shape)

for i, N in enumerate(Nx_scan):
    loger_main.info(f'Scaling scan at size {N}x{N} ({i}/{len(Nx_scan)})')

    # OBC kwant at this size
    lattice_N = AmorphousLattice_2d(Nx=N, Ny=N, w=width, r=r)
    lattice_N.seed = seed
    lattice_N.build_lattice(crystalline=crystalline)
    lattice_N.generate_onsite_disorder(K_onsite=0.5 * W)
    model_N = rashba_syst_Kwant(lattice_N, params_dict).finalized()
    _, _, rho_N = spectrum(model_N.hamiltonian_submatrix())
    gap_OBC_kwant_N = np.array([rashba_bhz_S_tilde(rho_N, theta, N * N)[1] for theta in theta_vec_scan])
    theta_OBC_kwant_scan[i] = theta_vec_scan[np.argmax(gap_OBC_kwant_N)]

    # OBC lattice at this size, same disorder convention as above
    rng_N = np.random.default_rng(julia_notes.SEED)
    disorder_N = rng_N.uniform(-W / 2., W / 2., size=N * N)
    H_obc_N = build_obc_hamiltonian_rashba(N, N, lambR, disorder_N, A, M)
    _, _, vectors_obc_N = julia_notes.half_filling_projector(H_obc_N)
    blocks_obc_N = julia_notes.projected_S_generators(vectors_obc_N)
    gap_OBC_lattice_N = np.array([julia_notes.auxiliary_gap_from_projected_blocks(theta, blocks_obc_N)
                                   for theta in theta_vec_scan])
    theta_OBC_lattice_scan[i] = theta_vec_scan[np.argmax(gap_OBC_lattice_N)]

    loger_main.info(f'Size {N}: theta_OBC_kwant={theta_OBC_kwant_scan[i] * 180 / pi:.2f} deg, '
                    f'theta_OBC_lattice={theta_OBC_lattice_scan[i] * 180 / pi:.2f} deg')

deviation_OBC_kwant = np.abs(theta_OBC_kwant_scan - theta_opt) * 180 / pi
deviation_OBC_lattice = np.abs(theta_OBC_lattice_scan - theta_opt) * 180 / pi



#%%
# ============================================================
# Checks
# ============================================================
# Encodes the findings from the PBC vs OBC investigation: the three periodic
# methods (independently implemented -- kwant, hand-rolled real-space PBC,
# hand-rolled Bloch) should agree closely with each other and with the
# analytic angle. The two open-boundary methods should agree with each other,
# but are genuinely shifted away from the periodic answer at this system size
# -- a real boundary effect, not a bug. If any of these stop holding, that is
# exactly the kind of silent regression this file exists to catch (e.g. the
# hopp() -> displacement2D_kwant boundary wiring breaking again).

TOL_AGREE = 0.05    # radians, ~3 degrees: independent methods for the same physics
TOL_SHIFT = 0.15    # radians, ~9 degrees: OBC vs PBC/analytic, at Nx=Ny=8


def test_periodic_methods_agree_with_analytic():
    assert abs(theta_PBC_bloch_opt - theta_opt) < TOL_AGREE
    assert abs(theta_PBC_lattice_opt - theta_opt) < TOL_AGREE
    assert abs(theta_PBC_kwant_opt - theta_opt) < TOL_AGREE


def test_open_boundary_methods_agree_with_each_other():
    assert abs(theta_OBC_kwant_opt - theta_OBC_lattice_opt) < TOL_AGREE


def test_open_boundary_is_shifted_from_periodic():
    assert abs(theta_OBC_kwant_opt - theta_opt) > TOL_SHIFT
    assert abs(theta_OBC_lattice_opt - theta_opt) > TOL_SHIFT


def test_hamiltonians_are_hermitian():
    assert np.allclose(H, H.conj().T)
    assert np.allclose(H_pbc_kwant, H_pbc_kwant.conj().T)



#%%
# ============================================================
# Figures
# ============================================================
# Only runs when this file is executed directly (python integration_test_PBC_vs_OBC.py),
# not when pytest imports it -- keeps the automated test fast and headless

if __name__ == '__main__':

    # Style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fontsize = 13
    main_color = 'teal'

    # Colors, markers, and labels shared by every figure below, one entry per method
    methods = [
        (theta_PBC_bloch_opt,   gap_PBC_bloch,   theta_vec_PBC, vals_PBC_bloch,   'crimson',       'o',  r'PBC bloch'),
        (theta_PBC_lattice_opt, gap_PBC_lattice, theta_vec_PBC, vals_PBC_lattice, 'darkorange',    '*',  r'PBC lattice'),
        (theta_PBC_kwant_opt,   gap_PBC_kwant,   theta_vec_OBC, vals_PBC_kwant,   'forestgreen',   'D',  r'PBC kwant'),
        (theta_OBC_lattice_opt, gap_OBC_lattice, theta_vec_PBC, vals_OBC_lattice, 'mediumpurple',  's',  r'OBC lattice'),
        (theta_OBC_kwant_opt,   gap_OBC_kwant,   theta_vec_OBC, vals_OBC_kwant,   main_color,      '^',  r'OBC kwant'),
    ]


    # --- Figure 1: gap vs theta for all five methods ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    for th_opt, gap, theta_vec, _, color, marker, name in methods:
        ax1.plot(theta_vec * 180 / pi, gap, color=color, linewidth=1.5, marker=marker,
                  markevery=0.05, markersize=6,
                  label=rf'{name}, $N_x=N_y={Nx}$, $W={W}$')

    ax1.axvline(theta_opt * 180 / pi, color='black', linestyle='--', linewidth=1,
                label=r'$\theta_{\rm opt}$ (analytic)')
    ax1.axvline(theta_hop * 180 / pi, color='black', linestyle=':', linewidth=1,
                label=r'$\theta_{\rm hop}$')

    trans = ax1.get_xaxis_transform()
    for i, (th_opt, _, _, _, color, marker, name) in enumerate(methods):
        ax1.axvline(th_opt * 180 / pi, color=color, linestyle='--', linewidth=1)
        ax1.text(th_opt * 180 / pi, 0.99 - 0.09 * i, rf'${th_opt * 180 / pi:.2f}^\circ$',
                  transform=trans, color=color, fontsize=fontsize - 3, ha='center', va='top',
                  bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    ax1.set_xlabel(r'$\theta$ (degrees)', fontsize=fontsize)
    ax1.set_ylabel(r'$\Delta_{\tilde{S}}$', fontsize=fontsize)
    ax1.set_title(rf'Auxiliary gap at $\lambda_R={lambR}$', fontsize=fontsize)
    ax1.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
    ax1.tick_params(which='major', length=6, labelsize=fontsize, color='black')
    ax1.legend(fontsize=fontsize - 3)
    ax1.grid(alpha=0.25)

    fig1.tight_layout()


    # --- Figure 2: finite-size scaling of the OBC angle offset ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    ax2.plot(Nx_scan, deviation_OBC_kwant, marker='o', color=main_color, label=r'OBC kwant')
    ax2.plot(Nx_scan, deviation_OBC_lattice, marker='s', color='mediumpurple', label=r'OBC lattice')

    ax2.set_xlabel('$N_x$', fontsize=fontsize)
    ax2.set_ylabel(r'$\vert \theta_{\rm opt} - \theta_{\rm analytic} \vert$ (degrees)', fontsize=fontsize)
    ax2.set_title(rf'Finite-size scaling of the OBC angle offset, $\lambda_R={lambR}$', fontsize=fontsize)
    ax2.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
    ax2.tick_params(which='major', length=6, labelsize=fontsize, color='black')
    ax2.set_xticks(Nx_scan)
    ax2.legend(fontsize=fontsize - 2)
    ax2.grid(alpha=0.25)

    fig2.tight_layout()


    # --- Figure 3: real-space density of the OBC kwant bottleneck eigenmode ---
    fig3, ax3 = plt.subplots(figsize=(7, 5))

    max_DoS, min_DoS = np.max(DoS_min), np.min(DoS_min)
    palette_DoS = sns.color_palette("mako_r", as_cmap=True)
    colors_DoS = palette_DoS(np.linspace(0.1, 1, 100))
    colors_DoS[0] = [1, 1, 1, 1]
    colormap_DoS = LinearSegmentedColormap.from_list("custom_colormap", colors_DoS)
    colorbar_DoS = cm.ScalarMappable(norm=Normalize(vmin=min_DoS, vmax=max_DoS), cmap=colormap_DoS)

    ax3.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS_min, cmap=colormap_DoS, edgecolor='black',
                s=30, linewidths=0.5, zorder=2)
    ax3.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
    ax3.tick_params(which='major', length=6, labelsize=fontsize, color='black')
    ax3.set_xlim(-1.5, Nx + 0.5)
    ax3.set_ylim(-1.5, Ny + 0.5)
    ax3.set(xticks=[0, Nx - 1], yticks=[0, Ny - 1])
    ax3.set_xlabel('$x$', fontsize=fontsize, labelpad=-20)
    ax3.set_ylabel('$y$', fontsize=fontsize, labelpad=-15)
    ax3.set_title(r'OBC kwant: density of the $\tilde{S}$ bottleneck eigenmode', fontsize=fontsize)

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig3.colorbar(colorbar_DoS, cax=cax, orientation='vertical', ticks=[0, max_DoS])
    cbar.set_ticklabels(['0.00', f'{max_DoS:.2f}'])
    cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    cbar.set_label(label='$\\vert \\psi (\\mathbf{r})\\vert ^2$', labelpad=-20, fontsize=20)

    fig3.tight_layout()


    # --- Figure 4: S_tilde spectrum at each method's own optimal angle
    fig4, ax4 = plt.subplots(figsize=(7, 5))

    # Methods have very different spectrum sizes (PBC bloch is sampled on a Nk x Nk
    # grid; the other four are a single D x D diagonalisation). Plotting every
    # eigenvalue makes the five curves impossible to tell apart where they overlap,
    # so each curve is thinned to the same number of points, evenly spaced along
    # its sorted spectrum (this keeps the overall shape, including the near-zero
    # kink, since the endpoints and the spacing are preserved)
    Nplot = 150
    for th_opt, _, _, vals, color, marker, name in methods:
        vals_sorted = np.sort(vals)
        idx_plot = np.linspace(0, len(vals_sorted) - 1, min(Nplot, len(vals_sorted))).astype(int)
        frac_index = idx_plot / (len(vals_sorted) - 1)
        ax4.plot(frac_index, vals_sorted[idx_plot], marker=marker, markerfacecolor='none',
                 markeredgecolor=color, linestyle='None', markersize=4, alpha=0.85,
                 markeredgewidth=1, label=rf'{name}, $\theta_{{\rm opt}}={th_opt * 180 / pi:.2f}^\circ$')

    ax4.axhline(0, color='black', linewidth=0.5)

    # Optimal angles repeated as text, in addition to the legend
    for i, (th_opt, _, _, _, color, marker, name) in enumerate(methods):
        ax4.text(0.02, 0.55 - 0.08 * i, rf'{name}: $\theta_{{\rm opt}}={th_opt * 180 / pi:.2f}^\circ$',
                 transform=ax4.transAxes, color=color, fontsize=fontsize - 3, ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    ax4.set_xlabel('fractional eigenstate index', fontsize=fontsize)
    ax4.set_ylabel(r'eig($\tilde{S}$)', fontsize=fontsize)
    ax4.set_title(rf'$\tilde{{S}}$ spectrum at the optimal $\theta$ of each method, $\lambda_R={lambR}$',
                  fontsize=fontsize)
    ax4.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
    ax4.tick_params(which='major', length=6, labelsize=fontsize, color='black')
    ax4.legend(fontsize=fontsize - 3, loc='upper left', bbox_to_anchor=(0.0, 0.98))
    ax4.grid(alpha=0.25)

    fig4.tight_layout()


    plt.show()
