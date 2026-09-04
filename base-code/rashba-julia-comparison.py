# Maths
import numpy as np
from numpy import pi
from numpy.linalg import eigh

# Kwant
import kwant

# Logging
import logging
import colorlog
from colorlog import ColoredFormatter

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, Polygon
import matplotlib.patheffects as path_effects
import seaborn as sns
import colorsys

# Loading Julia's notes script as a module (the hyphen in the filename means it
# cannot be reached with a normal "import" statement)
import importlib.util
from pathlib import Path

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.amorphous_rashba import rashba_syst_Kwant
from modules.OPDM import spectrum
from modules.S_optimisation import rashba_bhz_S_tilde, local_DoS

#%%
# ============================================================
# Logging setup
# ============================================================

loger_main = logging.getLogger('main')
loger_main.setLevel(logging.INFO)

stream_handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    '%(white)s%(asctime) -5s| %(blue)s%(name) -10s %(black)s| %(cyan)s %(funcName) '
    '-40s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'TRACE': 'white',
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
if not loger_main.handlers:
    loger_main.addHandler(stream_handler)

#%%
# ============================================================
# Load Julia's notes script (comparison reference, Step 5 / third figure)
# ============================================================

julia_path = Path(__file__).resolve().parent / 'analytics-Julia' / 'Julias-notes-plots.py'
julia_spec = importlib.util.spec_from_file_location('julia_notes', julia_path)
julia_notes = importlib.util.module_from_spec(julia_spec)
julia_spec.loader.exec_module(julia_notes)

#%%
# ============================================================
# Open-boundary counterpart of Julia's PBC Hamiltonian
# ============================================================
# Julia's own build_obc_hamiltonian is lambda_R=0 only (it belongs to a
# different check, the spin-frame disorder sweep), so it cannot be reused
# directly for the Rashba theta family. This reuses her actual hopping_x,
# hopping_y, TAU_Z and TAU_0 (which do carry lambda_R) and only replaces the
# periodic wraparound of her build_pbc_hamiltonian with the same boundary
# truncation her own build_obc_hamiltonian uses (bonds crossing the edge are
# dropped instead of wrapping around)

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

M                 = -2.
W                 = 0.
A                 = 1.
lambR             = 2
width             = 0.1
r                 = 1.3
Nx                = 15
Ny                = 15
crystalline       = True
params_dict = {'M': M, 'W': W, 'A': A, 'lambR': lambR}
dim_Hext = Nx * Ny
dim_Hint = 4
seed     = 12345
Ntheta_scan = 100
Nmin = 8
Nmax = 10

# Resolution of each theta scan
Ntheta_mine  = 81
Ntheta_julia = 361
NK_julia     = 40
theta_hop = julia_notes.theta_hop(lambR, A=A)
theta_opt = julia_notes.theta_optimal(lambR, A=A)

# Sigma matrices
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z


#%%
# ============================================================
# Main:
# ============================================================

# Finite crystalline lattice with Anderson disorder, kwant Hamiltonian, exact diagonalisation.
loger_main.info('Building OBC lattice with kwant: ...')
lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice.seed = seed
lattice.build_lattice(crystalline=crystalline)
lattice.generate_onsite_disorder(K_onsite=0.5 * W)
rashba_model = rashba_syst_Kwant(lattice, params_dict).finalized()
site_pos = np.array([site.pos for site in rashba_model.id_by_site])
H = rashba_model.hamiltonian_submatrix()
eps, eigenvectors, rho = spectrum(H)
loger_main.info('Building my finite lattice and Hamiltonian: Done')

theta_vec_mine = np.linspace(-pi / 2, pi / 2, Ntheta_mine)
gap_mine = np.zeros(theta_vec_mine.shape)
for i, theta in enumerate(theta_vec_mine):
    loger_main.info(f'My S_tilde gap at angle {i}/{Ntheta_mine}')
    _, gap_mine[i], _ = rashba_bhz_S_tilde(rho, theta, dim_Hext)
theta_mine_opt = theta_vec_mine[np.argmax(gap_mine)]

S0 = np.kron(tau_z, sigma_z)
S_opt = np.kron(np.eye(dim_Hext), np.cos(theta_mine_opt) * S0 + np.sin(theta_mine_opt) * np.kron(tau_y, sigma_0))
S_tilde = rho @ S_opt + S_opt @ rho - S_opt
vals, vecs = eigh(S_tilde)
min_eigvec = vecs[:, np.argmin(np.abs(vals))]
DoS_min = local_DoS(min_eigvec, dim_Hext)


# Clean Bloch Hamiltonian, gap minimised over the Brillouin zone
loger_main.info('Computing the translation invasriant S_tilde gap: ...')
theta_vec_julia = np.linspace(-pi / 2, pi / 2, Ntheta_julia)
gap_julia = np.zeros(theta_vec_julia.shape)
for i, theta in enumerate(theta_vec_julia):
    S = julia_notes.S_theta_local(theta)
    gap_julia[i], _ = julia_notes.clean_bloch_gap(lambR, S, NK=NK_julia)
theta_julia_opt = theta_vec_julia[np.argmax(gap_julia)]
loger_main.info('Computing the clean Bloch S_tilde gap (Julia): Done')



# Periodic-boundary lattice
loger_main.info('Computing the PBC S_tilde gap : ...')
rng_pbc = np.random.default_rng(julia_notes.SEED)
disorder_pbc = rng_pbc.uniform(-W / 2., W / 2., size=Nx * Ny)
H_pbc = julia_notes.build_pbc_hamiltonian(Nx, Nx, lambR, disorder_values=disorder_pbc)
P_pbc, energies_pbc, vectors_pbc = julia_notes.half_filling_projector(H_pbc)
blocks_pbc = julia_notes.projected_S_generators(vectors_pbc)
theta_vec_pbc = np.linspace(-pi / 2, pi / 2, Ntheta_julia)
gap_pbc = np.array([julia_notes.auxiliary_gap_from_projected_blocks(theta, blocks_pbc)
                     for theta in theta_vec_pbc])
theta_pbc_opt = theta_vec_pbc[np.argmax(gap_pbc)]
loger_main.info('Computing the disordered PBC S_tilde gap (Julia, Step 11): Done')



# OBC julia (not kwant)
loger_main.info('Computing the disordered Julia-style OBC S_tilde gap: ...')
H_obc_julia = build_obc_hamiltonian_rashba(Nx, Ny, lambR, disorder_pbc, A, M)
P_obc_julia, energies_obc_julia, vectors_obc_julia = julia_notes.half_filling_projector(H_obc_julia)
blocks_obc_julia = julia_notes.projected_S_generators(vectors_obc_julia)
gap_obc_julia = np.array([julia_notes.auxiliary_gap_from_projected_blocks(theta, blocks_obc_julia)
                           for theta in theta_vec_pbc])
theta_obc_julia_opt = theta_vec_pbc[np.argmax(gap_obc_julia)]
loger_main.info('Computing the disordered Julia-style OBC S_tilde gap: Done')

loger_main.info(f'theta_opt analytic               = {theta_opt * 180 / pi:.3f} deg')
loger_main.info(f'theta_opt Julia (clean Bloch)    = {theta_julia_opt * 180 / pi:.3f} deg')
loger_main.info(f'theta_opt Julia (disordered PBC) = {theta_pbc_opt * 180 / pi:.3f} deg')
loger_main.info(f'theta_opt Julia-style OBC (disord)= {theta_obc_julia_opt * 180 / pi:.3f} deg')
loger_main.info(f'theta_opt mine (finite open)     = {theta_mine_opt * 180 / pi:.3f} deg')



#%%
# ============================================================
# Main: finite-size scaling of the OBC optimal angle
# ============================================================
# If the OBC angle offset is really a finite-size effect, it should shrink as
# the lattice grows. Coarser theta resolution than above to keep this affordable

Nx_scan = np.arange(Nmin, Nmax, 2)
theta_vec_scan = np.linspace(-pi / 2, pi / 2, Ntheta_scan)

theta_mine_scan = np.zeros(Nx_scan.shape)
theta_obc_julia_scan = np.zeros(Nx_scan.shape)

for i, N in enumerate(Nx_scan):
    loger_main.info(f'Scaling scan at size {N}x{N} ({i}/{len(Nx_scan)})')

    # My kwant lattice at this size
    lattice_N = AmorphousLattice_2d(Nx=N, Ny=N, w=width, r=r)
    lattice_N.seed = seed
    lattice_N.build_lattice(crystalline=crystalline)
    lattice_N.generate_onsite_disorder(K_onsite=0.5 * W)
    model_N = rashba_syst_Kwant(lattice_N, params_dict).finalized()
    _, _, rho_N = spectrum(model_N.hamiltonian_submatrix())
    gap_mine_N = np.array([rashba_bhz_S_tilde(rho_N, theta, N * N)[1] for theta in theta_vec_scan])
    theta_mine_scan[i] = theta_vec_scan[np.argmax(gap_mine_N)]

    # Julia-style OBC at this size, same disorder convention as above
    rng_N = np.random.default_rng(julia_notes.SEED)
    disorder_N = rng_N.uniform(-W / 2., W / 2., size=N * N)
    H_obc_N = build_obc_hamiltonian_rashba(N, N, lambR, disorder_N, A, M)
    _, _, vectors_obc_N = julia_notes.half_filling_projector(H_obc_N)
    blocks_obc_N = julia_notes.projected_S_generators(vectors_obc_N)
    gap_obc_N = np.array([julia_notes.auxiliary_gap_from_projected_blocks(theta, blocks_obc_N)
                           for theta in theta_vec_scan])
    theta_obc_julia_scan[i] = theta_vec_scan[np.argmax(gap_obc_N)]

    loger_main.info(f'Size {N}: theta_mine={theta_mine_scan[i] * 180 / pi:.2f} deg, '
                    f'theta_obc_julia={theta_obc_julia_scan[i] * 180 / pi:.2f} deg')

deviation_mine = np.abs(theta_mine_scan - theta_opt) * 180 / pi
deviation_obc_julia = np.abs(theta_obc_julia_scan - theta_opt) * 180 / pi



#%%
# ============================================================
# Figures
# ============================================================

# Style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fontsize = 13
palette_color = 'crimson'
main_color = 'teal'

fig1, ax1 = plt.subplots(figsize=(7, 5))

# Different gaps
ax1.plot(theta_vec_julia * 180 / pi, gap_julia, color=palette_color, linewidth=1.5,
         label=r'clean translation-invariant')
ax1.plot(theta_vec_pbc * 180 / pi, gap_pbc, color='darkorange', linewidth=1.5,
         linestyle='dashed', label=rf'PBC, $N_x=N_y={Nx}$, $W={W}$')
ax1.plot(theta_vec_pbc * 180 / pi, gap_obc_julia, color='mediumpurple', linewidth=1.5,
         linestyle='solid', label=rf'Julia OBC, disordered, $N_x=N_y={Nx}$, $W={W}$')
ax1.plot(theta_vec_mine * 180 / pi, gap_mine, color=main_color, marker='o',
         linestyle='', markersize=4,
         label=rf'Amorphous_Lattice2d class (OBC), $N_x=N_y={Nx}$, $W={W}$')

ax1.axvline(theta_opt * 180 / pi, color='black', linestyle='--', linewidth=1,
            label=r'$\theta_{\rm opt}$ (analytic)')
ax1.axvline(theta_hop * 180 / pi, color='black', linestyle=':', linewidth=1,
            label=r'$\theta_{\rm hop}$')
ax1.axvline(theta_mine_opt * 180 / pi, color=main_color, linestyle='--', linewidth=1,
            label=r'$\theta_{\rm opt}$ (mine)')
ax1.axvline(theta_obc_julia_opt * 180 / pi, color='mediumpurple', linestyle='--', linewidth=1,
            label=r'$\theta_{\rm opt}$ (Julia-style OBC)')

# Numeric values of the two OBC optima
trans = ax1.get_xaxis_transform()
ax1.text(theta_mine_opt * 180 / pi, 0.99, rf'${theta_mine_opt * 180 / pi:.2f}^\circ$',
         transform=trans, color=main_color, fontsize=fontsize - 3, ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
ax1.text(theta_obc_julia_opt * 180 / pi, 0.90, rf'${theta_obc_julia_opt * 180 / pi:.2f}^\circ$',
         transform=trans, color='mediumpurple', fontsize=fontsize - 3, ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

ax1.set_xlabel(r'$\theta$ (degrees)', fontsize=fontsize)
ax1.set_ylabel(r'$\Delta_{\tilde{S}}$', fontsize=fontsize)
ax1.set_title(rf'Auxiliary gap at $\lambda_R={lambR}$', fontsize=fontsize)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax1.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax1.legend(fontsize=fontsize - 3)
ax1.grid(alpha=0.25)

fig1.tight_layout()


# Scaling with system size
fig2, ax2 = plt.subplots(figsize=(7, 5))

ax2.plot(Nx_scan, deviation_mine, marker='o', color=main_color,
         label=r'Amorphous\_Lattice2d class (OBC)')
ax2.plot(Nx_scan, deviation_obc_julia, marker='s', color='mediumpurple',
         label=r'Julia OBC')

ax2.set_xlabel('$N_x$', fontsize=fontsize)
ax2.set_ylabel(r'$\vert \theta_{\rm opt} - \theta_{\rm analytic} \vert$ (degrees)', fontsize=fontsize)
ax2.set_title(rf'Finite-size scaling of the OBC angle offset, $\lambda_R={lambR}$', fontsize=fontsize)
ax2.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax2.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax2.set_xticks(Nx_scan)
ax2.legend(fontsize=fontsize - 2)
ax2.grid(alpha=0.25)

fig2.tight_layout()





fig3, ax3 = plt.subplots(figsize=(7, 5))

# Colormap for the DoS
max_DoS, min_DoS = np.max(DoS_min), np.min(DoS_min)
palette_DoS = sns.color_palette("mako_r", as_cmap=True)
colors_DoS = palette_DoS(np.linspace(0.1, 1, 100))
colors_DoS[0] = [1, 1, 1, 1]
colormap_DoS = LinearSegmentedColormap.from_list("custom_colormap", colors_DoS)
norm_DoS = Normalize(vmin=min_DoS, vmax=max_DoS)
norm_theta = Normalize(vmin=0, vmax=1)
colorbar_DoS = cm.ScalarMappable(norm=Normalize(vmin=min_DoS, vmax=max_DoS), cmap=colormap_DoS)


# DoS of the zero modes
ax3.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS_min, cmap=colormap_DoS, edgecolor='black', s=30, linewidths=0.5, zorder=2)
ax3.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax3.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax3.set_xlim(-1.5, Nx + 0.5)
ax3.set_ylim(-1.5, Ny + 0.5)
ax3.set(xticks=[0, Nx-1], yticks=[0, Ny-1])
ax3.set_xlabel('$x$', fontsize=fontsize, labelpad=-20)
ax3.set_ylabel('$y$', fontsize=fontsize, labelpad=-15)
# 3oS for the zero modes: colorbar
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig3.colorbar(colorbar_DoS, cax=cax, orientation='vertical', ticks=[0, max_DoS])
cbar.set_ticklabels(['0.00', f'{max_DoS :.2f} '])
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.set_label(label='$\\vert \psi (\mathbf{r})\\vert ^2$', labelpad=-20, fontsize=20)










plt.show()




