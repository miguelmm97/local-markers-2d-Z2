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
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, Polygon
import matplotlib.patheffects as path_effects
import seaborn as sns
import colorsys
from matplotlib.patches import Rectangle

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.amorphous_rashba import rashba_syst_Kwant
from modules.OPDM import OPDM, spectrum
from modules.marker import local_marker, bulk_avg_marker
from modules.S_optimisation import rashba_bhz_S_tilde
from modules.colorbar_marker import get_continuous_cmap

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
# Code parameters
# ============================================================

M                 = -2.
W                 = 3.
A                 = 1.
lambR             = 2.
width             = 0.1
r                 = 1.3
Nx                = 10
Ny                = 10
Nsites            = Nx * Ny
cutoff_bulk_x     = 0.15
cutoff_bulk_y     = 0.15
params_dict = {'M': M, 'W': W, 'A': A, 'lambR': lambR}
crystalline = True
dim_Hext = Nx * Ny
dim_Hint = 4
dim_Hsp  = dim_Hint * dim_Hext
seed     = 12345

# Sigma matrices
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z


#%%
# ============================================================
# Main: Construction of the system, H, and OPDM
# ============================================================


# Lattice and BHZ Hamiltonian
loger_main.info('Generating site structure: ...')
lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice.seed = seed
lattice.build_lattice(crystalline=crystalline)
lattice.generate_onsite_disorder(K_onsite=0.5 * W)
loger_main.info('Generating site structure: Done')

loger_main.info('Defining Hamiltonian of the system in Kwant: ...')
rashba_model = rashba_syst_Kwant(lattice, params_dict).finalized()
site_pos = np.array([site.pos for site in rashba_model.id_by_site])
loger_main.info('Defining Hamiltonian of the system in Kwant: Done')

# OPDM (ED)
loger_main.info('Calculating OPDM through ED: ...')
H = rashba_model.hamiltonian_submatrix()
eps, eigenvectors, rho = spectrum(H)
loger_main.info('Calculating OPDM through ED: Done')

loger_main.info('Diagonalising OPDM: ...')
rho_values, rho_vecs = np.linalg.eigh(rho)
idx = rho_values.argsort()
rho_values, rho_vecs = rho_values[idx], rho_vecs[:, idx]


# Checks (only if debug level is set up)
if loger_main.isEnabledFor(logging.DEBUG):

    # TRS by commutation
    TRS = np.kron(np.eye(dim_Hext), 1j * np.kron(tau_0, sigma_y))
    if np.allclose(H, TRS @ H.conj() @ TRS.T.conj()):
        loger_main.info('Time-reversal symmetry H: True')
    else:
        loger_main.warning('Time-reversal symmetry H: False')

    # TRS by Kramer's degeneracy
    if np.allclose(eps[0::2], eps[1::2]):
        loger_main.info('Kramers degeneracy: True')
    else:
        loger_main.warning('Kramers degeneracy: False')

    if np.allclose(dim_Hsp % 2, 0.):
        pass
    else:
        loger_main.warning('The single-particle Hilbert space is odd dimensional')

    val_P = np.concatenate((np.zeros(int(dim_Hsp / 2)), np.ones(int(dim_Hsp / 2))))
    if np.allclose(rho_values, val_P):
        loger_main.info('Rho is a projector: True')
    else:
        loger_main.warning('Rho is a projector: False')





#%%
# ============================================================
# Calculation of the marker and one-parameter family minimisation
# ============================================================
theta_vec = np.linspace(-pi / 2, pi / 2, 50)
S0 = np.kron(tau_z, sigma_z)
avg_bulk_marker = np.zeros(theta_vec.shape)
gap_S_tilde = np.zeros(theta_vec.shape)

# Angles at which the marker is kept site by site, equally spaced along the family
Nsnapshots = 8
idx_snapshots = np.linspace(0, len(theta_vec) - 1, Nsnapshots, dtype=int)
marker_snapshots = []
theta_snapshots = []


for i, theta in enumerate(theta_vec):
    loger_main.info(f'Calculating local marker and S gap for angle {i}/{len(theta_vec)}')

    # Auxiliary "spin" operator for the bhz-rashba model
    S_tilde, gap_S_tilde[i], vals_S = rashba_bhz_S_tilde(rho, theta, dim_Hext)

    # Calculation of the local marker
    marker_per_site = local_marker(lattice.x, lattice.y, S_tilde, Nx=Nx, Ny=Ny)
    if loger_main.isEnabledFor(logging.DEBUG):
        if np.allclose(0, np.sum(marker_per_site)):
            loger_main.info('Local marker adds up to 0: True')
        else:
            loger_main.warning(f'Local marker adds up to 0: False || value: {np.sum(marker_per_site)}')
    avg_bulk_marker[i] = bulk_avg_marker(site_pos, marker_per_site, Nx, Ny, cutoff_x=cutoff_bulk_x, cutoff_y=cutoff_bulk_y)
    loger_main.info(f'Local marker for theta: {theta :.2f}: || Bulk average value: {avg_bulk_marker[i]:.2f}')

    # Snapshot of the marker site by site at this angle
    if i in idx_snapshots:
        marker_snapshots.append(marker_per_site)
        theta_snapshots.append(theta)



#%%
# ============================================================
# Figures
# ============================================================

# Style
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
markersize = 5
fontsize = 13
fontsize_inset = 13
palette = sns.color_palette("mako_r", as_cmap=True)
colors = palette(np.linspace(0.1, 1, 100))
colors[0] = [1, 1, 1, 1]
main_color = colors[30]

# Diverging colormap for the invariants
norm_invariants = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
hex_blues_mako_r = sns.color_palette("mako_r", 6).as_hex()[:4]
hex_white = ['#ffffff']
hex_reds = sns.color_palette("flare_r", 6).as_hex()[2:]
hex_list = hex_reds + hex_white + hex_blues_mako_r
float_list = [0.0, 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1.0]
hex_list_r = hex_list[::-1]
float_list_r = [1.0 - x for x in float_list[::-1]]
colormap_invariants = get_continuous_cmap(hex_list_r, float_list=float_list_r)
colorbar_invariants = cm.ScalarMappable(
    norm=norm_invariants,
    cmap=colormap_invariants
)



fig1 = plt.figure(figsize=(7, 7))
gs = GridSpec(6, 2, figure=fig1, wspace=0.4, hspace=0.85)
ax1 = fig1.add_subplot(gs[:2, 0])
ax2 = fig1.add_subplot(gs[0:2, 1])
ax3 = fig1.add_subplot(gs[2:4, :])
ax4 = fig1.add_subplot(gs[4:, :])


# Spectrum of the Hamiltonian
ax1.plot(np.arange(len(eps)), eps, marker='o', color=main_color, linestyle='None', markersize=0.5)
ax1.set_xlabel(' $N$', fontsize=fontsize, labelpad=4)
ax1.set_ylabel('$\\varepsilon$', fontsize=fontsize)
ax1.set_xlim(0, len(eps))
ax1.set_ylim(np.min(eps)-0.05, np.max(eps) + 0.05)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax1.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax1.set(xticks=[0, len(eps)])
ax1.text(
    0.05, 0.95,
    f'$L_x={Nx}$, $L_y={Ny}$\n$M={M}$\n$A={A}$\n$\\lambda_R={lambR}$',
    transform=ax1.transAxes, fontsize=8, va='top', ha='left'
)


# Spectrum of the OPDM
ax2.plot(np.arange(len(rho_values)), rho_values, marker='o', color=main_color, linestyle='None', markersize=0.5)
ax2.set_xlabel(' $ \\# \\vert n_m \\rangle$', fontsize=fontsize, labelpad=4)
ax2.set_ylabel('$n_m$', fontsize=fontsize)
ax2.set_xlim(0, len(rho_values))
ax2.set_ylim(np.min(rho_values)-0.05, np.max(rho_values) + 0.05)
ax2.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax2.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax2.set(xticks=[0, len(rho_values)])


# S gap vs angles
ax3.plot(theta_vec, gap_S_tilde, marker='o', color=main_color, linestyle='solid', markersize=2)
ax3.plot(theta_vec[idx_snapshots], gap_S_tilde[idx_snapshots], marker='o', color='crimson',
         linestyle='None', markersize=5, markeredgecolor='black', markeredgewidth=1, zorder=3)
ax3.set_xlabel(' $\\theta$', fontsize=fontsize, labelpad=6)
ax3.set_ylabel('$\\vert \\lambda_{\\tilde{S}} \\vert_{min}$', fontsize=fontsize)
ax3.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax3.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax3.set_xticks([-pi / 2, -pi / 4, 0, pi / 4, pi / 2])
ax3.set_xticklabels(['$-\\pi/2$', '$-\\pi/4$', '$0$', '$\\pi/4$', '$\\pi/2$'])

# Marker vs Angle
ax4.plot(theta_vec, avg_bulk_marker, marker='o', color=main_color, linestyle='solid', markersize=2)
ax4.plot(theta_vec[idx_snapshots], avg_bulk_marker[idx_snapshots], marker='o', color='crimson',
         linestyle='None', markersize=5, markeredgecolor='black', markeredgewidth=1, zorder=3)
ax4.set_xlabel(' $\\theta$', fontsize=fontsize, labelpad=6)
ax4.set_ylabel('$\\langle \\nu(\\theta) \\rangle_{bulk} $', fontsize=fontsize)
ax4.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax4.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax4.set_xticks([-pi / 2, -pi / 4, 0, pi / 4, pi / 2])
ax4.set_xticklabels(['$-\\pi/2$', '$-\\pi/4$', '$0$', '$\\pi/4$', '$\\pi/2$'])
fig1.subplots_adjust(top=0.96, bottom=0.08)





fig2 = plt.figure(figsize=(11, 5))
gs2 = GridSpec(2, 4, figure=fig2, wspace=0.3, hspace=0.4)
ax_snapshots = []
for i in range(Nsnapshots):
    ax_snapshots.append(fig2.add_subplot(gs2[i // 4, i % 4]))

# Bulk region entering the average, as defined in bulk_avg_marker
x_min_bulk = (Nx - 1) * cutoff_bulk_x
y_min_bulk = (Ny - 1) * cutoff_bulk_y
width_bulk = (Nx - 1) * (1 - 2 * cutoff_bulk_x)
height_bulk = (Ny - 1) * (1 - 2 * cutoff_bulk_y)


# Marker per site at each of the snapshot angles
for i, ax in enumerate(ax_snapshots):
    ax.scatter(site_pos[:, 0], site_pos[:, 1], c=marker_snapshots[i], cmap=colormap_invariants,
               norm=norm_invariants, edgecolor='black', s=20, zorder=2)

    # Bulk cutoff square
    bulk_square = Rectangle(
        (x_min_bulk, y_min_bulk),
        width_bulk,
        height_bulk,
        fill=False,
        linestyle='-',
        linewidth=1.5,
        edgecolor='dodgerblue',
        zorder=3
    )
    ax.add_patch(bulk_square)

    ax.set_xlim(-1, Nx + 1)
    ax.set_ylim(-1, Ny + 1)
    ax.set_xlabel('$x$', fontsize=fontsize, labelpad=-10)
    ax.set_ylabel('$y$', fontsize=fontsize, labelpad=-7)
    ax.set(xticks=[0, Nx], yticks=[0, Ny])
    ax.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
    ax.tick_params(which='major', length=6, labelsize=fontsize, color='black')
    ax.set_title(f'$\\theta = {theta_snapshots[i]:.2f}$, '
                 f'$\\langle \\nu \\rangle = {avg_bulk_marker[idx_snapshots[i]]:.2f}$', fontsize=fontsize)

# Colorbar shared by all the snapshots
cbar = fig2.colorbar(colorbar_invariants, ax=ax_snapshots, orientation='vertical', fraction=0.02, pad=0.02)
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.set_label(label='$\\nu(r)$', labelpad=-10, fontsize=fontsize)

plt.show()




