
# Maths
from numpy import pi
import numpy as np

# Kwant
import kwant

# Logging
import logging

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
from modules.Hamiltonian_Kwant import spectrum, BHZ_Hamiltonian_Kwant, OPDM, local_marker, bulk_avg_marker
from modules.colorbar_marker import get_continuous_cmap
from modules.logging_config import setup_logging

#%% Logging setup
setup_logging(logging.TRACE)
loger_main = logging.getLogger(__name__)

#%% Variables
m                 = -3.5
t                 = 1
lamb              = 1
width             = 0.1
r                 = 1.3
Nx                = 20
Ny                = 20
Nsites            = Nx * Ny
cutoff_bulk_x     = 0.25
cutoff_bulk_y     = 0.25
params_dict = {'m': m, 't': t, 'lamb': lamb}
crystalline = False
dim_Hext = Nx * Ny
dim_Hint = 4
dim_Hsp  = dim_Hint * dim_Hext
spin_mixing = False if np.allclose(lamb, 0) else True


# Sigma matrices
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z


#%% Main: System

# Lattice and BHZ Hamiltonian
loger_main.info('Generating site structure: ...')
lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice.build_lattice(crystalline=crystalline)
loger_main.info('Generating site structure: Done')
loger_main.info('Defining Hamiltonian of the BHZ model in Kwant: ...')
bhz_model = BHZ_Hamiltonian_Kwant(lattice, params_dict).finalized()
site_pos = np.array([site.pos for site in bhz_model.id_by_site])
loger_main.info('Defining Hamiltonian of the BHZ model in Kwant: Done')

# OPDM (ED)
loger_main.info('Calculating OPDM through ED: ...')
H = bhz_model.hamiltonian_submatrix()
eps, eigenvectors, rho = spectrum(H)
loger_main.info('Calculating OPDM through ED: Done')


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

    # Rho is a projector (no interactions, gs is gaussian)
    rho_values, rho_vecs = np.linalg.eigh(rho)
    idx = rho_values.argsort()
    rho_values, rho_vecs = rho_values[idx], rho_vecs[:, idx]
    val_P = np.concatenate((np.zeros(int(dim_Hsp / 2)), np.ones(int(dim_Hsp / 2))))
    if np.allclose(dim_Hsp % 2, 0.):
        pass
    else:
        loger_main.warning('The single-particle Hilbert space is odd dimensional')
    if np.allclose(rho_values, val_P):
        loger_main.info('Rho is a projector: True')
    else:
        loger_main.warning('Rho is a projector: False')


# Marker per site
S = np.kron(np.eye(Nsites), np.kron(tau_0, sigma_z))
loger_main.info('Calculating local marker: ...')
local_marker = local_marker(lattice.x, lattice.y, rho, S, spin_mixing=spin_mixing, shift_per_site=False, Nx=Nx, Ny=Ny)
avg_bulk_marker = bulk_avg_marker(site_pos, local_marker, Nx, Ny, cutoff_x=cutoff_bulk_x, cutoff_y=cutoff_bulk_y)
loger_main.info(f'Calculating local marker: Done || Bulk average value: {avg_bulk_marker:.2f}')
if loger_main.isEnabledFor(logging.DEBUG):
    if np.allclose(0, np.sum(local_marker)):
        loger_main.info('Local marker adds up to 0: True')
    else:
        loger_main.warning(f'Local marker adds up to 0: False || value: {np.sum(local_marker)}')





#%% Figures

# Style
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
markersize = 5
fontsize = 20
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



fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 2, figure=fig1, wspace=0.45, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[1, 0])
ax3 = fig1.add_subplot(gs[0, 1])
ax4 = fig1.add_subplot(gs[1, 1])



# Spectrum of the Hamiltonian
ax1.plot(np.arange(len(eps)), eps, marker='o', color=main_color, linestyle='None', markersize=0.5)
ax1.set_xlabel(' $N$', fontsize=fontsize, labelpad=-15)
ax1.set_ylabel('$\\varepsilon$', fontsize=fontsize)
ax1.set_xlim(0, len(eps))
ax1.set_ylim(np.min(eps)-0.05, np.max(eps) + 0.05)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax1.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax1.set(xticks=[0, len(eps)])
ax1.text(20, 8, f'$L_x={Nx}$, $L_y={Ny}$', fontsize=10)
ax1.text(20, 6, f'$t={t}$', fontsize=10)
ax1.text(20, 4, f'$m={m}$', fontsize=10)
ax1.text(20, 2, f'$\\lambda={lamb}$', fontsize=10)


# Spectrum of the  OPDM
ax2.plot(np.arange(len(rho_values)), rho_values, marker='o', color=main_color, linestyle='None', markersize=0.5)
ax2.set_xlabel(' $ \\# \\vert n_m \\rangle$', fontsize=fontsize, labelpad=-15)
ax2.set_ylabel('$n_m$', fontsize=fontsize)
ax2.set_xlim(0, len(rho_values))
ax2.set_ylim(np.min(rho_values)-0.05, np.max(rho_values) + 0.05)
ax2.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax2.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax2.set(xticks=[0, len(rho_values)])


# Marker per site
ax3.scatter(site_pos[:, 0], site_pos[:, 1], c=local_marker, cmap=colormap_invariants,
            norm=norm_invariants, edgecolor='black', s=20, zorder=2)
# Bulk cutoff square
x_center = Nx / 2
y_center = Ny / 2
x_half_width = cutoff_bulk_x * Nx
y_half_width = cutoff_bulk_y * Ny
x_min_bulk = x_center - x_half_width
y_min_bulk = y_center - y_half_width
bulk_square = Rectangle(
    (x_min_bulk, y_min_bulk),
    2 * x_half_width,
    2 * y_half_width,
    fill=False,
    linestyle='-',
    linewidth=1.5,
    edgecolor='dodgerblue',
    zorder=3
)
ax3.add_patch(bulk_square)

ax3.set_xlim(-1, Nx+1)
ax3.set_ylim(-1, Ny+1)
ax3.set_xlabel('$x$', fontsize=fontsize, labelpad=-20)
ax3.set_ylabel('$y$', fontsize=fontsize, labelpad=-7)
ax3.set(xticks=[0, Nx], yticks=[0, Ny])
ax3.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax3.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax3.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax3.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax3.set_title(f'Bulk average value: { avg_bulk_marker:.2f}')

# Colorbar
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig1.colorbar(colorbar_invariants, cax=cax, orientation='vertical')
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.set_label(label='$\\nu(r)$', labelpad=-10, fontsize=20)

# fig1.savefig('ex_test2.pdf', format='pdf')

plt.show()

