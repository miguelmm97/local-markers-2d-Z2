# Maths
import numpy as np
from numpy import pi

# Saving figures
from pathlib import Path
from datetime import date

# Logging
import logging

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.SO_rashba_dressel_Ham import SO_syst_Kwant
from modules.OPDM import spectrum
from modules.marker import marker_per_site_speedup
from modules.S_optimisation import get_S_tilde_and_gap
from modules.logging_config import setup_logging

#%%
# ============================================================
# Logging setup
# ============================================================

setup_logging()
loger_main = logging.getLogger(__name__)

#%%
# ============================================================
# Code parameters
# ============================================================

M                 = -2.
A                 = 1.
width             = 0.1
r                 = 1.3
seed     = 12345

# System sizes scanned (square lattices, Nx = Ny)
size_vec = [10, 20, 30]

# Fixed coupling point at which the disorder-driven transition is scanned
lambR = 1.
lambD = 0.5

# Onsite disorder scan
W_vec = np.linspace(0., 30., 30)

# Analytical (clean-lattice) angle at this (lambR, lambD) -- fixed for the whole scan, since it
# doesn't depend on disorder, so no numerical theta optimisation is needed at any W
param1 = (lambR + lambD) / A
param2 = min(0., lambR - lambD) / A
theta_k = 0.5 * (np.arctan(param1) + np.arctan(param2))

#%%
# ============================================================
# Main: marker and gap of S_tilde at theta_k, for each disorder strength W, repeated for
# each system size
# ============================================================

marker_vec = np.zeros((len(size_vec), len(W_vec)))
gap_vec    = np.zeros((len(size_vec), len(W_vec)))
gap_H_vec  = np.zeros((len(size_vec), len(W_vec)))

for id_L, L in enumerate(size_vec):
    Nx = Ny = L
    dim_Hext = Nx * Ny

    loger_main.info(f'Generating site structure for L={L}: ...')
    lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
    lattice.seed = seed
    lattice.boundary = 'Closed'
    lattice.build_lattice()
    loger_main.info(f'Generating site structure for L={L}: Done')

    for i, W in enumerate(W_vec):
        loger_main.info(f'L={L} ({id_L + 1}/{len(size_vec)}), W={W:.3f} ({i + 1}/{len(W_vec)})')

        # Same seed every call -> same underlying random pattern, just rescaled by K_onsite
        lattice.generate_onsite_disorder(K_onsite=0.5 * W)

        # Model
        params_dict = {'M': M, 'W': W, 'A': A, 'lambR': lambR, 'lambD': lambD}
        model = SO_syst_Kwant(lattice, params_dict).finalized()
        H = model.hamiltonian_submatrix()
        energy, _, rho = spectrum(H)

        # Gap of the Hamiltonian spectrum around the Fermi level (half filling)
        Nsp = int(len(H) / 2)
        gap_H_vec[id_L, i] = energy[Nsp] - energy[Nsp - 1]

        # Marker and gap of S_tilde at the fixed analytical angle
        S_tilde, gap_vec[id_L, i], _ = get_S_tilde_and_gap(rho, theta_k, dim_Hext)
        marker_sites = marker_per_site_speedup(lattice.x, lattice.y, S_tilde,
                                               Nx=Nx, Ny=Ny, boundary=lattice.boundary)
        marker_vec[id_L, i] = marker_sites.mean()

#%%
# ============================================================
# Figures
# ============================================================

# Style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend', frameon=False, handlelength=1.2, handletextpad=0.4,
       borderaxespad=0.3, labelspacing=0.3, columnspacing=0.8)

# Continuous palette sampled from Ellsworth Kelly's "Train Landscape" (Art Institute of Chicago),
# same as in 03_opt_angle_vs_coupling_v2.py -- one colour per system size
palette_train_landscape = LinearSegmentedColormap.from_list(
    'train_landscape', ['#FFEC0C', '#98D13C', '#16A950'])
color_list = palette_train_landscape(np.linspace(0, 1, len(size_vec)))
fontsize = 13
fontsize_inset = 10


def style_inset(ax):
    ax.tick_params(which='major', width=0.5, length=3, labelsize=fontsize_inset, color='black')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # usetex breaks the 'visible' rcParam for minor ticks, so the per-side flags are set explicitly
    ax.tick_params(axis='both', which='minor', width=0.4, length=1.5, color='black',
                    bottom=True, top=False, left=True, right=False)

def inset_ylabel(ax, text, fontsize):
    # Placed as a corner annotation inside the inset's own axes, instead of an external
    # axis label -- an external label drifts by a fixed axes-fraction offset, which for a
    # small inset spanning the same data range as the main axes lands it far outside the
    # inset (inside the main plot, on top of the curves) rather than hugging the inset
    ax.text(0.08, 0.92, text, transform=ax.transAxes, ha='left', va='top', fontsize=fontsize)

fig, ax = plt.subplots(figsize=(5, 4))

for id_L, L in enumerate(size_vec):
    color = color_list[id_L]
    ax.plot(W_vec, marker_vec[id_L], color=color, marker='o', markersize=4,
            linestyle='solid', label=f'${L}$')

ax.set_xlabel('$W$', fontsize=fontsize, labelpad=2)
ax.set_ylabel('$\\langle \\nu \\rangle$', fontsize=fontsize)
ax.set_xlim(W_vec[0], W_vec[-1])
ax.set_ylim(-1, 0)
ax.set_xticks([0, 10, 20, 30])
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(which='major', width=0.75, length=6, labelsize=fontsize, color='black')
ax.tick_params(axis='both', which='minor', width=0.6, length=3, color='black',
               bottom=True, top=False, left=True, right=False)

# Fixed-parameter text
param_text = (f'$\\lambda_R={lambR:.1f}$\n'
              f'$\\lambda_D={lambD:.1f}$\n'
              f'$M={M:.1f}$\n'
              f'$A={A:.1f}$\n'
              f'$w={width}$\n')
ax.text(0.2, 0.4, param_text, transform=ax.transAxes, ha='right', va='top', fontsize=fontsize_inset)

# Legend: bigger font, centered in the free space above the parameter text
leg = ax.legend(fontsize=fontsize_inset + 3, loc='center', bbox_to_anchor=(0.15, 0.70),
                 title='$L$', title_fontsize=fontsize_inset + 3)

# Inset (top): gap of S_tilde (Gamma) vs W -- no x tick labels, sits right above the H inset
ax_inset_gamma = ax.inset_axes([0.65, 0.46, 0.3, 0.2])
for id_L, L in enumerate(size_vec):
    ax_inset_gamma.plot(W_vec, gap_vec[id_L], color=color_list[id_L], marker=None,
                        markersize=3, linestyle='solid')
ax_inset_gamma.set_xlim(W_vec[0], W_vec[-1])
ax_inset_gamma.set_ylim(0, 1)
ax_inset_gamma.set_yticks([0, 0.5, 1])
ax_inset_gamma.set_xticks([0, 10, 20, 30])
ax_inset_gamma.tick_params(axis='x', which='both', labelbottom=False)
ax_inset_gamma.set_ylabel('$\\Delta_{\Gamma}$', fontsize=fontsize_inset)
style_inset(ax_inset_gamma)

# Inset (bottom): gap of the Hamiltonian spectrum vs W -- normal external ylabel, y ticks at 0/1/2
ax_inset_H = ax.inset_axes([0.65, 0.20, 0.3, 0.2])
for id_L, L in enumerate(size_vec):
    ax_inset_H.plot(W_vec, gap_H_vec[id_L], color=color_list[id_L], marker=None,
                    markersize=3, linestyle='solid')
ax_inset_H.set_xlim(W_vec[0], W_vec[-1])
ax_inset_H.set_ylim(0, 2)
ax_inset_H.set_yticks([0, 1, 2])
ax_inset_H.set_xticks([0, 10, 20, 30])
ax_inset_H.set_xlabel('$W$', fontsize=fontsize_inset)
ax_inset_H.set_ylabel('$\\Delta_{H}$', fontsize=fontsize_inset)
style_inset(ax_inset_H)

fig.tight_layout()

# Save to PDF
fig_dir = Path(__file__).resolve().parent.parent / 'figures'
fig_dir.mkdir(exist_ok=True)
size_tag = '-'.join(str(L) for L in size_vec)
fig_stem = f'transition_W_L{size_tag}_lambR{lambR}_lambD{lambD}_w{width}_{date.today().isoformat()}'
fig_path = fig_dir / f'{fig_stem}.pdf'
suffix = 1
while fig_path.exists():
    fig_path = fig_dir / f'{fig_stem}_{suffix}.pdf'
    suffix += 1
fig.savefig(fig_path, format='pdf', bbox_inches='tight')
loger_main.info(f'Figure saved to {fig_path}')

plt.show()
