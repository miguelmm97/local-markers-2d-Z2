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
from matplotlib.lines import Line2D

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.SO_rashba_dressel_Ham import SO_syst_Kwant
from modules.OPDM import spectrum
from modules.marker import local_marker, bulk_avg_marker
from modules.S_optimisation import S_tilde_one_parameter
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
W                 = 0.25
r                 = 1.3
Nx                = 20
Ny                = 20
cutoff_bulk_x     = 0.15
cutoff_bulk_y     = 0.15
dim_Hext = Nx * Ny
seed     = 12345

# Rashba coupling scan
lambR_vec = np.linspace(0., 2., 10)
lambD_vec = [0., 0.5, 1.]

# Resolution of the numerical theta optimisation at each (lambR, lambD)
Ntheta = 50
theta_vec = np.linspace(0, pi / 4, Ntheta)

#%%
# ============================================================
# Main: lattice (PBC), fixed for every (lambR, lambD) point
# ============================================================

loger_main.info('Generating site structure: ...')
lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice.seed = seed
lattice.boundary = 'Closed'
lattice.build_lattice()
lattice.generate_onsite_disorder(K_onsite=0.5 * W)
loger_main.info('Generating site structure: Done')


#%%
# ============================================================
# Main: numerically optimal angle and marker for each (lambR, lambD)
# ============================================================

theta_opt_vec            = np.zeros((len(lambD_vec), len(lambR_vec)))
marker_opt_vec           = np.zeros((len(lambD_vec), len(lambR_vec)))
gap_Stilde_opt_vec       = np.zeros((len(lambD_vec), len(lambR_vec)))
gap_H_vec                = np.zeros((len(lambD_vec), len(lambR_vec)))
marker_opt_clean_vec     = np.zeros((len(lambD_vec), len(lambR_vec)))
gap_Stilde_opt_clean_vec = np.zeros((len(lambD_vec), len(lambR_vec)))
theta_opt_clean_vec      = np.zeros((len(lambD_vec), len(lambR_vec)))

for id_D, lambD in enumerate(lambD_vec):
    for id_R, lambR in enumerate(lambR_vec):
        loger_main.info(f'lambD={lambD}, lambR={lambR:.3f} '
                         f'({id_D * len(lambR_vec) + id_R + 1}/{len(lambD_vec) * len(lambR_vec)})')

        # Model
        params_dict = {'M': M, 'W': W, 'A': A, 'lambR': lambR, 'lambD': lambD}
        model = SO_syst_Kwant(lattice, params_dict).finalized()
        site_pos = np.array([site.pos for site in model.id_by_site])
        H = model.hamiltonian_submatrix()
        eps, _, rho = spectrum(H)

        # Spectral gap H
        Nsp = H.shape[0] // 2
        gap_H_vec[id_D, id_R] = eps[Nsp] - eps[Nsp - 1]

        # Optimal numerical theta
        gap_theta = np.zeros(theta_vec.shape)
        for i, theta in enumerate(theta_vec):
            _, gap_theta[i], _ = S_tilde_one_parameter(rho, theta, dim_Hext)
        theta_opt_vec[id_D, id_R] = theta_vec[np.argmax(gap_theta)]

        #  Optimal analytical theta (clean lattice)
        param1 = (lambR + lambD) / A
        param2 = min(0., lambR - lambD) / A
        theta_opt_clean_vec[id_D, id_R] = 0.5 * (np.arctan(param1) + np.arctan(param2))

        # Marker and gap of S_tilde at the optimal angle
        S_tilde_opt, gap_Stilde_opt_vec[id_D, id_R], _ = S_tilde_one_parameter(
            rho, theta_opt_vec[id_D, id_R], dim_Hext)
        marker_per_site = local_marker(lattice.x, lattice.y, S_tilde_opt, Nx=Nx, Ny=Ny)
        marker_opt_vec[id_D, id_R] = bulk_avg_marker(site_pos, marker_per_site, Nx, Ny,
                                                       cutoff_x=cutoff_bulk_x, cutoff_y=cutoff_bulk_y)

        # Marker and gap of S_tilde at the clean analytical optimal angle
        S_tilde_clean, gap_Stilde_opt_clean_vec[id_D, id_R], _ = S_tilde_one_parameter(
            rho, theta_opt_clean_vec[id_D, id_R], dim_Hext)
        marker_per_site_clean = local_marker(lattice.x, lattice.y, S_tilde_clean, Nx=Nx, Ny=Ny)
        marker_opt_clean_vec[id_D, id_R] = bulk_avg_marker(site_pos, marker_per_site_clean, Nx, Ny,
                                                             cutoff_x=cutoff_bulk_x, cutoff_y=cutoff_bulk_y)

#%%
# ============================================================
# Figures
# ============================================================

# Style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Continuous palette sampled from Ellsworth Kelly's "Train Landscape" (Art Institute of Chicago),
palette_train_landscape = LinearSegmentedColormap.from_list(
    'train_landscape', ['#98D13C', '#16A950', '#FFEC0C'])
color_list = palette_train_landscape(np.linspace(0, 1, len(lambD_vec)))
fontsize = 13

# Curve styles
marker_numerics        = 'o'
markersize_numerics    = 6
linestyle_numerics     = 'None'
alpha_numerics         = 1
marker_analytics       = 'None'
markersize_analytics   = None
alpha_analytics        = 0.5
linestyle_clean        = 'solid'
linestyle_guide        = 'dashed'


fig, axes = plt.subplots(2, 2, figsize=(9, 7))
ax1, ax2, ax3, ax4 = axes.flatten()
for id_D, lambD in enumerate(lambD_vec):
    color = color_list[id_D % len(color_list)]
    label = f'$\\lambda_D={lambD}$'

    ax1.plot(lambR_vec, theta_opt_vec[id_D],
             color=color,
             marker=marker_numerics,
             markersize=markersize_numerics,
             linestyle=linestyle_numerics,
             alpha=alpha_numerics,
             label=label)

    ax1.plot(lambR_vec, theta_opt_clean_vec[id_D],
             color=color,
             marker=marker_analytics,
             markersize=markersize_analytics,
             linestyle=linestyle_clean,
             alpha=alpha_analytics)

    ax2.plot(lambR_vec, marker_opt_vec[id_D],
             color=color,
             marker=marker_numerics,
             markersize=markersize_numerics,
             linestyle=linestyle_numerics,
             alpha=alpha_numerics,
             label=label)

    ax2.plot(lambR_vec, marker_opt_clean_vec[id_D],
             color=color,
             marker=marker_analytics,
             markersize=markersize_analytics,
             linestyle=linestyle_clean,
             alpha=alpha_analytics)

    ax3.plot(lambR_vec, gap_Stilde_opt_vec[id_D],
             color=color,
             marker=marker_numerics,
             markersize=markersize_numerics,
             linestyle=linestyle_numerics,
             alpha=alpha_numerics,
             label=label)

    ax3.plot(lambR_vec, gap_Stilde_opt_clean_vec[id_D],
             color=color,
             marker=marker_analytics,
             markersize=markersize_analytics,
             linestyle=linestyle_clean,
             alpha=alpha_analytics)

    ax4.plot(lambR_vec, gap_H_vec[id_D],
             color=color,
             marker=marker_numerics,
             markersize=markersize_numerics,
             linestyle=linestyle_guide,
             alpha=alpha_numerics,
             label=label)

ax1.set_ylim(0, pi/4)
ax1.set_ylabel('$\\theta_{opt}$', fontsize=fontsize)
ax1.set_yticks([0, pi / 8, pi / 4])
ax1.set_yticklabels(['$0$', '$\\pi/8$', '$\\pi/4$'])
lambD_legend = ax1.legend(fontsize=fontsize - 2, loc='upper left')
ax1.add_artist(lambD_legend)

ax2.set_ylabel('$\\langle \\nu \\rangle_{bulk}$', fontsize=fontsize)
ax2.set_ylim(-1, 0)
ax3.set_ylabel('$\\Delta_{\\tilde{S}}$', fontsize=fontsize)
ax3.set_ylim(bottom=0)
ax4.set_ylabel('$\\Delta_{H}$', fontsize=fontsize)
ax4.set_ylim(bottom=0)
for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel('$\\lambda_R$', fontsize=fontsize)
    ax.tick_params(which='major', width=0.75, length=6, labelsize=fontsize, color='black')

# Legends explaining the marker/line styles, one per subplot with its own quantity
def style_legend(ax, label_numerics, label_clean, loc='best'):
    handles = [
        Line2D([0], [0], color='black', marker=marker_numerics, markersize=markersize_numerics,
               linestyle=linestyle_numerics, label=label_numerics),
        Line2D([0], [0], color='black', marker=marker_analytics, linestyle=linestyle_clean,
               alpha=alpha_analytics, label=label_clean),
    ]
    ax.legend(handles=handles, fontsize=fontsize - 2, loc=loc)

style_legend(ax1, '$\\theta_{opt}$', '$\\tilde{\\theta}^D_{opt}$ (clean)', loc='lower right')
style_legend(ax2, '$\\langle \\nu(\\theta_{opt}) \\rangle$', '$\\langle \\nu(\\tilde{\\theta}^D_{opt}) \\rangle$')
style_legend(ax3, '$\\Delta_{\\tilde{S}}(\\theta_{opt})$', '$\\Delta_{\\tilde{S}}(\\tilde{\\theta}^D_{opt})$')


fig.suptitle(f'$L_x={Nx}$, $L_y={Ny}$, $M={M}$, $A={A}$, $W={W}$, $w={width}$', fontsize=fontsize)
fig.tight_layout(rect=(0, 0, 1, 0.95))



# Save to PDF
fig_dir = Path(__file__).resolve().parent.parent / 'figures'
fig_dir.mkdir(exist_ok=True)
fig_stem = f'opt_angle_vs_coupling_Nx{Nx}_Ny{Ny}_M{M}_A{A}_W{W}_w{width}_{date.today().isoformat()}'
fig_path = fig_dir / f'{fig_stem}.pdf'
suffix = 1
while fig_path.exists():
    fig_path = fig_dir / f'{fig_stem}_{suffix}.pdf'
    suffix += 1
fig.savefig(fig_path, format='pdf', bbox_inches='tight')
loger_main.info(f'Figure saved to {fig_path}')

plt.show()
