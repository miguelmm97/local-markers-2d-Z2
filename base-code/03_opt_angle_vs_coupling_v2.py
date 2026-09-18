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
from matplotlib.ticker import AutoMinorLocator
from matplotlib.collections import LineCollection

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.SO_rashba_dressel_Ham import SO_syst_Kwant
from modules.OPDM import spectrum
from modules.marker import marker_per_site_speedup
from modules.S_optimisation import get_S_tilde_and_gap, S_tilde_linear_basis, S_tilde_gap
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
Nx                = 10
Ny                = 10
dim_Hext = Nx * Ny
seed     = 12345

# Rashba coupling scan
lambR_vec = np.linspace(0., 1.5, 10)
lambD_vec = [0., 0.5, 0.75, 1.]

# Grid point used for the theta-domain panel (closest to lambR=1, already in lambR_vec)
id_R_fixed = int(np.argmin(np.abs(lambR_vec - 1.)))

# Resolution of the numerical theta optimisation
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
gap_theta_fixedR         = np.zeros((len(lambD_vec), Ntheta))

for id_D, lambD in enumerate(lambD_vec):
    for id_R, lambR in enumerate(lambR_vec):
        loger_main.info(f'lambD={lambD}, lambR={lambR:.3f} '
                         f'({id_D * len(lambR_vec) + id_R + 1}/{len(lambD_vec) * len(lambR_vec)})')

        # Model
        params_dict = {'M': M, 'W': W, 'A': A, 'lambR': lambR, 'lambD': lambD}
        model = SO_syst_Kwant(lattice, params_dict).finalized()
        H = model.hamiltonian_submatrix()
        eps, _, rho = spectrum(H)

        # Spectral gap H
        Nsp = H.shape[0] // 2
        gap_H_vec[id_D, id_R] = eps[Nsp] - eps[Nsp - 1]

        # Optimal numerical theta
        S_tilde_0, S_tilde_1 = S_tilde_linear_basis(rho, dim_Hext)
        gap_theta = np.zeros(theta_vec.shape)
        for i, theta in enumerate(theta_vec):
            gap_theta[i] = S_tilde_gap(S_tilde_0, S_tilde_1, theta)
        theta_opt_vec[id_D, id_R] = theta_vec[np.argmax(gap_theta)]
        if id_R == id_R_fixed:
            gap_theta_fixedR[id_D] = gap_theta

        #  Optimal analytical theta (clean lattice)
        param1 = (lambR + lambD) / A
        param2 = min(0., lambR - lambD) / A
        theta_opt_clean_vec[id_D, id_R] = 0.5 * (np.arctan(param1) + np.arctan(param2))

        # Marker and gap of S_tilde at the optimal angle
        S_tilde_opt, gap_Stilde_opt_vec[id_D, id_R], _ = get_S_tilde_and_gap(
            rho, theta_opt_vec[id_D, id_R], dim_Hext)
        marker_sites = marker_per_site_speedup(lattice.x, lattice.y, S_tilde_opt,
                                               Nx=Nx, Ny=Ny, boundary=lattice.boundary)
        marker_opt_vec[id_D, id_R] = marker_sites.mean()

        # Marker and gap of S_tilde at the clean analytical optimal angle
        S_tilde_clean, gap_Stilde_opt_clean_vec[id_D, id_R], _ = get_S_tilde_and_gap(
            rho, theta_opt_clean_vec[id_D, id_R], dim_Hext)
        marker_sites_clean = marker_per_site_speedup(lattice.x, lattice.y, S_tilde_clean,
                                                     Nx=Nx, Ny=Ny, boundary=lattice.boundary)
        marker_opt_clean_vec[id_D, id_R] = marker_sites_clean.mean()

#%%
# ============================================================
# Figures
# ============================================================

# Style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend', frameon=False, handlelength=1.2, handletextpad=0.4,
       borderaxespad=0.3, labelspacing=0.3, columnspacing=0.8)

# Continuous palette sampled from Ellsworth Kelly's "Train Landscape" (Art Institute of Chicago)
palette_train_landscape = LinearSegmentedColormap.from_list(
    'train_landscape', ['#FFEC0C', '#98D13C', '#16A950'])
color_list = palette_train_landscape(np.linspace(0, 1, len(lambD_vec)))
fontsize = 13
fontsize_inset = 10

# Curve styles
marker_numerics        = 'o'
markersize_numerics    = 6
markersize_inset       = 3
linestyle_numerics     = 'None'
alpha_numerics         = 1
marker_analytics       = 'None'
markersize_analytics   = None
alpha_analytics        = 0.5
linestyle_clean        = 'dashed'
vline_alpha            = 0.4
peak_markers           = ['o', 's', '^', 'D', 'v']

# Shared lambda_R ticks/labels -- used on the bottom panel's main axes and both its insets so
# all three read off the same x-axis
lambR_ticks = [0., 0.5, 1., 1.5]
lambR_ticklabels = [f'${v:g}$' for v in lambR_ticks]


def style_legend(ax, label_numerics, label_clean, loc='best', fontsize=fontsize - 2, ncol=1):
    handles = [
        Line2D([0], [0], color='black', marker=marker_numerics, markersize=markersize_numerics - 3,
               linestyle=linestyle_numerics, label=label_numerics),
        Line2D([0], [0], color='black', marker=marker_analytics, linestyle=linestyle_clean,
               alpha=alpha_analytics, label=label_clean),
    ]
    ax.legend(handles=handles, fontsize=fontsize, loc=loc, ncol=ncol)


def style_inset(ax):
    ax.tick_params(which='major', width=0.5, length=3, labelsize=fontsize_inset, color='black')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # usetex breaks the 'visible' rcParam for minor ticks, so the per-side flags are set explicitly
    ax.tick_params(axis='both', which='minor', width=0.4, length=1.5, color='black',
                    bottom=True, top=False, left=True, right=False)


def horizontal_ylabel(ax, text, fontsize, xpad=-0.3, align='center'):
    # horizontal label, instead of the default rotated one. 'top'/'bottom' line it up with the
    # extremal tick label instead of centering between them -- valid when that tick sits at the
    # axis limit, i.e. ylim was set to match the tick values (as axTop_inset does)
    ax.set_ylabel(text, fontsize=fontsize, rotation=0, va='center', ha='right')
    y_frac = {'top': 1.0, 'bottom': 0.0, 'center': 0.5}[align]
    ax.yaxis.set_label_coords(xpad, y_frac)


def ylabel_in_tick_column(ax, text, fontsize, y_frac=0.5, side='left'):
    # Horizontal y-axis label sharing the tick labels' own x (matplotlib aligns them to a common
    # column -- right-aligned on the left side, left-aligned on the right side), vertically
    # centered on the axis so the midpoint sits in the empty gap between the two tick labels
    ha = 'right' if side == 'left' else 'left'
    ax.set_ylabel(text, fontsize=fontsize, rotation=0, va='center', ha=ha)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    tick_bbox = ax.yaxis.get_majorticklabels()[-1].get_window_extent(renderer=renderer)
    bbox_axes = tick_bbox.transformed(ax.transAxes.inverted())
    x_frac = bbox_axes.x0 if side == 'right' else bbox_axes.x1
    ax.yaxis.set_label_coords(x_frac, y_frac)


def xlabel_in_tick_row(ax, text, fontsize, x_frac=0.5):
    # Vertical-axis counterpart: shares the tick labels' own y (their common row), horizontally
    # centered on the axis, in the empty gap between the two tick labels
    ax.set_xlabel(text, fontsize=fontsize, va='top', ha='center')
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    tick_bbox = ax.xaxis.get_majorticklabels()[-1].get_window_extent(renderer=renderer)
    y_frac = tick_bbox.transformed(ax.transAxes.inverted()).y1
    ax.xaxis.set_label_coords(x_frac, y_frac)


def center_title_over_legend_labels(ax, legend, title_text, fontsize, ypad=0.015):
    # The labels are left-justified (they all share the same x0, e.g. '0.75' is the one that
    # extends further right) -- so lining the title up with that shared left edge reads as
    # "above the column" the way a left-justified table header would, whereas centering over
    # the full x0-x1 span gets pulled right by the one wide label ('0.75') and looks
    # disconnected from the three narrower ones
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    label_bboxes = [t.get_window_extent(renderer=renderer) for t in legend.get_texts()]
    x0 = min(b.x0 for b in label_bboxes)
    leg_bbox = legend.get_window_extent(renderer=renderer)
    x_frac, y_frac = ax.transAxes.inverted().transform((x0, leg_bbox.y1))
    ax.text(x_frac, y_frac + ypad, title_text, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=fontsize)


def enable_minor_ticks(ax, n=2, length=3):
    ax.xaxis.set_minor_locator(AutoMinorLocator(n))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n))
    # usetex breaks the 'visible' rcParam for minor ticks, so the per-side flags are set explicitly
    ax.tick_params(axis='both', which='minor', width=0.6, length=length, color='black',
                    bottom=True, top=False, left=True, right=False)


fig, (axTop, axBottom) = plt.subplots(2, 1, figsize=(5, 8))

# Panel labels
axTop.text(-0.14, 1.03, '(a)', transform=axTop.transAxes, fontsize=fontsize + 1,
           fontweight='bold', ha='right', va='bottom')
axBottom.text(-0.14, 1.03, '(b)', transform=axBottom.transAxes, fontsize=fontsize + 1,
              fontweight='bold', ha='right', va='bottom')



# Top panel: gap of S_tilde vs theta at lambR_vec[id_R_fixed]
lambD_legend_handles = []
for id_D, lambD in enumerate(lambD_vec):
    color = color_list[id_D % len(color_list)]
    marker = peak_markers[id_D % len(peak_markers)]
    axTop.plot(theta_vec, gap_theta_fixedR[id_D], color=color, linestyle='solid')
    axTop.axvline(theta_opt_clean_vec[id_D, id_R_fixed], color=color, linestyle='dashed', alpha=vline_alpha)
    axTop.plot(theta_opt_vec[id_D, id_R_fixed], gap_theta_fixedR[id_D].max(),
              marker=marker, markersize=markersize_numerics,
              color=color, linestyle='None', zorder=3)
    lambD_legend_handles.append(Line2D([0], [0], color=color, marker=marker,
                                        markersize=markersize_numerics - 2,
                                        linestyle='solid', label=f'{lambD}'))

axTop.set_xlim(0, pi / 4)
axTop.set_xticks([0, pi / 8, pi / 4])
axTop.set_xticklabels(['$0$', '$\\pi/8$', '$\\pi/4$'])
axTop.set_xlabel('$\\theta$', fontsize=fontsize)
axTop.set_ylabel('$\\Delta_{\\Gamma}$', fontsize=fontsize)
axTop.set_ylim(bottom=0)
axTop.tick_params(which='major', width=0.75, length=6, labelsize=fontsize, color='black')
enable_minor_ticks(axTop)

leg_lambD = axTop.legend(handles=lambD_legend_handles, fontsize=fontsize - 3, loc='upper left',
                        bbox_to_anchor=(0, 0.97))
center_title_over_legend_labels(axTop, leg_lambD, '$\\lambda_D$', fontsize=fontsize - 3, ypad=-0.005)


# Arrow spanning the clean angles, coloured with the same palette as the lambD curves
thetas_clean_fixedR = theta_opt_clean_vec[:, id_R_fixed]
order = np.argsort(thetas_clean_fixedR)
theta_lo, theta_hi = thetas_clean_fixedR[order[0]], thetas_clean_fixedR[order[-1]]
arrow_y = 0.6 * axTop.get_ylim()[1]
line_end = theta_hi + 0.03
x_arrow = np.linspace(theta_lo - 0.03, line_end, 100)
points = np.array([x_arrow, np.full_like(x_arrow, arrow_y)]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
arrow_line = LineCollection(segments, cmap=palette_train_landscape, norm=plt.Normalize(0, 1),
                             alpha=vline_alpha + 0.3, linewidth=1.8)
arrow_line.set_array(np.linspace(0, 1, len(x_arrow) - 1))
axTop.add_collection(arrow_line)
axTop.annotate('', xy=(line_end + 0.02, arrow_y), xytext=(line_end, arrow_y),
             arrowprops=dict(arrowstyle='-|>', color=color_list[order[-1]],
                              alpha=vline_alpha + 0.3, lw=1.8))

# Label above the arrow
label_x = 0.5 * (thetas_clean_fixedR[order[0]] + thetas_clean_fixedR[order[1]])
axTop.text(label_x, arrow_y + 0.03 * axTop.get_ylim()[1], '$\\theta_k$',
         ha='center', va='bottom', fontsize=fontsize, color='black', alpha=vline_alpha + 0.3)
axTop.text(0.97, 0.95, f'$\\lambda_R={lambR_vec[id_R_fixed]:.1f}$', transform=axTop.transAxes,
         ha='right', va='top', fontsize=fontsize)
param_text = f'$L_x={Nx}$\n$L_y={Ny}$\n$M={M}$\n$A={A}$\n$W={W}$\n$w={width}$'
axTop.text(0.03, 0.03, param_text, transform=axTop.transAxes, ha='left', va='bottom', fontsize=fontsize_inset)







# Inset: theta_opt vs lambR
axTop_inset = axTop.inset_axes([0.42, 0.12, 0.5, 0.36])
for id_D, lambD in enumerate(lambD_vec):
    color = color_list[id_D % len(color_list)]
    axTop_inset.plot(lambR_vec, theta_opt_vec[id_D], color=color,
                    marker=peak_markers[id_D % len(peak_markers)],
                    markersize=markersize_inset, linestyle=linestyle_numerics, alpha=alpha_numerics)
    axTop_inset.plot(lambR_vec, theta_opt_clean_vec[id_D], color=color, linestyle=linestyle_clean,
                    alpha=alpha_analytics)
axTop_inset.set_ylim(0, pi / 4)
axTop_inset.set_yticks([0, pi / 4])
axTop_inset.set_yticklabels(['$0$', '$\\pi/4$'])
axTop_inset.set_xlim(0, lambR_vec[-1])
axTop_inset.set_xticks([0, lambR_vec[-1]])
axTop_inset.set_xticklabels(['$0$', f'${lambR_vec[-1]:g}$'])
xlabel_in_tick_row(axTop_inset, '$\\lambda_R$', fontsize_inset)
ylabel_in_tick_column(axTop_inset, '$\\theta$', fontsize_inset)
style_inset(axTop_inset)
style_legend(axTop_inset, '$\\theta_{\\rm opt}$', '$\\theta_k$',
             loc='upper left', fontsize=fontsize_inset - 1)






# Bottom panel: marker vs lambR
for id_D, lambD in enumerate(lambD_vec):
    color = color_list[id_D % len(color_list)]
    marker = peak_markers[id_D % len(peak_markers)]
    axBottom.plot(lambR_vec, marker_opt_vec[id_D], color=color,
              marker=marker,
              markersize=markersize_numerics, linestyle=linestyle_numerics,
              alpha=alpha_numerics)
    axBottom.plot(lambR_vec, marker_opt_clean_vec[id_D], color=color, marker=marker_analytics,
              markersize=markersize_analytics, linestyle=linestyle_clean, alpha=alpha_analytics)

axBottom.set_ylabel('$\\langle \\nu \\rangle$', fontsize=fontsize)
axBottom.set_ylim(-1, -0.5)
axBottom.set_xlim(lambR_vec[0], lambR_vec[-1])
axBottom.set_xticks(lambR_ticks)
axBottom.set_xticklabels(lambR_ticklabels)
axBottom.set_xlabel('$\\lambda_R$', fontsize=fontsize)
axBottom.tick_params(which='major', width=0.75, length=6, labelsize=fontsize, color='black')
enable_minor_ticks(axBottom)

style_legend(axBottom, '$\\langle \\nu(\\theta_{\\rm opt}) \\rangle$',
             '$\\langle \\nu(\\theta_k) \\rangle$',
             loc='upper left', fontsize=fontsize - 5, ncol=1)
axBottom.get_legend().set_bbox_to_anchor((0.05, 0.97), transform=axBottom.transAxes)



# Insets: gap of S_tilde and gap of H, both vs lambR -- stacked one above the other (rather
# than side by side) since they share the same x-axis, and widened/centered in the panel
axBottom_inset_S = axBottom.inset_axes([0.45, 0.47, 0.50, 0.22])
for id_D, lambD in enumerate(lambD_vec):
    color = color_list[id_D % len(color_list)]
    axBottom_inset_S.plot(lambR_vec, gap_Stilde_opt_vec[id_D], color=color,
                      marker=peak_markers[id_D % len(peak_markers)],
                      markersize=markersize_inset, linestyle=linestyle_numerics, alpha=alpha_numerics)
    axBottom_inset_S.plot(lambR_vec, gap_Stilde_opt_clean_vec[id_D], color=color, linestyle=linestyle_clean,
                      alpha=alpha_analytics)
axBottom_inset_S.set_xlim(lambR_vec[0], lambR_vec[-1])
axBottom_inset_S.set_xticks(lambR_ticks)
axBottom_inset_S.set_xticklabels(lambR_ticklabels)
axBottom_inset_S.set_ylim(bottom=0)
axBottom_inset_S.set_xlabel('$\\lambda_R$', fontsize=fontsize_inset)
horizontal_ylabel(axBottom_inset_S, '$\\Delta_{\\Gamma}$', fontsize_inset, xpad=-0.02)
style_inset(axBottom_inset_S)
# Bottom, above the x-axis: the curves never reach down that far, so the unboxed legend sits
# clear of them without needing a frame
style_legend(axBottom_inset_S, '$\\theta_{\\rm opt}$', '$\\theta_k$',
             loc='lower center', fontsize=fontsize_inset - 1, ncol=2)

axBottom_inset_H = axBottom.inset_axes([0.45, 0.75, 0.50, 0.22])
for id_D, lambD in enumerate(lambD_vec):
    color = color_list[id_D % len(color_list)]
    # No clean-angle line here -- the Hamiltonian gap has no analytical estimate, so only the
    # numerically optimal points are meaningful, no guiding line between them
    axBottom_inset_H.plot(lambR_vec, gap_H_vec[id_D], color=color,
                      marker=peak_markers[id_D % len(peak_markers)],
                      markersize=markersize_inset, linestyle=linestyle_numerics, alpha=alpha_numerics)
axBottom_inset_H.set_xlim(lambR_vec[0], lambR_vec[-1])
axBottom_inset_H.set_xticks(lambR_ticks)
axBottom_inset_H.set_ylim(bottom=0)
horizontal_ylabel(axBottom_inset_H, '$\\Delta_{H}$', fontsize_inset, xpad=-0.02)
style_inset(axBottom_inset_H)
# Shared x-axis: only the bottom inset (S_tilde) keeps its x tick labels
axBottom_inset_H.tick_params(axis='x', which='major', labelbottom=False)

fig.tight_layout()

# Save to PDF
fig_dir = Path(__file__).resolve().parent.parent / 'figures'
fig_dir.mkdir(exist_ok=True)
fig_stem = f'opt_angle_vs_coupling_v2_Nx{Nx}_Ny{Ny}_M{M}_A{A}_W{W}_w{width}_{date.today().isoformat()}'
fig_path = fig_dir / f'{fig_stem}.pdf'
suffix = 1
while fig_path.exists():
    fig_path = fig_dir / f'{fig_stem}_{suffix}.pdf'
    suffix += 1
fig.savefig(fig_path, format='pdf', bbox_inches='tight')
loger_main.info(f'Figure saved to {fig_path}')

plt.show()
