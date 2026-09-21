"""
Plot the disorder-averaged M transition from a postproduced summary file.

    python plotting/plot_transition_M.py
    python plotting/plot_transition_M.py -f data/M_transition_summary_21sep.h5

Reads the .h5 written by cluster-simulations/M_transition/postproduce_data.py and draws the
same figure as base-code/05_transition_M.py -- marker vs M in the main panel, the Gamma and
H gaps in two stacked insets -- with the statistics over disorder realisations layered on
top. Two versions of the figure are written on every run, so the two ways of showing the
spread can be compared side by side:

    mean-std     the mean marker with a band of +/- STD_BAND standard deviations
    median-iqr   the median marker with a band between the 0.25 and 0.75 quantiles

Both take the band from the reduced arrays in the file, so nothing is recomputed here. The
insets show only the averages, plus one horizontal dashed line per system size at the
smallest gap that any realisation reached anywhere in the scan -- the quantity that says
whether the gap ever came close to closing, which an averaged curve hides.
"""

# Utilities
import argparse
import sys
from datetime import date
from pathlib import Path

# The repo root is one level up from this file, so this runs from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Maths
import numpy as np

# Logging
import logging

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator

# Data
import h5py

# Modules
from modules.logging_config import setup_logging


parser = argparse.ArgumentParser(description='Plot the M transition from a summary file')
parser.add_argument('-f', '--file', type=str, default='data/M_transition_summary_21sep.h5',
                    help='Summary .h5 written by postproduce_data.py')
args = parser.parse_args()

# Half-widths of the shaded band in the mean-std figure, in standard deviations. The
# previous project (fig1-marker-new.py) used 0.5; 1.0 is the full standard deviation
STD_BAND = 1.0

# Placement of the legend title, in axes fractions: the gap left above the legend box, and
# a horizontal nudge off the centre of the labels if one is ever wanted
LEGEND_TITLE_PAD = 0.015
LEGEND_TITLE_DX = 0.

#%%
# ============================================================
# Logging setup
# ============================================================

setup_logging()
loger_main = logging.getLogger(__name__)


#%%
# ============================================================
# Loading data
# ============================================================

file_path = Path(args.file).expanduser()
if not file_path.is_absolute():
    file_path = Path(__file__).resolve().parents[1] / file_path
loger_main.info(f'Loading {file_path}')

with h5py.File(file_path, 'r') as f:

    # Statistics, all (n_sizes, NM) except the quantiles, which are (n_sizes, n_q, NM)
    M_vec            = f['Simulation/M'][:]
    size_vec         = f['Simulation/size_vec'][:]
    n_samples        = f['Simulation/n_samples'][:]
    quantiles        = f['Simulation/quantiles'][:]
    marker_avg       = f['Simulation/marker_avg'][:]
    marker_std       = f['Simulation/marker_std'][:]
    marker_median    = f['Simulation/marker_median'][:]
    marker_quantiles = f['Simulation/marker_quantiles'][:]
    gap_avg          = f['Simulation/Gamma_gap_avg'][:]
    gap_min          = f['Simulation/Gamma_gap_min'][:]
    gap_H_avg        = f['Simulation/H_gap_avg'][:]

    # Parameters of the batch, so the annotations do not have to be kept in step by hand
    A       = f['Parameters/A'][()]
    width   = f['Parameters/width'][()]
    lambR   = f['Parameters/lambR'][()]
    lambD   = f['Parameters/lambD'][()]
    W       = f['Parameters/W'][()]
    run_dir = f['Parameters'].attrs.get('Run_dir', 'unknown')

# The interquartile band. Picking the levels by value rather than by position keeps this
# working if postproduce_data.py is re-run with a different --quantiles
id_q25 = int(np.argmin(np.abs(quantiles - 0.25)))
id_q75 = int(np.argmin(np.abs(quantiles - 0.75)))
if not (np.isclose(quantiles[id_q25], 0.25) and np.isclose(quantiles[id_q75], 0.75)):
    loger_main.warning(f'No 0.25/0.75 quantiles in the file (has {quantiles}), '
                       f'using the closest levels {quantiles[id_q25]}, {quantiles[id_q75]}')

loger_main.info(f'Sizes {size_vec.tolist()}, samples {n_samples.tolist()}, {len(M_vec)} M points')
loger_main.info(f'Batch: {run_dir}')

# The averages are over different numbers of realisations once a batch is only partly
# finished, which the figure should say rather than hide
if len(set(n_samples)) == 1:
    samples_text = f'$N_s={n_samples[0]}$'
else:
    samples_text = f'$N_s={min(n_samples)}\\!-\\!{max(n_samples)}$'
    loger_main.warning(f'Sizes averaged over different sample counts: '
                       f'{dict(zip(size_vec.tolist(), n_samples.tolist()))}')


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


def make_figure(centre, band_low, band_high, band_note, tag):
    """One figure: 'centre' as the solid curve, the band shaded around it.

    The two versions differ only in what is handed in here, so the layout, the annotations
    and the insets cannot drift apart between them.
    """

    fig, ax = plt.subplots(figsize=(5, 4))

    for id_L, L in enumerate(size_vec):
        color = color_list[id_L]
        ax.plot(M_vec, centre[id_L], color=color, marker='o', markersize=4,
                linestyle='solid', label=f'${L}$')
        # Same idiom as fig1-marker-new.py in the nanowires project
        ax.fill_between(M_vec, band_low[id_L], band_high[id_L], color=color, alpha=0.3,
                        linewidth=0)

    ax.set_xlabel('$M$', fontsize=fontsize, labelpad=2)
    ax.set_ylabel('$\\langle \\nu \\rangle$', fontsize=fontsize)
    ax.set_xlim(M_vec[0], M_vec[-1])
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(2., 5., 0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='major', width=0.75, length=6, labelsize=fontsize, color='black')
    ax.tick_params(axis='both', which='minor', width=0.6, length=3, color='black',
                   bottom=True, top=False, left=True, right=False)

    # The marker falls from ~1 to 0 across the transition near M~3.2, so the curves run from the
    # top left to the bottom right and the free corners are the bottom left (parameters, legend)
    # and the top right (insets) -- the mirror image of the W scan in 04_transition_W.py

    # Fixed-parameter text. N_s is added to the block of 05_transition_M.py, since these
    # curves are disorder averages and the count is what sets the width of the band
    param_text = (f'$\\lambda_R={lambR:.1f}$\n'
                  f'$\\lambda_D={lambD:.1f}$\n'
                  f'$W={W}$\n'
                  f'$A={A:.1f}$\n'
                  f'$w={width}$\n'
                  f'{samples_text}\n')
    ax.text(0.2, 0.28, param_text, transform=ax.transAxes, ha='right', va='top', fontsize=fontsize_inset)

    # Legend: bigger font, centered in the free space above the parameter text. The title is
    # not set here -- a legend title is centred over the whole box, handles included, which
    # leaves it hanging to the left of the numbers. It is placed by hand below instead
    leg = ax.legend(fontsize=fontsize_inset + 3, loc='center', bbox_to_anchor=(0.15, 0.60))

    # Inset (top): gap of S_tilde (Gamma) vs M -- no x tick labels, sits right above the H inset.
    # Only the average is drawn; the dashed line marks the smallest gap any realisation of that
    # size reached anywhere in the scan, which is what says whether the gap ever nearly closed
    ax_inset_gamma = ax.inset_axes([0.66, 0.70, 0.32, 0.22])
    for id_L, L in enumerate(size_vec):
        ax_inset_gamma.plot(M_vec, gap_avg[id_L], color=color_list[id_L], marker=None,
                            markersize=3, linestyle='solid')
        ax_inset_gamma.axhline(gap_min[id_L].min(), color=color_list[id_L], linestyle='dashed',
                               linewidth=0.7, alpha=0.8)
    ax_inset_gamma.set_xlim(M_vec[0], M_vec[-1])
    ax_inset_gamma.set_ylim(0, 1)
    ax_inset_gamma.set_yticks([0, 0.5, 1])
    ax_inset_gamma.set_xticks([2, 3, 4])
    ax_inset_gamma.tick_params(axis='x', which='both', labelbottom=False)
    ax_inset_gamma.set_ylabel('$\\langle \\Delta_{\\Gamma} \\rangle$', fontsize=fontsize_inset)
    style_inset(ax_inset_gamma)

    # Inset (bottom): gap of the Hamiltonian spectrum vs M -- normal external ylabel. It grows
    # roughly linearly with M once past the transition and runs off the top of the inset,
    # which is deliberate: the range is shared with the Gamma inset so that the two can be
    # read against each other, and what matters here is where the gap closes, not how big
    # it gets afterwards
    ax_inset_H = ax.inset_axes([0.66, 0.42, 0.32, 0.22])
    for id_L, L in enumerate(size_vec):
        ax_inset_H.plot(M_vec, gap_H_avg[id_L], color=color_list[id_L], marker=None,
                        markersize=3, linestyle='solid')
    ax_inset_H.set_xlim(M_vec[0], M_vec[-1])
    ax_inset_H.set_ylim(0, 1)
    ax_inset_H.set_yticks([0, 0.5, 1])
    ax_inset_H.set_xticks([2, 3, 4])
    ax_inset_H.set_xlabel('$M$', fontsize=fontsize_inset)
    ax_inset_H.set_ylabel('$\\langle \\Delta_{H} \\rangle$', fontsize=fontsize_inset)
    style_inset(ax_inset_H)

    fig.tight_layout()

    # The legend title, centred on the labels rather than on the legend box. Taking the
    # position from the drawn legend keeps it aligned whatever the label widths or the font
    # size are, instead of being a pair of hand-tuned coordinates that silently go stale.
    # tight_layout runs first: it resizes the axes, and the legend keeps its size in points,
    # so its extent as a fraction of the axes is only final once the layout is
    fig.canvas.draw()
    to_axes = ax.transAxes.inverted()
    label_bbox = leg.get_texts()[0].get_window_extent()
    x_label = to_axes.transform((0.5 * (label_bbox.x0 + label_bbox.x1), 0.))[0]
    y_top = to_axes.transform((0., leg.get_window_extent().y1))[1]
    ax.text(x_label + LEGEND_TITLE_DX, y_top + LEGEND_TITLE_PAD, '$\\underline{L}$',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=fontsize_inset + 3)

    # Save to PDF, in their own subdirectory so that the figures made from postproduced
    # cluster data stay separate from the ones the single-machine scripts write
    fig_dir = Path(__file__).resolve().parent.parent / 'figures' / 'postproduced-figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    size_tag = '-'.join(str(L) for L in size_vec)
    fig_stem = (f'transition_M_L{size_tag}_lambR{lambR}_lambD{lambD}_W{W}_w{width}'
                f'_{tag}_{date.today().isoformat()}')
    fig_path = fig_dir / f'{fig_stem}.pdf'
    suffix = 1
    while fig_path.exists():
        fig_path = fig_dir / f'{fig_stem}_{suffix}.pdf'
        suffix += 1
    fig.savefig(fig_path, format='pdf', bbox_inches='tight')
    loger_main.info(f'Figure ({band_note}) saved to {fig_path}')
    return fig


# Version 1: mean with a standard-deviation band
make_figure(centre=marker_avg,
            band_low=marker_avg - STD_BAND * marker_std,
            band_high=marker_avg + STD_BAND * marker_std,
            band_note=f'mean, +/- {STD_BAND:g} std',
            tag='mean-std')

# Version 2: median with an interquartile band. Being built from quantiles, this one does
# not assume the spread is symmetric about the centre, which near the transition it is not
make_figure(centre=marker_median,
            band_low=marker_quantiles[:, id_q25, :],
            band_high=marker_quantiles[:, id_q75, :],
            band_note=f'median, {quantiles[id_q25]:g}-{quantiles[id_q75]:g} quantiles',
            tag='median-iqr')

plt.show()
