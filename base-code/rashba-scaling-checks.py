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
W                 = 0.
A                 = 1.
lambR             = np.linspace(0., 2., 10)
width             = 0.1
r                 = 1.3
Nx                = np.arange(10, 20)
Ny                = np.arange(8, 20)
cutoff_bulk_x     = 0.15
cutoff_bulk_y     = 0.15
dim_Hint = 4


# Sigma matrices
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z



Nsites = Nx * Ny
params_dict = {'M': M, 'W': W, 'A': A, 'lambR': lambR}
crystalline = False
dim_Hext = Nx * Ny

dim_Hsp  = dim_Hint * dim_Hext

#%%
# ============================================================
# Main: Scaling of theta_max and marker vs system size and rashba coupling
# ============================================================

theta_max = np.zeros(len(lambR), len(Nx))
bulk_marker_max = np.zeros(len(lambR), len(Nx))
theta_vec = np.linspace(0, pi, 50)
S0 = np.kron(tau_z, sigma_z)

for i, lamb in enumerate(lambR):
    for j, N in enumerate(Nx):

        # Lattice
        lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
        lattice.build_lattice(crystalline=crystalline)
        dim_Hext = Nx * Ny

        # Hamiltonian
        rashba_model = rashba_syst_Kwant(lattice, params_dict).finalized()
        site_pos = np.array([site.pos for site in rashba_model.id_by_site])

        # OPDM (ED)
        H = rashba_model.hamiltonian_submatrix()
        eps, eigenvectors, rho = spectrum(H)

        # optimal S
        gap_S_tilde = np.zeros(theta_vec.shape)
        for k, theta in enumerate(theta_vec):
            S_tilde, gap_S_tilde[j], _ = rashba_bhz_S_tilde(rho, theta, dim_Hext)
        theta_max[i, j] = theta_vec[np.argmax(gap_S_tilde)]

        # Local marker
        marker_per_site = local_marker(lattice.x, lattice.y, S_tilde[np.argmax(gap_S_tilde)], Nx=Nx, Ny=Ny)
        bulk_marker_max[i, j] = bulk_avg_marker(site_pos, marker_per_site, Nx, Ny, cutoff_x=cutoff_bulk_x,
                                                                                cutoff_y=cutoff_bulk_y)



