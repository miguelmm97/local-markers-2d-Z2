#%% modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Kwant
import kwant
import tinyarray as ta

# Managing classes
from dataclasses import dataclass, field

# Tracking time
import time

# Managing logging
import logging

loger_amorphous = logging.getLogger(__name__)

#%% Module

# Functions for creating the lattice
def gaussian_point_set_2D(x, y, width, seed):
    rng = np.random.default_rng(seed=seed)
    x = rng.normal(x, width, len(x))
    y = rng.normal(y, width, len(y))
    return x, y

@dataclass
class AmorphousLattice_2d:

    # Class fields set upon instantiation
    Nx:  int                                        # Number of lattice sites along x direction
    Ny:  int                                        # Number of lattice sites along y direction
    w:   float                                      # Width of the Gaussian distribution
    r:   float                                      # Cutoff distance to consider neighbours

    # Class fields that can be set externally
    x: np.ndarray   = None                          # x position of the sites
    y: np.ndarray   = None                          # y position of the sites
    coords: np.ndarray = None                       # Coordinates of the lattice sites
    K_onsite: float = None                          # Strength of the onsite disorder distribution
    onsite_disorder: np.ndarray = None              # Disorder array for only the onsite case
    seed: int = None                                # Seed both for amorphicity and anderson disorder
    boundary: str = 'Open'                          # 'Open' or 'Closed' boundary conditions

    # Class fields that can only be set internally
    Nsites: int = field(init=False)                 # Number of sites in the cross-section
    neighbours: np.ndarray = field(init=False)      # Neighbours list for each site
    area: float = field(init=False)                 # Area of the wire's cross-section


    # Methods for building the lattice
    def build_lattice(self, crystalline=False):
        if self.w < 1e-10 and not crystalline:
            loger_amorphous.error('The amorphicity cannot be strictly 0')
            exit()
        self.generate_configuration(crystalline=crystalline)
        self.generate_neighbour_tree()

    def generate_neighbour_tree(self):
        if self.boundary == 'Closed':
            if self.r >= min(self.Nx, self.Ny) / 2:
                loger_amorphous.warning(
                    f'Neighbour cutoff r={self.r} must stay under half the system size '
                    f'(min(Nx, Ny) / 2 = {min(self.Nx, self.Ny) / 2}) for periodic boundary '
                    f'conditions to be unambiguous.'
                )

            coords_wrapped = np.mod(self.coords.T, [self.Nx, self.Ny])
            self.neighbours = KDTree(coords_wrapped, boxsize=[self.Nx, self.Ny]).query_ball_point(
                coords_wrapped, self.r)
        else:
            self.neighbours = KDTree(self.coords.T).query_ball_point(self.coords.T, self.r)
        for i in range(self.Nsites):
            self.neighbours[i].remove(i)

    def generate_configuration(self, crystalline=False):
        loger_amorphous.trace('Generating lattice and neighbour tree...')

        # Positions of x and y coordinates on the amorphous structure
        self.Nsites = int(self.Nx * self.Ny)
        if self.x is None and self.y is None:
            list_sites = np.arange(0, self.Nsites)
            x_crystal = list_sites % self.Nx
            y_crystal = list_sites // self.Nx
            if crystalline:
                self.x, self.y = x_crystal, y_crystal
            else:
                self.x, self.y = gaussian_point_set_2D(x_crystal, y_crystal, self.w, seed=self.seed)
        self.coords = np.array([self.x, self.y])

        # Set up preliminary disorder
        self.K_onsite = 0.

    def generate_onsite_disorder(self, K_onsite):
        loger_amorphous.trace('Generating disorder configuration...')
        self.K_onsite = K_onsite
        rng = np.random.default_rng(seed=self.seed)
        self.onsite_disorder = rng.uniform(-self.K_onsite, self.K_onsite, self.Nsites)

    # Setters and erasers
    def set_configuration(self, x, y):
        self.x, self.y = x, y

    def set_disorder(self, onsite_disorder, K_onsite):
        self.K_onsite = K_onsite
        self.onsite_disorder = onsite_disorder

    def erase_configuration(self):
        self.x, self.y = None, None

    def erase_disorder(self):
        self.onsite_disorder= None

    def plot_lattice(self, ax, sitecolor='deepskyblue', linkcolor='blue', alpha_site=1, alpha_link=1):

        # Lattice sites
        ax.scatter(self.x, self.y, color=sitecolor, s=50, alpha=alpha_site)

        # Neighbour links
        for site in range(self.Nsites):
            for n in self.neighbours[site]:
                ax.plot([self.x[site], self.x[n]], [self.y[site], self.y[n]], color=linkcolor,
                        alpha=alpha_link, linewidth=1)
                # ax.text(self.x[n] + 0.1, self.y[n] + 0.1, str(n))

@dataclass
class AmorphousLattice_2d_Kwant(kwant.builder.SiteFamily):
    def __init__(self, norbs, lattice, name=None):

        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError("The norbs parameter must be an integer > 0.")
            norbs = int(norbs)

        # Class fields
        loger_amorphous.trace('Initialising cross section as a SiteFamily...')
        self.norbs = norbs
        self.coords = np.array([lattice.x, lattice.y]).T
        self.Nsites = lattice.Nsites
        self.Nx = lattice.Nx
        self.Ny = lattice.Ny
        self.name = name
        self.canonical_repr = "1" if name is None else name

    def pos(self, tag):
        return self.coords[tag, :][0, :]

    def normalize_tag(self, tag):
        return ta.array(tag)

    def __hash__(self):
        return 1