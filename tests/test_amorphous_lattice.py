import logging
import numpy as np

from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.amorphous_rashba import displacement2D_kwant


class FakeSite:
    """Minimal stand-in for a kwant Site, just enough for displacement2D_kwant."""
    def __init__(self, x, y):
        self.pos = (x, y)


def test_displacement_wraps_under_closed_boundary():
    # Two sites just past opposite edges of an 8x8 box are neighbours under
    # periodic boundaries, but far apart under open ones
    Nx, Ny = 8, 8
    s0 = FakeSite(-0.2, 4.0)
    s1 = FakeSite(7.9, 4.0)

    r_closed, _ = displacement2D_kwant(s0, s1, boundary='Closed', Nx=Nx, Ny=Ny)
    r_open, _ = displacement2D_kwant(s0, s1, boundary='Open', Nx=Nx, Ny=Ny)

    assert r_closed < 0.5
    assert r_open > Nx - 1


def test_periodic_neighbour_tree_wraps_around_corner():
    # A crystalline lattice's corner site has 2 neighbours under open boundaries
    # (its two interior bonds) and 4 under periodic ones (plus the two wraparound
    # bonds to the opposite edges)
    lattice_open = AmorphousLattice_2d(Nx=6, Ny=6, w=0.1, r=1.3)
    lattice_open.build_lattice(crystalline=True)
    assert len(lattice_open.neighbours[0]) == 2

    lattice_closed = AmorphousLattice_2d(Nx=6, Ny=6, w=0.1, r=1.3)
    lattice_closed.boundary = 'Closed'
    lattice_closed.build_lattice(crystalline=True)
    assert len(lattice_closed.neighbours[0]) == 4


def test_large_cutoff_warns_under_closed_boundary(caplog):
    lattice = AmorphousLattice_2d(Nx=4, Ny=4, w=0.1, r=3.0)
    lattice.boundary = 'Closed'
    with caplog.at_level(logging.WARNING, logger='modules.AmorphousLattice_2d'):
        lattice.build_lattice(crystalline=True)
    assert any('half the system size' in record.message for record in caplog.records)


def test_small_cutoff_does_not_warn_under_closed_boundary(caplog):
    lattice = AmorphousLattice_2d(Nx=6, Ny=6, w=0.1, r=1.3)
    lattice.boundary = 'Closed'
    with caplog.at_level(logging.WARNING, logger='modules.AmorphousLattice_2d'):
        lattice.build_lattice(crystalline=True)
    assert not any('half the system size' in record.message for record in caplog.records)
