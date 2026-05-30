"""
Functions used for pre- and post-processing of PySCF DFT calculations.
"""

import numpy as np
import scipy as sp
from pyscf import dft, lib, lo, ao2mo
from pyscf.tools import cubegen 


def xyz_string(symbls, coords, filename=None):
    """
    Create xyz string from atomic symbols and coordinates.
    Write it to file if filename is provided.
    """
    xyz = f"{len(symbls)}\n\n"
    for s, (x,y,z) in zip(symbls, coords):
        xyz += f"{s} {x} {y} {z}\n"
    if isinstance(filename, str):
        with open(filename, "w") as f:
            f.write(xyz)
    return xyz


def mf_from_chk(chkfile):
    """Return SCF object created from checkfile."""
    mol = lib.chkfile.load_mol(chkfile)
    mf = dft.RKS(mol)
    mf.__dict__.update(lib.chkfile.load(chkfile, "scf"))
    return mf


def Boys_frontier_orbitals(mf, norb):
    """
    !!! Important: norb must be even !!!
    Identify the frontier molecular orbitals
    (HOMO - norb/2 + 1, ..., LUMO + norb/2).
    Return Boys localized MO coefficients and frontier indices.
    """
    # Identify frontier orbital indices
    mo_occ = mf.mo_occ
    nocc = np.count_nonzero(mo_occ > 0)
    homo = nocc - 1
    fomo = homo - int(norb/2) + 1 # lowest frontier OMO
    fumo = homo + int(norb/2) + 1 # highest frontier UMO
    frontier_idx = list(range(fomo,fumo))
    # Localize frontier orbitals using Boys method
    mo_coeff = mf.mo_coeff
    mo_front = mo_coeff[:, frontier_idx]
    mo_front_Boys = lo.Boys(mf.mol, mo_front).kernel()
    return mo_front_Boys, frontier_idx


def atomic_populations(mf, mo_coeff):
    """Calculate atomic populations of molecular orbitals."""
    natm = mf.mol.natm
    atomic_pop = np.zeros((natm, mo_coeff.shape[1]))
    labels = mf.mol.ao_labels()
    for i, C in enumerate(mo_coeff.T):
        dm_mo = np.outer(C, C)
        pop, _ = mf.mulliken_pop(mf.mol, dm_mo, verbose=0)
        for a in range(natm):
            # find indices of all atomic orbitals on the same atom
            idx = [k for k, l in enumerate(labels) if int(l.split()[0])==a]
            # calculate total population on an atom
            atomic_pop[a, i] = np.sum(pop[idx])
    return atomic_pop


def pair_neighboring_orbitals(mf, mo_coeff):
    """
    !!! Important: mo_coeff.shape[1] must be even !!!
    Calculate center of mass of molecular orbitals and reorder them so 
    that they appear in pairs of orbitals with close center of mass,
    i.e. on the same molecular fragment. 
    """
    norb = mo_coeff.shape[1]
    atomic_pop = atomic_populations(mf, mo_coeff)
    # centers of mass of the electronic densities
    centers = atomic_pop.T @ mf.mol.atom_coords(unit='Angstrom')
    # Run through each centers and find its nearest neighbor
    remaining = list(range(norb))
    ordered_indices = []
    while remaining:
        i = remaining.pop(0)
        # distances from point i to all remaining points
        dists = np.linalg.norm(centers[remaining] - centers[i], axis=1)
        # nearest neighbor
        jpos = np.argmin(dists)
        j = remaining.pop(jpos)
        ordered_indices.extend([i, j])
    mo_coeff_sorted = mo_coeff.copy()[:,ordered_indices]
    return mo_coeff_sorted, ordered_indices


def block_frontier_orbitals(mf, mo_coeff, fock=None):
    """
    Block-diagonalise the Fock matrix within 2-dimensional blocks.
    Return the localized orthogonal molecular orbital coefficients.
    """
    if fock is None:
        fock = fock_matrix(mf)
    npairs = int(mo_coeff.shape[1]/2)
    mo_coeff_pairs, _ = pair_neighboring_orbitals(mf, mo_coeff)
    fock_pairs = mo_coeff_pairs.T @ fock @ mo_coeff_pairs
    # block diagonalise Fock matrix for each pair
    mo_coeff_block = np.zeros(mo_coeff.shape)
    u = []
    for i in range(npairs):
        fock_sub = fock_pairs[2*i:2*i+2, 2*i:2*i+2]
        _, vec_sub = eighsort(fock_sub)
        u += [vec_sub]
    u_block = sp.linalg.block_diag(*u)
    mo_coeff_block = mo_coeff_pairs @ u_block
    return mo_coeff_block


def fock_matrix(mf):
    dm = mf.make_rdm1()
    fock_ao = mf.get_fock(dm=dm)
    return fock_ao


def eighsort(mat):
    val, vec = np.linalg.eigh(mat)
    i = np.argsort(val)
    return val[i], vec[:,i]


def two_electron_integrals(mol, mo_coeff):
    """Return two-electron integrals indexed in chemist notation (ij|kl)."""
    eri4mo = ao2mo.kernel(mol, mo_coeff)
    eri4mo = ao2mo.restore(1, eri4mo, 4)
    return eri4mo


def save_orbital(mol, cubefile, mo, grid=(100,50,50)):
    """Save cube file for a molecular orbital."""
    nx, ny, nz = grid
    cubegen.orbital(mol, cubefile, mo, nx=nx, ny=ny, nz=nz)
    return

