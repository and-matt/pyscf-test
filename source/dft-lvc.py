'''
Calculate linear electron-phonon couplings using DFT in PySCF.
Results are written in file out.h5, containing e-ph coupling matrix (cm^-1), 
mode frequencies (cm^-1) and vectors (mass-weighted Hessian eigenvectors).
Equilibrium DFT calculation results are stored in calc.chk.

How to use this script:
python dft-lvc.py -ncpus 4 -xyz water.xyz --opt
'''


import time
import h5py
import argparse
from pyscf import gto, eph, lib
from pyscf.geomopt.geometric_solver import optimize
from pyscfutils import xyz_string

BASIS_SET = 'cc-pvdz'
FUNCTIONAL = 'camb3lyp'
EPH_FUNCTIONAL = 'b3lyp'
PCM_METHOD = 'IEF-PCM'
SOLVENT_EPSILON = 2.38 # toluene
CUTOFF_FREQUENCY = 80 # cm^-1


# Read number of CPUs and input geometry
parser = argparse.ArgumentParser()
parser.add_argument('-ncpus', type=int, default=1, help='Number of CPU cores')
parser.add_argument('-xyz', type=str, help='Input geometry xyz file')
parser.add_argument('--opt', action='store_true', help='Optimize geometry')
args = parser.parse_args()
ncpus = args.ncpus
xyzfile = args.xyz
opt = args.opt


# Load molecule and set number of CPUs
mol = gto.M(atom=xyzfile, basis=BASIS_SET)
lib.num_threads(ncpus)


# Optimize geometry
if opt:
    opt_mf = mol.RKS(xc=FUNCTIONAL).PCM()
    opt_mf.with_solvent.method = PCM_METHOD
    opt_mf.with_solvent.eps = SOLVENT_EPSILON
    start = time.time()
    mol = optimize(opt_mf)
    end = time.time()
    elapsed = end - start
    print(f'Geometry optimization took {elapsed:.6f} s.')
    # Save optimized geometry
    _ = xyz_string(mol.elements, mol.atom_coord(unit='Angstrom'), 'opt.xyz')


# Calculate equilibrium electronic structure
dftcalc = mol.RKS(xc=FUNCTIONAL).PCM()
dftcalc.with_solvent.method = PCM_METHOD
dftcalc.with_solvent.eps = SOLVENT_EPSILON
dftcalc.chkfile = 'calc.chk'
start = time.time()
mf = dftcalc.run()
end = time.time()
elapsed = end - start
print(f'DFT calculation took {elapsed:.6f} s.')


# Calculate electron–phonon couplings
mf.xc = EPH_FUNCTIONAL
start = time.time()
eph_obj = eph.EPH(mf, cutoff_frequency=CUTOFF_FREQUENCY, keep_imag_frequency=True)
# electron-phonon matrix units cm^-1, shape (nmodes, nao, nao)
ephmat, omega = eph_obj.kernel()
# mode vectors units mass-weighted coordinates, shape (3*natm, nmodes)
_, modevec = eph_obj.get_mode()
end = time.time()
elapsed = end - start
print(f'EPH calculation took {elapsed:.6f} s.')


# Save electron-phonon couplings, frequencies and mode vectors
hdf5_file_path = 'out.h5'
with h5py.File(hdf5_file_path, 'w') as hdf5_file:
    hdf5_file.create_dataset('ephmat',  data=ephmat)
    hdf5_file.create_dataset('omega',   data=omega)
    hdf5_file.create_dataset('modevec', data=modevec)
