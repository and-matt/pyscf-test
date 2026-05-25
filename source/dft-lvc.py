'''
How to use this script:
python dft-lvc-loc.py -ncpus 4 -xyz water.xyz
'''

from pyscf import gto, dft, eph, lib
from pyscf.geomopt.geometric_solver import optimize
import numpy as np
import time, h5py, argparse


# --- Read number of CPUs and input geometry ---
parser = argparse.ArgumentParser()
parser.add_argument("-ncpus", type=int, default=1, help="Number of CPU cores")
parser.add_argument("-xyz", type=str, help="Input geometry xyz file")
args = parser.parse_args()
ncpus = args.ncpus
lib.num_threads(ncpus)
xyzfile = args.xyz


# --- Load molecule ---
mol = gto.M(atom=xyzfile, basis='cc-pvdz')


# --- Run optimisation ---
opt_mf = mol.RKS(xc='camb3lyp').PCM()
opt_mf.with_solvent.method = 'IEF-PCM'
opt_mf.with_solvent.eps = 2.38 # toluene
start = time.time()
mol = optimize(opt_mf)
end = time.time()
elapsed = end - start
print(f'Optimize: {elapsed:.6f} seconds')


# --- Run equilibrium DFT calculation ---
dftcalc = mol.RKS(xc='camb3lyp').PCM()
dftcalc.with_solvent.method = 'IEF-PCM'
dftcalc.with_solvent.eps = 2.38 # toluene
dftcalc.chkfile = 'calc.chk'
start = time.time()
mf = dftcalc.run()
end = time.time()
elapsed = end - start
print(f'DFT calc: {elapsed:.6f} seconds')


# --- Run electron–phonon calculation ---
mf.xc = 'b3lyp'
start = time.time()
eph_obj = eph.EPH(mf)
ephmat, omega = eph_obj.kernel()
_, modevec = eph_obj.get_mode()
end = time.time()
elapsed = end - start
print(f'EPH calc: {elapsed:.6f} seconds')


# --- Save electron-phonon couplings and frequencies ---
hdf5_file_path = 'dft-lvc-out.h5'
with h5py.File(hdf5_file_path, 'w') as hdf5_file:
    hdf5_file.create_dataset('ephmat',  data=ephmat)
    hdf5_file.create_dataset('omega',   data=omega)
    hdf5_file.create_dataset('modevec', data=modevec)
