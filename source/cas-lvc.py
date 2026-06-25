'''
Calculate linear electron-phonon couplings using DFT in PySCF.

Results are written in file out.h5, containing e-ph coupling matrix (cm^-1), 
mode frequencies (cm^-1) and vectors (mass-weighted Hessian eigenvectors).
Equilibrium DFT calculation results are stored in calc.chk.

How to use this script:
python cas-lvc.py -ncpus 4 -chk calc.chk -h5 loc.h5
'''


import time
import h5py
import argparse
import numpy as np
from pyscf import lib, mcscf
from pyscf.geomopt.geometric_solver import optimize
from dftutils import mf_from_chk, Boys_frontier_orbitals, xyz_string

NORB = 4
NELEC = 4
#NROOTS = 17
HARTREE2EV = 27.2114079527
PCM_METHOD = 'IEF-PCM'
SOLVENT_EPSILON = 2.38 # toluene


# Read number of CPUs and input geometry
parser = argparse.ArgumentParser()
parser.add_argument('-ncpus', type=int, default=1, help='Number of CPU cores')
parser.add_argument('-chk', type=str, help='Checkfile path')
parser.add_argument('-h5', type=str, help='h5 file path')
parser.add_argument('-roots', type=int, default=12, help='Number of CASCI roots')
args = parser.parse_args()
ncpus = args.ncpus
chk = args.chk
h5file = args.h5
NROOTS = args.roots

print('Importing data...')

# Set number of CPUs
lib.num_threads(ncpus)


# import mf object
mf = mf_from_chk(chk)


# import localized orbitals from h5 file
with h5py.File(h5file, "r") as f:
    fock_ao = np.array(list(f["fock_ao"])) # hartree
    mo_coeff_block = np.array(list(f["mo_coeff_block"]))
    integral2e = np.array(list(f["integral2e"]))
_, idx = Boys_frontier_orbitals(mf, NORB)
# mo_coeff_block orbitals order: hA, lA, hB, lB
mo = mf.mo_coeff.copy()
mo[:, idx] = mo_coeff_block


# set up and run CASCI calculation
print('Setting up CASCI calculation...')
mc = mcscf.CASCI(mf, NORB, NELEC)
# attach PCM
mc = mc.PCM()
mc.with_solvent.method = PCM_METHOD
mc.with_solvent.eps = SOLVENT_EPSILON
# use ground-state solvent polarization
# - this avoids inconsistencies with state averaging
mc.with_solvent.state_id = 0
# molecular orbitals
mc.mo_coeff = mo
mc.fcisolver.nroots = NROOTS
weights = np.ones(NROOTS) / NROOTS
mc = mcscf.state_average(mc, weights)
mc.verbose = 4
mc.chkfile = 'casci.chk'
print('Starting CASCI calculation...')
start = time.time()
mc.kernel()
end = time.time()
elapsed = end - start
print(f'CASCI took {elapsed:.6f} s.')


print("CASCI energies:")
energies_meV = (mc.e_states - mc.e_states[0]) * HARTREE2EV * 1e3
for i, e in enumerate(energies_meV):
    print(f"Root {i:2d} : {e:.8f} meV")


for root in range(NROOTS):
    print(f"\n==== Root {root} ====")
    ci_info = mc.fcisolver.large_ci(mc.ci[root], norb=NORB, nelec=NELEC)
    for coeff, alpha, beta in ci_info:
            print(f"{coeff:+8.4f}   {alpha}   {beta}")

