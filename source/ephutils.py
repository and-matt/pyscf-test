"""
Functions used for pre- and post-processing of PySCF EPH calculations.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def cartesian_modevec(modevec, mass):
    """Convert Hessian eigenvector to cartesian displacement array."""
    natm = len(mass)
    modevec_cart = modevec.reshape(natm, 3)
    for a in range(natm):
        modevec_cart[a] = modevec_cart[a] / np.sqrt(mass[a])
    modevec_cart = modevec_cart / np.linalg.norm(modevec_cart)
    return modevec_cart


def multiconfig_vibronic_coupling(g, which="Schroder"):
    """
    Convert vibronic coupling from frontier MO basis to the singlet space 
    formed by symmetric / antisymmetric singlet exciton (SE+, SE-), 
    charge-transfer (CT+, CT-), and correlated triplet-triplet (TT) states.
    Molecular orbital index ordering is assumed to be hA, lA, hB, lB
    (localized HOMO/LUMO on monomer A/B).
    Further reading:
    - Schroder et al. (2019) https://doi.org/10.1038/s41467-019-09039-7
    - Smith & Michl (2010) https://doi.org/10.1021/cr1002613
    - Alguire et al. (2015) https://doi.org/10.1021/jp510777c
    """
    nvib = g.shape[0]
    W = np.zeros((nvib, 5, 5))
    if which=="Schroder":
        W[:,0,1] = 0.5 * ( g[:,2,2] - g[:,0,0] - g[:,3,3] + g[:,1,1] )
        W[:,2,3] = 0.5 * ( g[:,2,2] - g[:,0,0] + g[:,3,3] - g[:,1,1] )
        W[:,0,2] = g[:,1,3] - g[:,0,3]
        W[:,1,3] = g[:,1,3] + g[:,0,3]
        W[:,4,2] = np.sqrt(3) / 2 * ( g[:,1,2] + g[:,0,3] )
        W[:,4,3] = np.sqrt(3) / 2 * ( g[:,1,2] - g[:,0,3] )
    elif which=="Smith":
        W[:,2,4] = np.sqrt(3) / 2 * ( g[:,1,2] + g[:,3,0] )
        W[:,3,4] = np.sqrt(3) / 2 * ( g[:,1,2] - g[:,3,0] )
        W[:,2,0] = 0.5 * ( g[:,1,3] - g[:,0,2] - g[:,2,0] - g[:,3,1] )
        W[:,2,1] = 0.5 * ( g[:,1,3] + g[:,0,2] - g[:,2,0] + g[:,3,1] )
        W[:,3,0] = 0.5 * ( g[:,1,3] - g[:,0,2] + g[:,2,0] + g[:,3,1] )
        W[:,3,1] = 0.5 * ( g[:,1,3] + g[:,0,2] + g[:,2,0] - g[:,3,1] )
    W = W + W.transpose(0,2,1)
    return W


def vectorize_ephmat(ephmat, normalize=True):
    """Turn e-ph coupling tensor (nvib,nel,ne) into a matrix (nvib,nel*nel)."""
    data = np.copy(ephmat).reshape(ephmat.shape[0], ephmat.shape[1]*ephmat.shape[2])
    if normalize:
        for i in range(data.shape[0]):
            data[i] = data[i] / np.linalg.norm(data[i])
    return data


def cluster_ephmat(data, n_clusters=3, random_state=0, plot=True):
    """K-means clustering and plotting in 2D using tSNE."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data)
    # Dimensionality reduction to 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    centers = kmeans.cluster_centers_
    nc = centers.shape[0]
    data_and_centers = np.vstack((data, centers))
    data_and_centers_2d = tsne.fit_transform(data_and_centers)
    data_2d = data_and_centers_2d[:-nc]
    centers_2d = data_and_centers_2d[-nc:]
    if plot:
        plt.figure()
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker="*", s=200, c="r")
        plt.axis('off')
        plt.tight_layout()
        plt.title(f"K-means clustering (k={n_clusters})")
        plt.show()
    return kmeans


def cluster_indices(labels):
    """Find normal mode indices belonging to each cluster (labels=kmeans.labels_)."""
    mode_indices = []
    nc = labels.max() + 1
    for c in range(nc):
        where_cluster = np.where(labels==c)[0]
        mode_indices += [where_cluster]
    return mode_indices


def lineshape(x, freq, hwhm=5):
    """Anti-symmetrized Lorentzian, equivalent to Browninan spectral density."""
    spec_dens = 2 * x * freq * hwhm  / np.arctan(freq / hwhm) \
                / ((x-freq)**2 + hwhm**2) / ((x+freq)**2 + hwhm**2)
    return spec_dens


def effective_spectral_density(fgrid, omega, lvc, hwhm=5, plot=True):
    """Calculate effective spectral density from LVC operators."""
    jgrid = np.zeros(fgrid.shape)
    for f,w in zip(omega, lvc):
        jgrid += np.linalg.norm(w)**2 * lineshape(fgrid, f, hwhm)
    if plot:
        plt.figure()
        plt.plot(fgrid, jgrid, "k-")
        plt.xlabel(r"Frequency / cm$^{-1}$")
        plt.ylabel(r"Spectral Density / cm$^{-1}$")
        plt.tight_layout()
        plt.show()
    return jgrid


def eigh_block(mat, block_size):
    """
    Returns block-diagonalized matrix in blocks of size block_size (list) 
    with block-eigenvalues in ascending value, and block-unitary.
    """
    if np.isscalar(block_size):
        block_size = [block_size, len(mat) - block_size]
    if sum(block_size) != len(mat):
        print("Sum of block dimensions must equal matrix dimension!")
    block_unitaries = []
    nblocks = len(block_size)
    for i in range(nblocks):
        start = sum(block_size[:i])
        end = start + block_size[i]
        mat_sub = mat[start:end, start:end]
        _, vec_sub = sp.linalg.eigh(mat_sub)
        block_unitaries += [vec_sub]
    u_block = sp.linalg.block_diag(*block_unitaries)
    mat_block_diag = u_block.T @ mat @ u_block
    return mat_block_diag, u_block


def diagonalize_block(H, start, size):
    N = H.shape[0]
    stop = start + size
    Hblk = H[start:stop, start:stop]
    _, Ublk = np.linalg.eigh(Hblk)
    Uembed = np.eye(N, dtype=complex)
    Uembed[start:stop, start:stop] = Ublk
    H = Uembed.conj().T @ H @ Uembed
    return H, Uembed


def nn_block_mapping(mat, first_block_size, tolerance=1e-8):
    H = np.array(mat, dtype=complex, copy=True)
    N = H.shape[0]
    if H.shape != (N, N):
        raise ValueError("H must be square")
    Utot = np.eye(N, dtype=complex)
    block_sizes = []
    current_start = 0
    current_size = first_block_size
    H, U = diagonalize_block(H, current_start, current_size)
    Utot = U
    block_sizes.append(current_size)
    while True:
        # Define current block
        current_stop = current_start + current_size
        remainder_size = N - current_stop
        if remainder_size == 0:
            break
        # SVD on block-remainder coupling (rectangular block)
        V = H[current_start:current_stop, current_stop:]
        if np.allclose(V, 0):
            break
        _, s, Wdag = np.linalg.svd(V)
        smax = s[0]
        if smax == 0:
            break
        next_size = np.count_nonzero(s > tolerance * smax)
        next_size = min(next_size, remainder_size)
        # Rotate remainder by right singular vectors
        W = Wdag.conj().T
        Uembed = np.eye(N, dtype=complex)
        Uembed[current_stop:, current_stop:] = W
        H = Uembed.conj().T @ H @ Uembed
        Utot = Utot @ Uembed
        # New block = first next_size states of rotated remainder
        next_start = current_stop
        H, U = diagonalize_block(H, next_start, next_size)
        Utot = Utot @ U
        block_sizes.append(next_size)
        current_start = next_start
        current_size = next_size
    return H, Utot, block_sizes

