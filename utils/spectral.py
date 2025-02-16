import numpy as np
import torch


def spectral_symmetrization(W, return_decompose=True):
    # W shoud be skew-symmetric
    assert np.allclose(np.linalg.norm(W + W.T, ord='fro'), 1e-10)

    # Pytorch does not support eigen-decomposition of non-Hermitian matrix.
    if 'torch' in str(type(W)):
        dtype_ = W.type()
        W = W.numpy()
    else:
        dtype_ = str(type(W))

    e_complex, V_complex = np.linalg.eig(W)
    # Discard zero eigenvalues and corresponding eigenvectors.
    idx = abs(e_complex.imag) > 1e-12
    V_complex = V_complex[:,idx]
    e_complex = e_complex[idx]

    dtype_ = W.dtype
    k = W.shape[0]

    # Eigenvectors of skew-symmetrix matrix are of form (a+ib, a-ib), hence we 
    # transform this into ((a-b)/2, (a+b)/2) by multiplying rotation matrix and 
    # taking real-value part. 
    rot = np.kron(np.eye(len(e_complex)//2), np.array([[1, -1j], [-1j, 1]]))
    V = torch.from_numpy((V_complex @ rot).real)
    E_skew = torch.from_numpy((np.diag(e_complex)@rot).real)
    E_sym = torch.diagonal(E_skew, offset=1)[::2].abs().repeat(2,1).T.reshape(-1).abs().diag_embed()
    if return_decompose:
        return V @ E_sym @ V.T
    else:
        return V, E_sym