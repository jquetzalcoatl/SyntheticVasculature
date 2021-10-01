#Dirichlet-to-Neumann tools

import numpy as np
import matplotlib.pyplot as plt

def schur_comp(M,idx_set):
    """
    computes the schur complement/the pieces of the schur complement for a matrix M
    """
    comp_idx = [i for i in range(len(M)) if i not in idx_set]
    A = M[np.idx_(idx_set,idx_set)]
    B = M[np.idx_(idx_set,comp_idx)]
    C = M[np.idx_(comp_idx,idx_set)]
    D = M[np.idx_(comp_idx,comp_idx)]


    return A - B @ np.linalg.inv(D) @ C

def steklov_spec(M,bdy_idx):
    """
    computes the steklov spectrum corresponding to a Laplacian M and boundary nodes bdy_idx
    """

    eig_vals, eigvecs = np.linalg.eig(schur_comp(M,bdy_idx))

    eig_val_sort = np.argsort(eigvals)

    return eig_vals[eig_val_sort], eigvecs.T[eig_val_sort]
