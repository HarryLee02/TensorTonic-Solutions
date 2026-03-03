import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A_shape = np.shape(A)
    A_T = np.zeros( (A_shape[1], A_shape[0]) )
    
    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            A_T[j][i] = A[i][j]

    
    return A_T
