import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    if len(A) == 0 or len(A[0]) ==0:
        return np.array([])
    # Write code here
    A = np.array(A)
    H, W = A.shape
    B = np.empty((W, H))

    for i in range(H):
        for j in range(W):
            B[j, i] = A[i, j]

    return B

    
