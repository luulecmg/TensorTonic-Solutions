import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    if x == y == 0:
        return 0.0
        
    x = np.asarray(x)
    y = np.asarray(y)

    return np.sqrt(np.sum((x - y)**2)).item()