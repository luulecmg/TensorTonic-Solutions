import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)

    if p == 1.0:
        mask = np.zeros_like(x)
        return x * mask, mask
    
    if rng is not None:
        prob = rng.random(x.shape)
    else:
        prob = np.random.random(x.shape)
        
    dropout_pattern = np.full(x.shape, 1 / (1 - p))
    dropout_pattern[prob < p] = 0

    x = x * dropout_pattern

    return (x, dropout_pattern)