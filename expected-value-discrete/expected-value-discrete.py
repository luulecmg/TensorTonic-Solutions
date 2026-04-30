import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    if not 1 - 1e-6 <= sum(p) <= 1 + 1e-6:
        raise ValueError(f"Prob should be sum up to 1, got {sum(p)}")
        
    x = np.array(x)
    p = np.array(p)

    if x.shape != p.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {p.shape}")

    return np.sum(x * p).item()
