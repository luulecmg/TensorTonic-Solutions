import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch {len(y_true)} vs {len(y_pred)}")
    
    n_samples = len(y_true)
    prob = y_pred[np.arange(n_samples), y_true]

    ce_loss = -np.log(prob).mean()
    
    return ce_loss