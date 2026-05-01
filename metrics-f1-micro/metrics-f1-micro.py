
def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    import numpy as np
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch:{len(y_true)} vs {len(y_pred)}")

    TP = sum(y_true == y_pred).item()
    FP = FN = sum(y_true != y_pred).item()

    return TP * 2 / (2 * TP + FN + FP)