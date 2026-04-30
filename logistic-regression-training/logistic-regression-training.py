import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def loss_function(pred, labels):
    loss = -(labels * np.log(pred) + (1-labels)*np.log(1-pred)).mean()
    return loss

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    w = np.zeros((X.shape[1], 1))
    b = np.array(0.0, dtype=float)

    y = y.reshape(-1, 1)
    N = X.shape[0]

    for step in range(steps):
        logits = X@w + b
        pred = _sigmoid(logits)

        loss = loss_function(pred, y)

        gradient_w = (X.T @ (pred - y)) * (1 / N)
        gradient_b = (pred - y).mean()

        w = w - lr * gradient_w
        b = b - lr * gradient_b

    return (w.flatten(), b.item())