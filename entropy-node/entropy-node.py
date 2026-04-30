import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if len(y) == 0:
        return 0.0
    
    y = np.array(y)

    classes, classes_count = np.unique(y, return_counts=True)

    class_prob = classes_count / classes_count.sum()

    entropy_score = -np.sum(class_prob * np.log2(class_prob))
    
    return entropy_score.item()