def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    if k <= 0:
        raise ValueError(f"K must be a positive integer")
    elif len(recommended) < k:
        raise ValueError(f"Recommended must contain at least {k} items")
    elif not relevant:
        return [0.0, 0.0]
    
    top_k_recommended = recommended[:k]
    relevant = set(relevant)

    hits = sum(1 for item in top_k_recommended if item in relevant)

    precision = hits / k
    recall = hits / len(relevant)

    return [precision, recall]
    
    