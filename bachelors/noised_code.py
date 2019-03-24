def noised_soft_dcg(real_marks, scores, n_docs_scored=5, sigma=1):
    n = scores.shape[0]
    n_docs_scored = min(n_docs_scored, n)
    discounts = 1. / (np.arange(n_docs_scored) + 1)
    
    dcg = 0.
    n_generations = 1000
    for it in range(n_generations):
        noised_scores = scores + np.random.normal(scale=sigma, size=n)
        docs_permutation = np.argsort(noised_scores)[::-1][:n_docs_scored]
        cur_dcg = np.sum(real_marks[docs_permutation] * discounts)
        dcg += cur_dcg

    return dcg / n_generations

