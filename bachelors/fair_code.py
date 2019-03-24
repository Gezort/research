def fair_soft_dcg(real_marks, scores, n_docs_scored, sigma=1, sort_by_real_scores=False):
    n = scores.shape[0]
    if n_docs_scored > 2:
        n = min(n, 7)
    n_docs_scored = min(n_docs_scored, n)
    discounts = 1. / (np.arange(n_docs_scored) + 1) 

    dcg = 0.
    real_marks = np.array(real_marks)
    
    if sort_by_real_scores:
        ind = np.argsort(real_marks)
    else:
        ind = np.argsort(scores)
    ind = ind[::-1][:n]
    real_marks = real_marks[ind]
    scores = scores[ind]
    scores /= sigma
    scores -= scores.max()
    
    # exponentiation problem check
    if scores.min() < -20:
        return -1

    scores = np.exp(scores)
    Z = np.sum(scores)

    for prm in permutations(range(n), n_docs_scored):
        prob = 1.
        perm = list(prm)
        sorted_scores = scores[perm]
        z = Z
        for i in range(n_docs_scored):
            prob *= sorted_scores[i] / z
            z -= sorted_scores[i]

        dcg += prob * np.sum(real_marks[perm] * discounts)

    return dcg

