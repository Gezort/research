def missrank_probs(scores, sigma=1, prob_model='norm'):
    '''Calculates matrix P[ij] of probabilities that doc_i will rank above doc_j
    Input:
        scores : array_like, ranking function predictions
        sigma : std
        prob_model : string, one of ['norm', 'exp']
    Output:
        P : 2d array with predicted probabilities
    '''
    if prob_model == 'norm':
        n = scores.shape[0]
        mean = scores.reshape(-1, 1) - scores
        std = np.ones((n,n)) * sigma * 2
        return norm.cdf(1000000, loc=mean, scale=std) - norm.cdf(0, loc=mean, scale=std)
    elif prob_model == 'exp':
        diffs = scores.reshape(-1, 1) - scores
        return np.exp(sigma * diffs)

def rank_distribution(scores, sigma=1, prob_model='norm'):
    '''
    Input:
        scores : array_like, ranking function predictions 
    Output:
        P : 2d array with rank distribution for each of N documents
    '''
    n = scores.shape[0]
    P = missrank_probs(scores, sigma, prob_model)
    res = np.zeros((n,n))
    res[:,0] = 1
    for i in range(n):
        shifted_distribution = np.roll(res, 1, axis=1)
        shifted_distribution[:, 0] = 0
        prob = P[i].reshape(-1,1)
        prob[i,0] = 0
        res = shifted_distribution * prob + (1 - prob) * res
    return res

def expected_discount(scores, sigma=1, prob_model='norm'):
    P = rank_distribution(scores, sigma, prob_model)
    n = scores.shape[0]
    P /= (1 + np.arange(n))
    return np.sum(P, axis=1)

def soft_dcg(real_marks, scores, sigma=1, prob_model='norm'):
    discounts = expected_discount(scores, sigma, prob_model)
    return np.sum(real_marks * discounts)

