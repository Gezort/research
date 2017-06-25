def compute_dcg_deviation(dcg_values, metric_values):
    '''
    computes MSE between dcg and (a * metric + b) dcg approximation
    Returns:
        MSE
    '''
    metric_values = np.array(metric_values)
    dcg_values = np.array(dcg_values)
    metric_values *= np.mean(dcg_values) / np.mean(metric_values)
    p = np.poly1d(np.polyfit(metric_values, dcg_values, 1))
    return ((dcg_values - p(metric_values)) ** 2).mean()

