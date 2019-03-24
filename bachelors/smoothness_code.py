def compute_smoothness_via_polynom(metric_values, approx_degree=1, window_size=1):
    '''
    computes MSE with approximation polynom
    Returns:
        MSE
    '''
    metric_values = np.array(metric_values)
    n = metric_values.shape[0]
    mse = 0.
    for i in range(n - window_size):
        x = np.arange(window_size, dtype=int) + i
        y = metric_values[x]
        poly = np.poly1d(np.polyfit(x, y, approx_degree))
        mse += (poly[(window_size + 1) // 2] - y[(window_size + 1) // 2]) ** 2
    return mse / (n - window_size)

