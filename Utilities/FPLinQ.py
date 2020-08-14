# For any reproduce, further research or development, please kindly cite our SPAWC conference paper: 
# @Conference{cluster_schedule, 
#             author = "W. Cui and W. Yu", 
#             title = "A Clustering Approach to Wireless Scheduling", 
#             booktitle = "IEEE Workshop Signal Process. Advances Wireless Commun. (SPAWC)", 
#             year = 2020, 
#             month = may }
#
# Implementation of FPLinQ algorithm including iterative optimization

import numpy as np
import utils
import time

# Parallel computation over multiple layouts
def FP_optimize(general_para, g, weights):
    number_of_samples, N, _ = np.shape(g)
    assert np.shape(g)==(number_of_samples, N, N)
    assert np.shape(weights)==(number_of_samples, N)
    g_diag = utils.get_directlink_channel_losses(g)
    g_nondiag = utils.get_crosslink_channel_losses(g)
    # For matrix multiplication and dimension matching requirement, reshape into column vectors
    weights = np.expand_dims(weights, axis=-1)
    g_diag = np.expand_dims(g_diag, axis=-1)
    x = np.ones([number_of_samples, N, 1])
    tx_power = general_para.tx_power
    output_noise_power = general_para.output_noise_power
    tx_powers = np.ones([number_of_samples, N, 1]) * tx_power  # assume same power for each transmitter
    # In the computation below, every step's output is with shape: number of samples X N X 1
    for i in range(general_para.FP_iter_amount):
        # Compute z
        p_x_prod = x * tx_powers
        z_denominator = np.matmul(g_nondiag, p_x_prod) + output_noise_power
        z_numerator = g_diag * p_x_prod
        z = z_numerator / z_denominator
        # compute y
        y_denominator = np.matmul(g, p_x_prod) + output_noise_power
        y_numerator = np.sqrt(z_numerator * weights * (z + 1))
        y = y_numerator / y_denominator
        # compute x
        x_denominator = np.matmul(np.transpose(g, (0,2,1)), np.power(y, 2)) * tx_powers
        x_numerator = y * np.sqrt(weights * (z + 1) * g_diag * tx_powers)
        x_new = np.power(x_numerator / x_denominator, 2)
        x_new[x_new > 1] = 1  # thresholding at upperbound 1
        x = x_new
    assert np.shape(x)==(number_of_samples, N, 1)
    x_final = np.squeeze(x, axis=-1)
    return x_final


