# For any reproduce, further research or development, please kindly cite our SPAWC conference paper: 
# @Conference{cluster_schedule, 
#             author = "W. Cui and W. Yu", 
#             title = "A Clustering Approach to Wireless Scheduling", 
#             booktitle = "IEEE Workshop Signal Process. Advances Wireless Commun. (SPAWC)", 
#             year = 2020, 
#             month = may }
#
# Utility functions used by other scripts

import numpy as np
import matplotlib.pyplot as plt
import general_parameters
import FPLinQ
import time

# Generate layout one at a time
def layout_generate(general_para):
    N = general_para.number_of_links
    # first, generate transmitters' coordinates
    tx_xs = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    tx_ys = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    while(True): # loop until a valid layout generated
        # generate rx one by one rather than N together to ensure checking validity one by one
        rx_xs = []; rx_ys = []
        for i in range(N):
            got_valid_rx = False
            while(not got_valid_rx):
                pair_dist = np.random.uniform(low=general_para.shortest_directlink_length, high=general_para.longest_directlink_length)
                pair_angles = np.random.uniform(low=0, high=np.pi*2)
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)
                if(0<=rx_x<=general_para.field_length and 0<=rx_y<=general_para.field_length):
                    got_valid_rx = True
            rx_xs.append(rx_x); rx_ys.append(rx_y)
        # For now, assuming equal weights and equal power, so not generating them
        layout = np.concatenate((tx_xs, tx_ys, rx_xs, rx_ys), axis=1)
        distances = np.zeros([N, N])
        # compute distance between every possible Tx/Rx pair
        for rx_index in range(N):
            for tx_index in range(N):
                tx_coor = layout[tx_index][0:2]
                rx_coor = layout[rx_index][2:4]
                # according to paper notation convention, Hij is from jth transmitter to ith receiver
                distances[rx_index][tx_index] = np.linalg.norm(tx_coor - rx_coor)
        # Check whether a tx-rx link (potentially cross-link) is too close
        if(np.min(distances)<general_para.shortest_crosslink_length):
            print("Created a layout with min tx-rx distance: {}, drop this and re-create Rxs!".format(np.min(distances)))
        else:
            break # go ahead and return the layout
    return layout, distances

# Input: allocs: layouts X N
#        directlink_channel_losses: layouts X N; crosslink_channel_losses: layouts X N X N
# Output: SINRs: layouts X N
def compute_SINRs(general_para, allocs, directlink_channel_losses, crosslink_channel_losses):
    assert np.shape(directlink_channel_losses) == np.shape(allocs), \
        "Mismatch shapes: {} VS {}".format(np.shape(directlink_channel_losses), np.shape(allocs))
    SINRs_numerators = allocs * directlink_channel_losses  # layouts X N
    SINRs_denominators = np.squeeze(np.matmul(crosslink_channel_losses, np.expand_dims(allocs, axis=-1))) + general_para.output_noise_power / general_para.tx_power  # layouts X N
    SINRs = SINRs_numerators / SINRs_denominators  # layouts X N
    return SINRs

# Input: allocs: layouts X N
#        directlink_channel_losses: layouts X N; crosslink_channel_losses: layouts X N X N
# Output: rates: layouts X N
def compute_rates(general_para, allocs, directlink_channel_losses, crosslink_channel_losses):
    SINRs = compute_SINRs(general_para, allocs, directlink_channel_losses, crosslink_channel_losses)
    rates = general_para.bandwidth * np.log2(1 + SINRs/general_para.SNR_gap) # layouts X N
    # Cap at max rate upper limit
    # rates, links_capped = rate_capping(general_para, rates)
    # return rates, links_capped
    return rates

def compute_direct_rates(general_para, directlink_channel_losses):
    # for now, assume constant channel tx power and weight
    SINRS = directlink_channel_losses*general_para.tx_power / general_para.output_noise_power
    rates = general_para.bandwidth * np.log2(1 + SINRS/general_para.SNR_gap)  # layouts X N
    return rates

def get_directlink_channel_losses(channel_losses):
    return np.diagonal(channel_losses, axis1=1, axis2=2)  # layouts X N

def get_crosslink_channel_losses(channel_losses):
    N = np.shape(channel_losses)[-1]
    return channel_losses * ((np.identity(N) < 1).astype(float))

# Add in shadowing into channel losses
def add_shadowing(channel_losses):
    shadow_coefficients = np.random.normal(loc=0, scale=8, size=np.shape(channel_losses))
    channel_losses = channel_losses * np.power(10.0, shadow_coefficients / 10)
    return channel_losses

# Add in fast fading into channel nosses
def add_fast_fading(channel_losses):
    fastfadings = (np.power(np.random.normal(loc=0, scale=1, size=np.shape(channel_losses)), 2) +
                   np.power(np.random.normal(loc=0, scale=1, size=np.shape(channel_losses)), 2)) / 2
    channel_losses = channel_losses * fastfadings
    return channel_losses

# put divided data (in minibatches) into a new dictionary. Not changing the original data_dict in the argument.
def prepare_batches(data_dict, minibatch_size, shuffle=False):
    one_key = list(data_dict.keys())[0]
    number_of_layouts = np.shape(data_dict[one_key])[0]
    assert number_of_layouts >= minibatch_size and (number_of_layouts % minibatch_size == 0), "Set size {}; Minibatch size {}!".format(number_of_layouts, minibatch_size)
    number_of_minibatches = int(number_of_layouts / minibatch_size)
    new_data_dict = {}
    if (shuffle): # prepare the shuffling order
        shuffle_perm = np.arange(number_of_layouts)
        np.random.shuffle(shuffle_perm)
    for key in data_dict.keys():
        data_item = data_dict[key]
        if(shuffle):
            data_item = data_item[shuffle_perm]
        new_data_dict[key] = np.split(data_item, number_of_minibatches) # replace the dictionary content in place
    return new_data_dict, number_of_minibatches

# plot results for a single layout
def plot_allocs_on_layout(ax, layout, plot_vals, plot_title, whether_allocs=True, cluster_assignments=[]):
    N = np.shape(layout)[0]
    assert np.shape(layout) == (N, 4)
    assert len(plot_vals) == N # might be a list of tuples representing colors
    tx_locs = layout[:, 0:2]
    rx_locs = layout[:, 2:4]
    ax.set_title(plot_title)
    ax.scatter(tx_locs[:, 0], tx_locs[:, 1], c='r', label='Tx', s=9)
    ax.scatter(rx_locs[:, 0], rx_locs[:, 1], c='b', label='Rx', s=9)
    for j in range(N):  # plot all activated links
        if(whether_allocs): # plot with 0~1 grayscale value based on allocation results
            ax.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], "{}".format(1 - plot_vals[j].astype(float)))
        else: # the plot_vals should include some color identifiers
            ax.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], linewidth=2.0, c=plot_vals[j])
    ax.legend()
    return
