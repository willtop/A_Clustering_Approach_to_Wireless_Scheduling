# For any reproduce, further research or development, please kindly cite our SPAWC conference paper: 
# @Conference{cluster_schedule, 
#             author = "W. Cui and W. Yu", 
#             title = "A Clustering Approach to Wireless Scheduling", 
#             booktitle = "IEEE Workshop Signal Process. Advances Wireless Commun. (SPAWC)", 
#             year = 2020, 
#             month = may }
#
# This generator only generates layouts and corresponding path losses

import numpy as np
import general_parameters
import utils
import os

general_para = general_parameters.parameters()

def generate_layouts(general_para, number_of_layouts):
    N = general_para.number_of_links
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(
        number_of_layouts, general_para.setting_str))
    layouts = []
    dists = []
    for i in range(number_of_layouts):
        layout, dist = utils.layout_generate(general_para)
        layouts.append(layout)
        dists.append(dist)
        if ((i + 1) % 5000 == 0):
            print("At {}/{} layouts...".format(i + 1, number_of_layouts))
    layouts = np.array(layouts)
    dists = np.array(dists)
    assert np.shape(layouts)==(number_of_layouts, N, 4)
    assert np.shape(dists)==(number_of_layouts, N, N)
    return layouts, dists

# compute path loss components of channel path_losses
# should be used with multiple layouts:
#        distances shape: number of layouts X N X N
def generate_path_losses(general_para, distances):
    N = np.shape(distances)[-1]
    assert N==general_para.number_of_links

    h1 = general_para.tx_height
    h2 = general_para.rx_height
    signal_lambda = 2.998e8 / general_para.carrier_f
    antenna_gain_decibel = general_para.antenna_gain_decibel
    # compute relevant quantity
    Rbp = 4 * h1 * h2 / signal_lambda
    Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * h1 * h2)))
    # compute coefficient matrix for each Tx/Rx pair
    sum_term = 20 * np.log10(distances / Rbp)
    Tx_over_Rx = Lbp + 6 + sum_term + ((distances > Rbp).astype(int)) * sum_term  # adjust for longer path loss
    pathlosses = -Tx_over_Rx + np.eye(N) * antenna_gain_decibel  # only add antenna gain for direct channel
    pathlosses = np.power(10, (pathlosses / 10))  # convert from decibel to absolute
    return pathlosses

def save_generated_files(general_para, store_folder, data_save):
    for key in data_save.keys():
        data = np.array(data_save[key])
        file_name = general_para.file_names[key]
        np.save(store_folder + file_name, data)
    print("[save_generated_files] Saving Completed at {}!".format(store_folder))
    return

# Generate training, validation, and testing samples
# Assume only path loss for channel path_losses
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', help='number of training layouts to generate', type=int, default=0)
    parser.add_argument('--valid', help='number of validation layouts to generate', type=int, default=0)
    parser.add_argument('--test', help='number of testing layouts to generate', type=int, default=0)
    args = parser.parse_args()

    data_save = dict()
    n_train_layouts = args.train
    n_valid_layouts = args.valid
    n_test_layouts = args.test
    n_layouts = n_train_layouts + n_valid_layouts + n_test_layouts
    if(n_layouts==0):
        print("Nothing to do for generating 0 layouts. Exiting...")
        exit(0)
    else:
        print("Going to generate {} training layouts; {} validation layouts; {} testing layouts.".format(
            n_train_layouts, n_valid_layouts, n_test_layouts    ))

    layouts, dists = generate_layouts(general_para, n_layouts)
    pathlosses = generate_path_losses(general_para, dists)
    if(n_train_layouts>0):
        print("Saving training data...")
        data_save["layouts"] = layouts[:n_train_layouts]
        data_save["path_losses"] = pathlosses[:n_train_layouts]
        save_generated_files(general_para, general_para.train_dir, data_save)
        print("Stats of train path_losses: mean: {}; std: {}".format(np.mean(data_save["path_losses"]), np.std(data_save["path_losses"])))
        # save per-dimension stats among training set for CSI input normalization
        input_normalization_mean = np.mean(np.reshape(data_save['path_losses'], [-1, general_para.number_of_links ** 2]), axis=0)
        input_normalization_std = np.std(np.reshape(data_save['path_losses'], [-1, general_para.number_of_links ** 2]), axis=0)
        np.save("../Fully_Connected_Models/Normalization_Stats/"+general_para.file_names["CSI_mean_stats"], input_normalization_mean)
        np.save("../Fully_Connected_Models/Normalization_Stats/"+general_para.file_names["CSI_std_stats"], input_normalization_std)
        print("Computed and Saved input normalization stats from the training dataset for FC neural net!")
    if(n_valid_layouts>0):
        print("Saving validation data...")
        data_save["layouts"] = layouts[n_train_layouts: n_train_layouts+n_valid_layouts]
        data_save["path_losses"] = pathlosses[n_train_layouts: n_train_layouts+n_valid_layouts]
        save_generated_files(general_para, general_para.valid_dir, data_save)
        print("Stats of valid path_losses: mean: {}; std: {}".format(np.mean(data_save["path_losses"]), np.std(data_save["path_losses"])))
    if(n_test_layouts > 0):
        print("Saving testing data...")
        data_save["layouts"] = layouts[n_train_layouts + n_valid_layouts: ]
        data_save["path_losses"] = pathlosses[n_train_layouts + n_valid_layouts: ]
        save_generated_files(general_para, general_para.test_dir, data_save)
        print("Stats of test path_losses: mean: {}; std: {}".format(np.mean(data_save["path_losses"]), np.std(data_save["path_losses"])))

    print("Generator Function Completed Successfully!")
