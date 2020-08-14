# For any reproduce, further research or development, please kindly cite our SPAWC conference paper: 
# @Conference{cluster_schedule, 
#             author = "W. Cui and W. Yu", 
#             title = "A Clustering Approach to Wireless Scheduling", 
#             booktitle = "IEEE Workshop Signal Process. Advances Wireless Commun. (SPAWC)", 
#             year = 2020, 
#             month = may }
#
# Storing all general parameters about environmental settings

import numpy as np

class parameters():
    def __init__(self):
        # general dataset setting
        self.number_of_links = 20
        self.base_dir = "../" # This file should always be placed as the first level subfolder
        # FPLinQ setting
        self.FP_iter_amount = 100
        self.quantize_levels = 2  # power control setting
        self.alpha_proportional_fairness_update = 0.95  # for weights updates
        # specific channel setting
        self.field_length = 1000
        self.shortest_directlink_length = 30
        self.longest_directlink_length = 50
        self.shortest_crosslink_length = 0.5 # can't exist a tx-rx pair (potentially from different links) closer than this distance
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5
        self.tx_power_milli_decibel = 40
        self.tx_power = np.power(10, (self.tx_power_milli_decibel-30)/10)
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        self.SNR_gap_dB = 6
        self.SNR_gap = np.power(10, self.SNR_gap_dB/10)
        # testing setting
        self.log_utility_time_slots = 200 # number of time slots for computing the average rates
        # setting descriptor
        self.setting_str = "{}_links_{}X{}_pairdists_{}_{}".format(self.number_of_links, self.field_length, self.field_length, self.shortest_directlink_length, self.longest_directlink_length)
        # for files saved after generation
        self.data_dir = self.base_dir+"Data/"
        self.train_dir = self.data_dir+"Train/"
        self.valid_dir = self.data_dir+"Valid/"
        self.test_dir = self.data_dir+"Test/"
        self.file_names = {"layouts": "layouts_{}.npy".format(self.setting_str),
            "path_losses": "path_losses_{}.npy".format(self.setting_str),
            "CSI_mean_stats": "CSI_mean_stats_{}.npy".format(self.setting_str),
            "CSI_std_stats": "CSI_std_stats_{}.npy".format(self.setting_str),
            "trained_Conv_model": "Conv_model.ckpt".format(self.setting_str)
        }
