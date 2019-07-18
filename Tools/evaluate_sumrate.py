# Script for evaluating
# all 5pert suffix means the 5-percentile per link rate

import numpy as np
# to silent tensorflow WARNING
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt
import benchmarks
import general_parameters
import utils
import FPLinQ
import data_generator
import sys
sys.path.append("../Models/")
import FastFading_Injection_Net
sys.path.append("../SpectralClustering/")
import Spectral_Clustering

investigation_topic = "sumrate"
investigation_topic = "sumrate_robust"
#investigation_topic = "conservative_scheduling_sumrate_robust"

only_pathlosses = True

evaluate_methods = ["FP",
                    "NeuralNet",
                    "AllActive",
                    "Random",
                    "Random_Scheduling",
                    "StrongestLink",
                    "FP_sch_sqrtscale",
                    "FP_sch_thresholds",
                    #"Spectral_Clustering"
                    ]

baseline_method = "FP" # For power control task evaluations
#baseline_method = "FP Binary" # For scheduling task evaluations

evaluate_metrics = ["allocs_stats",
                    "nominal_sumrate",
                    "visualize_allocs",
                    "compare_allocs",
                    "fastfade_5pert"
                    ]

if(not only_pathlosses):
    assert "fastfade_5pert" not in evaluate_metrics, "Shouldn't evaluate with 5 percentile when original CSI already has fast fading!"

def compute_avg_ratio(result_dict, title):
    print("[Avg of ratios for {}]".format(title))
    assert baseline_method in result_dict.keys(), "Didn't include {} in computation".format(baseline_method)
    number_of_layouts = np.size(result_dict[baseline_method])
    print("Baseline {} mean value: {}".format(baseline_method, np.mean(result_dict[baseline_method])))
    for method_key in result_dict.keys():
        assert np.shape(result_dict[method_key])==(number_of_layouts, )
        if (method_key == baseline_method):
            continue
        ratios = result_dict[method_key] / result_dict[baseline_method] * 100
        print("[{}]: avg {}% of {};".format(method_key, round(np.mean(ratios), 2), baseline_method), end="")
    print("\n")
    return


if(__name__ =='__main__'):
    general_para = general_parameters.parameters()
    N = general_para.number_of_links
    layouts = np.load(general_para.test_dir + general_para.file_names["layouts"])
    gains = np.load(general_para.test_dir + general_para.file_names["gains"])
    if(not only_pathlosses):
        print("Adding fast fading into CSI at the beginning!")
        gains = data_generator.add_fast_fadings(gains)
    gains_diagonal = utils.get_diagonal_gains(gains)
    gains_nondiagonal = utils.get_nondiagonal_gains(gains)
    number_of_layouts = np.shape(gains)[0]
    all_allocs = dict()

    print("Evaluate settings: {} over {} layouts".format(general_para.setting_str, number_of_layouts))

    if("FP" in evaluate_methods):
        all_allocs["FP"] = benchmarks.FP(general_para, gains)

    if("NeuralNet" in evaluate_methods and investigation_topic=="general"):
        if(general_para.inputs_info == "CSI"):
            print("==============================Neural Network Model with CSI inputs==================================")
            inputs_for_neural_net = np.reshape(gains, [-1, N**2])
        else:
            print("==============================Neural Network Model with GLI inputs==================================")
            inputs_for_neural_net = np.reshape(layouts, [-1, 4*N])
        network_model = FastFading_Injection_Net.Network()
        TFgraph, inputs_placeholder, gains_diagonal_placeholder, gains_nondiagonal_placeholder, \
        whether_train_placeholder, power_controls, train_step, cost  = network_model.build_network()
        model_loc = network_model.model_filename
        print("Evaluating model loaded from: ", model_loc)
        with TFgraph.as_default():
            save_saver = tf.train.Saver()
            with tf.Session() as sess:
                save_saver.restore(sess, model_loc)
                test_dict = {inputs_placeholder: inputs_for_neural_net,
                    gains_diagonal_placeholder: gains_diagonal,
                    gains_nondiagonal_placeholder: gains_nondiagonal,
                    whether_train_placeholder: False}
                allocs = sess.run(power_controls, feed_dict=test_dict)
        all_allocs["Deep Learning"] = allocs

    if("AllActive" in evaluate_methods and investigation_topic=="general"):
        all_allocs["All Active"] = benchmarks.All_Active(general_para, number_of_layouts)

    if("Random" in evaluate_methods and investigation_topic=="general"):
        all_allocs["Random"] = benchmarks.Random(general_para, number_of_layouts)

    if("Random_Scheduling" in evaluate_methods and investigation_topic=="general"):
        all_allocs["Random Scheduling"] = benchmarks.Random_Scheduling(general_para, number_of_layouts)

    # Binary activation: only activating the strongest link, at full power
    if("StrongestLink" in evaluate_methods and investigation_topic=="general"):
        all_allocs["Strongest Link"] = benchmarks.Strongest_Link(general_para, gains_diagonal)
    
    # Binary activation: correct FP scheduling quantization
    if ("FP_sch_sqrtscale" in evaluate_methods):
        print("==================================FP scheduling with 0.5 thresholding in sqrt scale====================================")
        all_allocs["FP Binary"] = (np.sqrt(all_allocs["FP"]) > 0.5).astype(float)  # use float for scheduling plotting, shouldn't affect numerically

    # Binary activation: crude FP quantization
    if ("FP_sch_thresholds" in evaluate_methods and investigation_topic=="general"):
        sch_threshold = 0.7
        print("==================================FP scheduling with thresholding {}====================================".format(sch_threshold))
        all_allocs["Conservative FP Binary {} threshold".format(sch_threshold)] = (all_allocs["FP"] > sch_threshold).astype(float) # use float for scheduling plotting, shouldn't affect numerically

    if (investigation_topic=="conservative_scheduling"):
        for val in np.arange(0.1,0.99,0.1):
            all_allocs["FP_sch_threshold_{}".format(round(val,1))] = (all_allocs["FP"] > val).astype(float)


    # spectral clustering scheduling
    if ("Spectral_Clustering" in evaluate_methods and investigation_topic=="general"):
        print("==================================Spectral clustering scheduling=================================")
        allocs = []
        for i in range(number_of_layouts):
            if((i+1)*100/number_of_layouts % 10 ==0):
                print("At {}/{} layouts.".format(i+1, number_of_layouts))
            allocs_per_layout = Spectral_Clustering.schedule(general_para, layouts[i], gains[i])
            allocs.append(allocs_per_layout)
        allocs = np.array(allocs)
        assert np.shape(allocs) == (number_of_layouts, N)
        all_allocs["Spectral Clustering"] = allocs


    print("##########################EVALUATION AND COMPARISON########################")
    if("allocs_stats" in evaluate_metrics):
        all_allocs_means = dict()
        for method_key in all_allocs.keys():
            allocs = all_allocs[method_key]
            assert np.shape(allocs) == (number_of_layouts, N)
            all_allocs_means[method_key] = np.mean(allocs,axis=1)
        compute_avg_ratio(all_allocs_means, "power allocations mean")
       
    all_sumrates_dicts = {}
    if("nominal_sumrate" in evaluate_metrics):
        all_nominal_sumrates = dict()
        for method_key in all_allocs.keys():
            allocs = all_allocs[method_key]
            assert np.shape(allocs) == (number_of_layouts, N)
            all_links_rates = utils.compute_rates(general_para, allocs, gains_diagonal, gains_nondiagonal)
            sumrates = np.sum(all_links_rates,axis=1)
            assert np.shape(sumrates) == (number_of_layouts,)
            all_nominal_sumrates[method_key] = sumrates
        compute_avg_ratio(all_nominal_sumrates, "nominal sumrates (with pathlosses)")
        all_sumrates_dicts["nominal"] = all_nominal_sumrates

    # visualize allocations for worst and best layouts, 3 each
    if("visualize_allocs" in evaluate_metrics):
        x_range = general_para.field_length; y_range = x_range
        for method_key in all_allocs.keys():
            if(method_key in ["All Active", "Random", "Random Scheduling"]):
                continue # Don't plot for these trivial allocations
            fig, axs = plt.subplots(nrows=2,ncols=3)
            fig.suptitle("{} allocs".format(method_key))
            layout_indices_ranked = np.argsort(all_nominal_sumrates[method_key])
            rank_titles = {0: "Worst",
                           1: "2nd Worst",
                           2: "3rd Worst",
                           -1: "Best",
                           -2: "2nd Best",
                           -3: "3rd Best"}
            for i, rank_tuple in enumerate(rank_titles.items()):
                v_layout_index = layout_indices_ranked[rank_tuple[0]]
                v_alloc = all_allocs[method_key][v_layout_index]
                layout = layouts[v_layout_index]
                utils.plot_allocs_on_layout(axs.flatten()[i], layout, v_alloc, rank_tuple[1])
            plt.show()

    # compare allocations between FP and neural network
    if("compare_allocs" in evaluate_metrics):
        assert "Deep Learning" in all_allocs.keys(), "Can't compare allocations with FP without collecting neural net's allocations!"
        layouts_indices = np.random.randint(low=0,high=number_of_layouts,size=3)
        for layout_index in layouts_indices:
            plt.title("Layout #{}".format(layout_index))
            # plot for FP allocation
            ax = plt.subplot(221)
            utils.plot_allocs_on_layout(ax, layouts[layout_index], all_allocs['FP'][layout_index], "FP")
            # plot for neural net allocation
            ax = plt.subplot(222)
            utils.plot_allocs_on_layout(ax, layouts[layout_index], all_allocs['Deep Learning'][layout_index], "Deep Learning")
            # plot allocations comparison
            ax = plt.subplot(212)
            ax.plot(all_allocs['FP'][layout_index], 'b', label='FP')
            ax.plot(all_allocs['Deep Learning'][layout_index], 'r--', linewidth=1.2, label='Deep Learning')
            ax.legend()
            plt.show()


    # TEST ON GAINS WITH FAST FADING
    if("fastfade_5pert" in evaluate_metrics):
        ff_realizations = 500
        gains_ff_tmp = np.tile(gains, reps=[ff_realizations, 1, 1])
        gains_ff = data_generator.add_fast_fadings(gains_ff_tmp)
        gains_ff_diag = utils.get_diagonal_gains(gains_ff)
        gains_ff_nondiag = utils.get_nondiagonal_gains(gains_ff)
        all_sum_5pert_ff_rates = dict()
        highest_mean_sum_5pert_ff_rates = 0
        best_robust_method_key = ""
        for method_key in all_allocs.keys():
            allocs = np.tile(all_allocs[method_key], reps=[ff_realizations, 1])
            rates_ff = utils.compute_rates(general_para, allocs, gains_ff_diag, gains_ff_nondiag)
            rates_ff = np.reshape(rates_ff, [ff_realizations, number_of_layouts, N])
            rates_ff_5pert = np.percentile(rates_ff, q=5, axis=0, interpolation="lower")
            sum_5pert_ff_rates = np.sum(rates_ff_5pert, axis=1)
            assert np.shape(sum_5pert_ff_rates) == (number_of_layouts, )
            all_sum_5pert_ff_rates[method_key] = sum_5pert_ff_rates
            if(np.mean(sum_5pert_ff_rates)>highest_mean_sum_5pert_ff_rates):
                best_robust_method_key = method_key
                highest_mean_sum_5pert_ff_rates = np.mean(sum_5pert_ff_rates)
        compute_avg_ratio(all_sum_5pert_ff_rates, "sum of 5-percentile rate per link with fastfadings")
        all_sumrates_dicts["5pert_link"] = all_sum_5pert_ff_rates

    # Plot sumrates CDF curves
    fig = plt.figure()
    ax = fig.gca()
    colormap = plt.get_cmap('gist_rainbow')
    n_methods = len(all_allocs.keys())
    ax.set_prop_cycle(color=[colormap(1.*i/n_methods) for i in range(n_methods)])
    plottitle = "{} links {}mX{}m pairdists {}~{}m max Tx power {}dBm".format(
               N, general_para.field_length, general_para.field_length, general_para.shortest_dist, general_para.longest_dist, general_para.tx_power_milli_decibel) 
    if("fastfade_5pert" in evaluate_metrics):
        plottitle += " Most robust method: {}".format(best_robust_method_key)
    plt.title(plottitle)
    plt.xlabel("Sum Rates (Mbps)")
    plt.ylabel("Cumulative Distribution of D2D Networks")
    plt.grid(linestyle="dotted")
    ax.set_ylim(bottom=0)
    for key, sumrates_dict in all_sumrates_dicts.items():
        if(key=="nominal"):
            ls = '-'
            if(only_pathlosses):
                legend_suffix = '_nominal'
            else:
                legend_suffix = ""
        else:
            ls = '--'
            legend_suffix = '_robust'
        for method_key in sumrates_dict.keys():
            if(baseline_method == "FP Binary"):
                if(method_key == "FP"):
                    continue  # don't plot FP if simulating the scheduling task
            sum_rates = np.sort(sumrates_dict[method_key])
            plt.plot(sum_rates, np.arange(1, number_of_layouts + 1) / number_of_layouts, label="{}{}".format(method_key, legend_suffix), linestyle=ls)
        plt.legend()
    plt.show()

    print("Script Completed Successfully!")
