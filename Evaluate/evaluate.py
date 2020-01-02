# Script for evaluating
# all 5pert suffix means the 5-percentile per link rate

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../Clustering_Methods/")
sys.path.append("../Utilities_Research/")
import benchmarks
import general_parameters
import utils
import proportional_fairness_scheduling
import Spectral_Clustering
import Hierarchical_Clustering
import K_Means

debug_visualize = False # visualize allocations and rates of each method

def compute_avg_ratio(result_dict, title):
    print("[Avg of ratios for {}]".format(title))
    assert "FP Scheduling" in result_dict.keys(), "Didn't include FP Scheduling in computation"
    number_of_layouts = np.size(result_dict["FP Scheduling"])
    print("FP Scheduling mean value: {}".format(np.mean(result_dict["FP Scheduling"])))
    for method_key in result_dict.keys():
        assert np.shape(result_dict[method_key])==(number_of_layouts, ), "Wrong shape for method {}: {}".format(method_key, np.shape(result_dict[method_key]))
        if (method_key == "FP Scheduling"):
            continue
        ratios = result_dict[method_key] / result_dict["FP Scheduling"] * 100
        print("[{}]: avg {}% of {};".format(method_key, round(np.mean(ratios), 2), "FP Scheduling"), end="")
    print("\n")
    return

def obtain_line_colors(n_colors): # return colors differentiating among different methods
    colormap = plt.get_cmap('gist_rainbow')
    line_colors = [colormap(1. * i / n_colors) for i in range(n_colors)]
    return line_colors

def plot_rates_CDF(general_para, rate_results, task_title):
    fig = plt.figure()
    ax = fig.gca()
    plt.title("[{}] {}".format(task_title, general_para.setting_str))
    plt.xlabel("Mbps")
    plt.ylabel("Cumulative Distribution of D2D Networks")
    plt.grid(linestyle="dotted")
    ax.set_ylim(bottom=0)
    line_colors = obtain_line_colors(len(rate_results.keys()))
    for method_index, method_key in enumerate(rate_results.keys()):
        rates = np.sort(rate_results[method_key])
        plt.plot(rates / 1e6, np.arange(1, np.size(rates) + 1) / np.size(rates), label="{}".format(method_key), color=line_colors[method_index])
        plt.legend()
    plt.show()
    return

if(__name__ =='__main__'):
    general_para = general_parameters.parameters()
    N = general_para.number_of_links
    layouts = np.load(general_para.test_dir + general_para.file_names["layouts"])
    gains = np.load(general_para.test_dir + general_para.file_names["gains"])
    gains_diagonal = utils.get_diagonal_gains(gains)
    gains_nondiagonal = utils.get_nondiagonal_gains(gains)
    number_of_layouts = np.shape(gains)[0]
    print("Evaluate {} over {} layouts".format(general_para.setting_str, number_of_layouts))

    all_allocs = {}
    all_cluster_assignments = {}

    # Integer results
    all_allocs["FP Scheduling"] = benchmarks.FP(general_para, gains, np.ones([number_of_layouts, N]), scheduling_output=True)
    n_links_on_FP = np.sum(all_allocs["FP Scheduling"],axis=1)
    print("From FP scheduling, average number of links activated per layout: {}/{} links".format(np.mean(n_links_on_FP), N))

    print("Spectral Clustering...")
    cluster_assignments = []
    allocs = []
    for i in range(number_of_layouts):
        if ((i + 1) * 100 / number_of_layouts % 50 == 0):
            print("At {}/{} layouts.".format(i + 1, number_of_layouts))
        clusters_one_layout = Spectral_Clustering.clustering(layouts[i], gains[i], n_links_on_FP[i])
        allocs_one_layout = Spectral_Clustering.scheduling(gains[i], clusters_one_layout)
        cluster_assignments.append(clusters_one_layout)
        allocs.append(allocs_one_layout)
    cluster_assignments = np.array(cluster_assignments)
    allocs = np.array(allocs)
    assert np.shape(cluster_assignments) == np.shape(allocs) == (number_of_layouts, N)
    all_allocs["Spectral Clustering"] = allocs
    all_cluster_assignments["Spectral Clustering"] = cluster_assignments

    print("Hierarchical Clustering...")
    cluster_assignments = []
    allocs = []
    for i in range(number_of_layouts):
        if ((i + 1) * 100 / number_of_layouts % 50 == 0):
            print("At {}/{} layouts.".format(i + 1, number_of_layouts))
        clusters_one_layout = Hierarchical_Clustering.clustering(layouts[i], gains[i], n_links_on_FP[i])
        allocs_one_layout = Hierarchical_Clustering.scheduling(gains[i], clusters_one_layout)
        cluster_assignments.append(clusters_one_layout)
        allocs.append(allocs_one_layout)
    cluster_assignments = np.array(cluster_assignments)
    allocs = np.array(allocs)
    assert np.shape(cluster_assignments) == np.shape(allocs) == (number_of_layouts, N)
    all_allocs["Hierarchical Clustering"] = allocs
    all_cluster_assignments["Hierarchical Clustering"] = cluster_assignments

    print("K-Means...")
    cluster_assignments = []
    allocs = []
    for i in range(number_of_layouts):
        if ((i + 1) * 100 / number_of_layouts % 50 == 0):
            print("At {}/{} layouts.".format(i + 1, number_of_layouts))
        clusters_one_layout = K_Means.clustering(layouts[i], n_links_on_FP[i])
        allocs_one_layout = K_Means.scheduling(layouts[i], clusters_one_layout)
        cluster_assignments.append(clusters_one_layout)
        allocs.append(allocs_one_layout)
    cluster_assignments = np.array(cluster_assignments)
    allocs = np.array(allocs)
    assert np.shape(cluster_assignments) == np.shape(allocs) == (number_of_layouts, N)
    all_allocs["K-Means"] = allocs
    all_cluster_assignments["K-Means"] = cluster_assignments

    all_allocs["Greedy Scheduling"] = benchmarks.greedy_scheduling(general_para, gains_diagonal, gains_nondiagonal, np.ones([number_of_layouts, N]))
    all_allocs["All Active"] = np.ones([number_of_layouts, N]).astype(float)
    all_allocs["Random Scheduling"] = np.random.randint(2, size=[number_of_layouts, N]).astype(float) # Return float type
    all_allocs["Strongest Link"] = benchmarks.Strongest_Link(general_para, gains_diagonal)


    # EVALUATION AND COMPARISON
    all_allocs_means = {}
    for method_key in all_allocs.keys():
        assert np.shape(all_allocs[method_key]) == (number_of_layouts, N) # checking dimension validity
        all_allocs_means[method_key] = np.mean(all_allocs[method_key],axis=1)
    compute_avg_ratio(all_allocs_means, "Scheduling mean")

    # Evaluate Stage I: single time slot
    links_rates_I = {}
    evaluate_results_I = {}
    for method_key in all_allocs.keys():
        links_rates_I[method_key] = utils.compute_rates(general_para, all_allocs[method_key], gains_diagonal, gains_nondiagonal)
        assert np.shape(links_rates_I[method_key]) == (number_of_layouts, N)
        evaluate_results_I[method_key] = np.sum(links_rates_I[method_key], axis=1)
    compute_avg_ratio(evaluate_results_I, "Sum Rate Single Timeslot")
    plot_rates_CDF(general_para, evaluate_results_I, "Sum Rate Single Timeslot")

    # visualize allocations for worst and best layouts, 3 each
    if(debug_visualize):
        for method_key in all_allocs.keys():
            if(method_key in ["All Active", "Random Power Control", "Random Scheduling", "Strongest Link", "Directlinks Inverse Proportions"]):
                continue # Don't plot for these trivial allocations
            fig, axs = plt.subplots(nrows=2,ncols=3)
            fig.suptitle("{} allocs for Sum Rate Single Timeslot".format(method_key))
            layout_indices_ranked = np.argsort(evaluate_results_I[method_key])
            rank_titles = {0: "Worst",   1: "2nd Worst", 2: "3rd Worst",  -1: "Best", -2: "2nd Best",  -3: "3rd Best"}
            for i, rank_tuple in enumerate(rank_titles.items()):
                v_layout_index = layout_indices_ranked[rank_tuple[0]]
                v_alloc = all_allocs[method_key][v_layout_index]
                layout = layouts[v_layout_index]
                utils.plot_allocs_on_layout(axs.flatten()[i], layout, v_alloc, rank_tuple[1])
            plt.show()

    # compare allocations between neural network or spectral clustering and the corresponding best benchmark
    layouts_to_visualize = np.random.randint(low=0, high=number_of_layouts, size=3)
    for layout_to_visualize in layouts_to_visualize:
        plt.title("Layout #{}".format(layout_to_visualize))
        ax = plt.subplot(231)
        utils.plot_allocs_on_layout(ax, layouts[layout_to_visualize], all_allocs["FP Scheduling"][layout_to_visualize], "FP Scheduling")
        ax = plt.subplot(232)
        utils.plot_allocs_on_layout(ax, layouts[layout_to_visualize], all_allocs['Spectral Clustering'][layout_to_visualize], "Spectral Clustering")
        ax = plt.subplot(233)
        utils.plot_allocs_on_layout(ax, layouts[layout_to_visualize], all_allocs['K-Means'][layout_to_visualize], "K-Means")
        ax = plt.subplot(212)
        ax.plot(all_allocs["FP Scheduling"][layout_to_visualize], 'b', label="FP Scheduling")
        ax.plot(all_allocs['Spectral Clustering'][layout_to_visualize], 'r--', linewidth=1.2, label='Spectral Clustering')
        ax.plot(all_allocs['K-Means'][layout_to_visualize], 'g--', linewidth=1.2, label='K-Means')
        ax.legend()
        plt.show()

    # Evaluate Stage II: multiple time slots
    all_allocs_prop_fair = {}
    links_rates_prop_fair = {}
    print("Evaluating log long term avg rate results...")

    all_allocs_prop_fair["FP Scheduling"], links_rates_prop_fair["FP Scheduling"] = proportional_fairness_scheduling.FP_prop_fair(general_para, gains, gains_diagonal, gains_nondiagonal)
    all_allocs_prop_fair["K-Means"], links_rates_prop_fair["K-Means"] = proportional_fairness_scheduling.K_Means_prop_fair(general_para, gains_diagonal, gains_nondiagonal, all_cluster_assignments["K-Means"])
    all_allocs_prop_fair["Spectral Clustering"], links_rates_prop_fair["Spectral Clustering"] = proportional_fairness_scheduling.Spectral_Clustering_prop_fair(general_para, gains_diagonal, gains_nondiagonal, all_cluster_assignments["Spectral Clustering"])
    all_allocs_prop_fair["Hierarchical Clustering"], links_rates_prop_fair["Hierarchical Clustering"] = proportional_fairness_scheduling.Hierarchical_Clustering_prop_fair(general_para, gains_diagonal, gains_nondiagonal, all_cluster_assignments["Hierarchical Clustering"])
    all_allocs_prop_fair["Greedy Scheduling"], links_rates_prop_fair["Greedy Scheduling"] = proportional_fairness_scheduling.Greedy_Scheduling_prop_fair(general_para, gains_diagonal, gains_nondiagonal)
    all_allocs_prop_fair["All Active"], links_rates_prop_fair["All Active"] = proportional_fairness_scheduling.all_active_prop_fair(general_para, gains_diagonal, gains_nondiagonal)
    all_allocs_prop_fair["Random Scheduling"], links_rates_prop_fair["Random Scheduling"] = proportional_fairness_scheduling.random_scheduling_prop_fair(general_para, gains_diagonal, gains_nondiagonal)
    # Simplest round robin (one at a time)
    all_allocs_prop_fair["Vanilla Round Robin"], links_rates_prop_fair["Vanilla Round Robin"] = proportional_fairness_scheduling.vanilla_round_robin_prop_fair(general_para, gains_diagonal, gains_nondiagonal)

    # Compute sum log avg rate utility
    links_avg_rates_II = {}
    print("[Layouts Avg for sum log mean rate (Mbps)]:")
    for method_key, rates in links_rates_prop_fair.items():
        assert np.shape(rates) == (number_of_layouts, general_para.log_utility_time_slots, N)
        links_avg_rates_II[method_key] = np.mean(rates, axis=1) # number of layouts X N
        evaluate_result = np.sum(np.log(links_avg_rates_II[method_key]/1e6), axis=1) # number of layouts
        links_avg_rates_II[method_key] = links_avg_rates_II[method_key].flatten() # flatten for plotting
        print("[{}]:{};".format(method_key, round(np.mean(evaluate_result),2)), end="")
    print("\n")
    plot_rates_CDF(general_para, links_avg_rates_II, "Log Utilities Multiple Timeslots")

    print("Script Completed Successfully!")