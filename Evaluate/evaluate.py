# For any reproduce, further research or development, please kindly cite our SPAWC conference paper: 
# @Conference{cluster_schedule, 
#             author = "W. Cui and W. Yu", 
#             title = "A Clustering Approach to Wireless Scheduling", 
#             booktitle = "IEEE Workshop Signal Process. Advances Wireless Commun. (SPAWC)", 
#             year = 2020, 
#             month = may }
#
# Script for evaluating
# all 5pert suffix means the 5-percentile per link rate

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../Clustering_Methods/")
sys.path.append("../Utilities/")
import benchmarks
import general_parameters
import utils
from clustering_scheduling_caller import clustering_and_scheduling
import proportional_fairness_scheduling


debug_visualize = False # visualize allocations and rates of each method

def compute_avg_ratio(result_dict, title):
    print("[Avg of ratios for {}]".format(title))
    assert "FP Scheduling" in result_dict.keys(), "Didn't include FP Scheduling in computation"
    n_layouts = np.size(result_dict["FP Scheduling"])
    print("FP Scheduling mean value: {}".format(np.mean(result_dict["FP Scheduling"])))
    for clustering_method in result_dict.keys():
        assert np.shape(result_dict[clustering_method])==(n_layouts, ), "Wrong shape for method {}: {}".format(clustering_method, np.shape(result_dict[clustering_method]))
        if (clustering_method == "FP Scheduling"):
            continue
        ratios = result_dict[clustering_method] / result_dict["FP Scheduling"] * 100
        print("[{}]: avg {}% of {};".format(clustering_method, round(np.mean(ratios), 2), "FP Scheduling"), end="")
    print("\n")
    return

def obtain_line_colors(n_colors): # return colors differentiating among different methods
    colormap = plt.get_cmap('gist_rainbow')
    line_colors = [colormap(1. * i / n_colors) for i in range(n_colors)]
    return line_colors

def plot_meanrates_CDF(general_para, rate_results):
    fig = plt.figure()
    ax = fig.gca()
    plt.title("Mean-Rates over Multiple Time-Slots of each D2D Link among All Networks", fontsize=20)
    plt.xlabel("Mbps", fontsize=20)
    plt.ylabel("Cumulative Distribution of all D2D Links", fontsize=20)
    plt.grid(linestyle="dotted")
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=np.percentile(rate_results["FP Scheduling"]/1e6, 90))
    line_colors = obtain_line_colors(len(rate_results.keys()))
    line_styles = ['-', ':', '-.', '--']
    for method_index, scheduling_method in enumerate(rate_results.keys()):
        rates = np.sort(rate_results[scheduling_method])
        plt.plot(rates / 1e6, np.arange(1, np.size(rates) + 1) / np.size(rates), label="{}".format(scheduling_method), color=line_colors[method_index], linestyle=line_styles[method_index % 4], linewidth=1.2)
        plt.legend(prop={'size': 18})
    plt.show()
    return

if(__name__ =='__main__'):
    general_para = general_parameters.parameters()
    N = general_para.number_of_links
    layouts = np.load(general_para.test_dir + general_para.file_names["layouts"])
    path_losses = np.load(general_para.test_dir + general_para.file_names["path_losses"])
    n_layouts = np.shape(layouts)[0]
    assert np.shape(layouts) == (n_layouts, N, 4) and np.shape(path_losses) == (n_layouts, N, N)
    all_clustering_methods = ["Spectral Clustering", "Hierarchical Clustering", "Hierarchical Clustering EqualSize", "K-Means", "K-Means EqualSize"]
    print("Evaluate {} over {} layouts".format(general_para.setting_str, n_layouts))

    for channel_model in ["Pure Path Losses"]:
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<Evaluating {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>".format(channel_model))
        if channel_model == "Pure Path Losses":
            channel_losses = path_losses
        elif channel_model == "With Fading":
            channel_losses = utils.add_fast_fading(path_losses)
        elif channel_model == "With Shadowing & Fading":
            channel_losses = utils.add_shadowing(path_losses)
            channel_losses = utils.add_fast_fading(channel_losses)
        else:
            print("Shouldn't be here!")
            exit(1)
        directlink_channel_losses = utils.get_directlink_channel_losses(channel_losses)
        crosslink_channel_losses = utils.get_crosslink_channel_losses(channel_losses)
        all_allocs = {}
        all_clusters = {}

        # Integer results
        all_allocs["FP Scheduling"] = benchmarks.FP(general_para, channel_losses, np.ones([n_layouts, N]), scheduling_output=True)
        n_links_on_FP_sumrate = np.sum(all_allocs["FP Scheduling"],axis=1)
        print("FP Sumrate activates {}/{} links on avg per layout.".format(np.mean(n_links_on_FP_sumrate), N))

        for clustering_method in all_clustering_methods:
            print("{} Sum Rate...".format(clustering_method))
            allocs = []
            clusters = []
            for i in range(n_layouts):
                clusters_one_layout, allocs_one_layout = clustering_and_scheduling(layouts[i], channel_losses[i], n_links_on_FP_sumrate[i], clustering_method)
                clusters.append(clusters_one_layout)
                allocs.append(allocs_one_layout)
            clusters = np.array(clusters)
            allocs = np.array(allocs)
            assert np.shape(clusters) == np.shape(allocs) == (n_layouts, N)
            all_clusters[clustering_method] = clusters
            all_allocs[clustering_method] = allocs

        all_allocs["Greedy Scheduling"] = benchmarks.greedy_scheduling(general_para, directlink_channel_losses, crosslink_channel_losses, np.ones([n_layouts, N]))
        all_allocs["All Active"] = np.ones([n_layouts, N]).astype(float)
        all_allocs["Random Scheduling"] = np.random.randint(2, size=[n_layouts, N]).astype(float) # Return float type
        all_allocs["Strongest Link"] = benchmarks.Strongest_Link(general_para, directlink_channel_losses)


        # EVALUATION AND COMPARISON
        all_allocs_means = {}
        for clustering_method in all_allocs.keys():
            assert np.shape(all_allocs[clustering_method]) == (n_layouts, N) # checking dimension validity
            all_allocs_means[clustering_method] = np.mean(all_allocs[clustering_method],axis=1)
        compute_avg_ratio(all_allocs_means, "Scheduling Mean")

        # Evaluate Stage I: single time slot
        links_rates_I = {}
        evaluate_results_I = {}
        for clustering_method in all_allocs.keys():
            links_rates_I[clustering_method] = utils.compute_rates(general_para, all_allocs[clustering_method], directlink_channel_losses, crosslink_channel_losses)
            assert np.shape(links_rates_I[clustering_method]) == (n_layouts, N)
            evaluate_results_I[clustering_method] = np.sum(links_rates_I[clustering_method], axis=1)
        compute_avg_ratio(evaluate_results_I, "Sum Rate Single Timeslot")

        # visualize clustering based methods allocations for worst and best layouts, 3 each
        if (debug_visualize):
            for clustering_method in all_clustering_methods:
                fig, axs = plt.subplots(nrows=2, ncols=3)
                fig.suptitle("{} clustering and scheduling for Sum-Rate".format(clustering_method))
                layout_indices_ranked = np.argsort(evaluate_results_I[clustering_method])
                rank_titles = {0: "Worst", 1: "2nd Worst", 2: "3rd Worst", -1: "Best", -2: "2nd Best", -3: "3rd Best"}
                for i, rank_tuple in enumerate(rank_titles.items()):
                    layout_index = layout_indices_ranked[rank_tuple[0]]
                    layout = layouts[layout_index]
                    clusters = all_clusters[clustering_method][layout_index]
                    allocs = all_allocs[clustering_method][layout_index]
                    ax = axs.flatten()[i]
                    ax.set_title(rank_tuple[1])
                    utils.plot_stations_on_layout(ax, layout)
                    utils.plot_clusters_on_layout(ax, layout, clusters)
                    utils.plot_schedules_on_layout(ax, layout, allocs)
                plt.show()

        # Evaluate Stage II: multiple time slots
        all_allocs_prop_fair = {}
        links_rates_prop_fair = {}
        print("Evaluating log long-term avg rate results...")

        all_allocs_prop_fair["FP Scheduling"], links_rates_prop_fair["FP Scheduling"] = proportional_fairness_scheduling.FP_prop_fair(general_para, channel_losses, directlink_channel_losses, crosslink_channel_losses)
        # With taking ceil, always have slightly lower number as the number of clusters than FP long-term avg over time slots
        n_links_on_FP_propfair = np.floor(np.sum(np.mean(all_allocs_prop_fair["FP Scheduling"], axis=1),axis=1)).astype(int)
        print("FP Propfair activates {}/{} links on avg per layout.".format(np.mean(n_links_on_FP_propfair), N))
        assert np.shape(n_links_on_FP_propfair) == (n_layouts, )
        for clustering_method in all_clustering_methods:
            # Construct clusters first
            cluster_assignments = []
            for i in range(n_layouts):
                clusters_one_layout, _ = clustering_and_scheduling(layouts[i], channel_losses[i], n_links_on_FP_propfair[i], clustering_method)
                cluster_assignments.append(clusters_one_layout)
            cluster_assignments = np.array(cluster_assignments)
            assert np.shape(cluster_assignments) == (n_layouts, N)
            all_allocs_prop_fair[clustering_method], links_rates_prop_fair[clustering_method] = proportional_fairness_scheduling.Clustering_based_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses, cluster_assignments, clustering_method)
        all_allocs_prop_fair["Greedy Scheduling"], links_rates_prop_fair["Greedy Scheduling"] = proportional_fairness_scheduling.Greedy_Scheduling_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses)
        all_allocs_prop_fair["All Active"], links_rates_prop_fair["All Active"] = proportional_fairness_scheduling.all_active_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses)
        all_allocs_prop_fair["Random Scheduling"], links_rates_prop_fair["Random Scheduling"] = proportional_fairness_scheduling.random_scheduling_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses)
        # Simplest round robin (one at a time). Not plotting this for the final paper.
        # all_allocs_prop_fair["Vanilla Round Robin"], links_rates_prop_fair["Vanilla Round Robin"] = proportional_fairness_scheduling.vanilla_round_robin_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses)

        # Compute sum log avg rate utility
        links_avg_rates_II = {}
        print("[Layouts Avg for sum log mean rate (Mbps)]:")
        for clustering_method, rates in links_rates_prop_fair.items():
            assert np.shape(rates) == (n_layouts, general_para.log_utility_time_slots, N)
            links_avg_rates_II[clustering_method] = np.mean(rates, axis=1) # number of layouts X N
            evaluate_result = np.sum(np.log(links_avg_rates_II[clustering_method]/1e6), axis=1) # number of layouts
            links_avg_rates_II[clustering_method] = links_avg_rates_II[clustering_method].flatten() # flatten for plotting
            print("[{}]:{};".format(clustering_method, round(np.mean(evaluate_result),2)), end="")
        print("\n")
        plot_meanrates_CDF(general_para, links_avg_rates_II)

    print("Script Completed Successfully!")