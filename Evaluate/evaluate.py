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
    n_layouts = np.size(result_dict["FP Scheduling"])
    print("FP Scheduling mean value: {}".format(np.mean(result_dict["FP Scheduling"])))
    for method_key in result_dict.keys():
        assert np.shape(result_dict[method_key])==(n_layouts, ), "Wrong shape for method {}: {}".format(method_key, np.shape(result_dict[method_key]))
        if (method_key == "FP Scheduling"):
            continue
        ratios = result_dict[method_key] / result_dict["FP Scheduling"] * 100
        print("[{}]: avg {}% of {};".format(method_key, round(np.mean(ratios), 2), "FP Scheduling"), end="")
        print("{} mean value: {}".format(method_key, np.mean(result_dict[method_key])))
    print("\n")
    return

def obtain_line_colors(n_colors): # return colors differentiating among different methods
    colormap = plt.get_cmap('gist_rainbow')
    line_colors = [colormap(1. * i / n_colors) for i in range(n_colors)]
    return line_colors

def plot_rates_CDF(general_para, rate_results, task_title, channel_model):
    fig = plt.figure()
    ax = fig.gca()
    plot_description = "[{}] {} with channel model {}".format(task_title, general_para.setting_str, channel_model)
    plt.title(plot_description)
    plt.xlabel("Mbps")
    plt.ylabel("Cumulative Distribution of D2D Networks")
    plt.grid(linestyle="dotted")
    ax.set_ylim(bottom=0)
    line_colors = obtain_line_colors(len(rate_results.keys()))
    for method_index, method_key in enumerate(rate_results.keys()):
        rates = np.sort(rate_results[method_key])
        plt.plot(rates / 1e6, np.arange(1, np.size(rates) + 1) / np.size(rates), label="{}".format(method_key), color=line_colors[method_index])
        plt.legend()
    plt.savefig("{}.png".format(plot_description))
    return

if(__name__ =='__main__'):
    general_para = general_parameters.parameters()
    N = general_para.number_of_links
    layouts = np.load(general_para.test_dir + general_para.file_names["layouts"])
    path_losses = np.load(general_para.test_dir + general_para.file_names["path_losses"])
    n_layouts = np.shape(layouts)[0]
    assert np.shape(layouts) == (n_layouts, N, 4) and np.shape(path_losses) == (n_layouts, N, N)
    print("Evaluate {} over {} layouts".format(general_para.setting_str, n_layouts))

    for channel_model in ["Pure Path Losses", "With Fading", "With Shadowing & Fading"]:
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
        all_cluster_assignments = {}

        # Integer results
        all_allocs["FP Scheduling"] = benchmarks.FP(general_para, channel_losses, np.ones([n_layouts, N]), scheduling_output=True)
        n_links_on_FP = np.sum(all_allocs["FP Scheduling"],axis=1)

        cluster_assignments = []
        allocs = []
        for i in range(n_layouts):
            if ((i + 1) * 100 / n_layouts % 50 == 0):
                print("[SC SumRate] At {}/{} layouts.".format(i + 1, n_layouts))
            clusters_one_layout = Spectral_Clustering.clustering(layouts[i], channel_losses[i], n_links_on_FP[i])
            allocs_one_layout = Spectral_Clustering.scheduling(channel_losses[i], clusters_one_layout)
            cluster_assignments.append(clusters_one_layout)
            allocs.append(allocs_one_layout)
        cluster_assignments = np.array(cluster_assignments)
        allocs = np.array(allocs)
        assert np.shape(cluster_assignments) == np.shape(allocs) == (n_layouts, N)
        all_allocs["Spectral Clustering"] = allocs
        all_cluster_assignments["Spectral Clustering"] = cluster_assignments

        cluster_assignments = []
        allocs = []
        for i in range(n_layouts):
            if ((i + 1) * 100 / n_layouts % 50 == 0):
                print("[HC SumRate] At {}/{} layouts.".format(i + 1, n_layouts))
            clusters_one_layout = Hierarchical_Clustering.clustering(layouts[i], channel_losses[i], n_links_on_FP[i])
            allocs_one_layout = Hierarchical_Clustering.scheduling(channel_losses[i], clusters_one_layout)
            cluster_assignments.append(clusters_one_layout)
            allocs.append(allocs_one_layout)
        cluster_assignments = np.array(cluster_assignments)
        allocs = np.array(allocs)
        assert np.shape(cluster_assignments) == np.shape(allocs) == (n_layouts, N)
        all_allocs["Hierarchical Clustering"] = allocs
        all_cluster_assignments["Hierarchical Clustering"] = cluster_assignments

        cluster_assignments = []
        allocs = []
        for i in range(n_layouts):
            if ((i + 1) * 100 / n_layouts % 50 == 0):
                print("[KM SumRate] At {}/{} layouts.".format(i + 1, n_layouts))
            clusters_one_layout = K_Means.clustering(layouts[i], n_links_on_FP[i])
            allocs_one_layout = K_Means.scheduling(layouts[i], clusters_one_layout)
            cluster_assignments.append(clusters_one_layout)
            allocs.append(allocs_one_layout)
        cluster_assignments = np.array(cluster_assignments)
        allocs = np.array(allocs)
        assert np.shape(cluster_assignments) == np.shape(allocs) == (n_layouts, N)
        all_allocs["K-Means"] = allocs
        all_cluster_assignments["K-Means"] = cluster_assignments

        all_allocs["Greedy Scheduling"] = benchmarks.greedy_scheduling(general_para, directlink_channel_losses, crosslink_channel_losses, np.ones([n_layouts, N]))
        all_allocs["All Active"] = np.ones([n_layouts, N]).astype(float)
        all_allocs["Random Scheduling"] = np.random.randint(2, size=[n_layouts, N]).astype(float) # Return float type
        all_allocs["Strongest Link"] = benchmarks.Strongest_Link(general_para, directlink_channel_losses)


        # EVALUATION AND COMPARISON
        all_allocs_means = {}
        for method_key in all_allocs.keys():
            assert np.shape(all_allocs[method_key]) == (n_layouts, N) # checking dimension validity
            all_allocs_means[method_key] = np.mean(all_allocs[method_key],axis=1)
        compute_avg_ratio(all_allocs_means, "Scheduling Mean")

        # Evaluate Stage I: single time slot
        links_rates_I = {}
        evaluate_results_I = {}
        for method_key in all_allocs.keys():
            links_rates_I[method_key] = utils.compute_rates(general_para, all_allocs[method_key], directlink_channel_losses, crosslink_channel_losses)
            assert np.shape(links_rates_I[method_key]) == (n_layouts, N)
            evaluate_results_I[method_key] = np.sum(links_rates_I[method_key], axis=1)
        compute_avg_ratio(evaluate_results_I, "Sum Rate Single Timeslot")
        plot_rates_CDF(general_para, evaluate_results_I, "Sum Rate Single Timeslot", channel_model)

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
            layouts_to_visualize = np.random.randint(low=0, high=n_layouts, size=3)
            for layout_to_visualize in layouts_to_visualize:
                plt.title("Layout #{}".format(layout_to_visualize))
                ax = plt.subplot(241)
                utils.plot_allocs_on_layout(ax, layouts[layout_to_visualize], all_allocs["FP Scheduling"][layout_to_visualize], "FP Scheduling")
                ax = plt.subplot(242)
                utils.plot_allocs_on_layout(ax, layouts[layout_to_visualize], all_allocs['Spectral Clustering'][layout_to_visualize], "Spectral Clustering")
                ax = plt.subplot(243)
                utils.plot_allocs_on_layout(ax, layouts[layout_to_visualize], all_allocs['Hierarchical Clustering'][layout_to_visualize], "Spectral Clustering")
                ax = plt.subplot(244)
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
        print("Evaluating log long-term avg rate results...")

        all_allocs_prop_fair["FP Scheduling"], links_rates_prop_fair["FP Scheduling"] = proportional_fairness_scheduling.FP_prop_fair(general_para, channel_losses, directlink_channel_losses, crosslink_channel_losses)
        all_allocs_prop_fair["K-Means"], links_rates_prop_fair["K-Means"] = proportional_fairness_scheduling.K_Means_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses, all_cluster_assignments["K-Means"])
        all_allocs_prop_fair["Spectral Clustering"], links_rates_prop_fair["Spectral Clustering"] = proportional_fairness_scheduling.Spectral_Clustering_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses, all_cluster_assignments["Spectral Clustering"])
        all_allocs_prop_fair["Hierarchical Clustering"], links_rates_prop_fair["Hierarchical Clustering"] = proportional_fairness_scheduling.Hierarchical_Clustering_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses, all_cluster_assignments["Hierarchical Clustering"])
        all_allocs_prop_fair["Greedy Scheduling"], links_rates_prop_fair["Greedy Scheduling"] = proportional_fairness_scheduling.Greedy_Scheduling_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses)
        all_allocs_prop_fair["All Active"], links_rates_prop_fair["All Active"] = proportional_fairness_scheduling.all_active_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses)
        all_allocs_prop_fair["Random Scheduling"], links_rates_prop_fair["Random Scheduling"] = proportional_fairness_scheduling.random_scheduling_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses)
        # Simplest round robin (one at a time)
        all_allocs_prop_fair["Vanilla Round Robin"], links_rates_prop_fair["Vanilla Round Robin"] = proportional_fairness_scheduling.vanilla_round_robin_prop_fair(general_para, directlink_channel_losses, crosslink_channel_losses)

        # Compute sum log avg rate utility
        links_avg_rates_II = {}
        print("[Layouts Avg for sum log mean rate (Mbps)]:")
        for method_key, rates in links_rates_prop_fair.items():
            assert np.shape(rates) == (n_layouts, general_para.log_utility_time_slots, N)
            links_avg_rates_II[method_key] = np.mean(rates, axis=1) # number of layouts X N
            evaluate_result = np.sum(np.log(links_avg_rates_II[method_key]/1e6), axis=1) # number of layouts
            links_avg_rates_II[method_key] = links_avg_rates_II[method_key].flatten() # flatten for plotting
            print("[{}]:{};".format(method_key, round(np.mean(evaluate_result),2)), end="")
        print("\n")
        plot_rates_CDF(general_para, links_avg_rates_II, "Log Utilities Multiple Timeslots", channel_model)

    print("Script Completed Successfully!")