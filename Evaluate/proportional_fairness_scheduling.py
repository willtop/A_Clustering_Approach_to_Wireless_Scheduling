# Script for computing proportional fairness schedulings over consequtive time slots
import numpy as np
import benchmarks
import utils
from itertools import cycle

def proportional_update_weights(general_para, rates, weights):
    return 1 / (general_para.alpha_proportional_fairness_update / weights + (1 - general_para.alpha_proportional_fairness_update) * rates)

def FP_prop_fair(general_para, gains, gains_diagonal, gains_nondiagonal):
    print("Sequential scheduling for FP...")
    number_of_layouts, N = np.shape(gains_diagonal)
    allocs_alltime = []
    rates_alltime = []
    prop_weights = np.ones([number_of_layouts, N])
    for i in range(general_para.log_utility_time_slots):
        if ((i + 1) * 100 / general_para.log_utility_time_slots % 50 == 0):
            print("At {}/{} time slots...".format(i + 1, general_para.log_utility_time_slots))
        allocs = benchmarks.FP(general_para, gains, prop_weights, scheduling_output=True)
        rates = utils.compute_rates(general_para, allocs, gains_diagonal, gains_nondiagonal)
        allocs_alltime.append(allocs)
        rates_alltime.append(rates)
        prop_weights = proportional_update_weights(general_para, rates, prop_weights)
    allocs_alltime = np.transpose(np.array(allocs_alltime), (1, 0, 2))
    rates_alltime = np.transpose(np.array(rates_alltime), (1, 0, 2))
    assert np.shape(allocs_alltime) == np.shape(rates_alltime) == (number_of_layouts, general_para.log_utility_time_slots, N)
    print("[FP multiple timeslots] Computation finished!")
    return allocs_alltime, rates_alltime

# For now, the round robin iterating order is random within each cluster (with fixed orders for each round)
# unlike prioritizing the link with the strongest direct link gain
# Can't think of a way to parallelize among layouts or even clusters within the same layout. For now, linearly
# go through each cluster within each layout with cycle iterators.
def Spectral_Clustering_prop_fair(general_para, gains_diagonal, gains_nondiagonal, cluster_assignments):
    print("Sequential scheduling for Spectral Clustering...")
    number_of_layouts, N = np.shape(gains_diagonal)
    assert np.shape(cluster_assignments) == (number_of_layouts, N)
    n_clusters = (np.max(cluster_assignments, axis=1)+1).astype(int) # number of layouts
    allocs_alltime = []
    rates_alltime = []
    # create iterator
    iterators_all_layouts = []
    for layout_id in range(number_of_layouts):
        iterators_one_layout = []
        for cluster_id in range(n_clusters[layout_id]):
            iterators_one_layout.append(cycle(np.where(cluster_assignments[layout_id]==cluster_id)[0]))
        iterators_all_layouts.append(iterators_one_layout)
    print("Iterators construction completed!")
    # Start sequential time slots scheduling
    for time_slot in range(general_para.log_utility_time_slots):
        if ((time_slot + 1) * 100 / general_para.log_utility_time_slots % 50 == 0):
            print("At {}/{} time slots...".format(time_slot + 1, general_para.log_utility_time_slots))
        allocs = np.zeros([number_of_layouts, N])
        for layout_id in range(number_of_layouts):
            for cluster_id in range(n_clusters[layout_id]):
                iterator_to_schedule = iterators_all_layouts[layout_id][cluster_id]
                link_to_schedule = next(iterator_to_schedule)
                allocs[layout_id][link_to_schedule] = 1
        rates = utils.compute_rates(general_para, allocs, gains_diagonal, gains_nondiagonal)
        allocs_alltime.append(allocs)
        rates_alltime.append(rates)
    allocs_alltime = np.transpose(np.array(allocs_alltime), (1, 0, 2))
    rates_alltime = np.transpose(np.array(rates_alltime), (1, 0, 2))
    assert np.shape(allocs_alltime) == np.shape(rates_alltime) == (number_of_layouts, general_para.log_utility_time_slots, N)
    print("[Spectral Clustering multiple timeslots] Computation finished")
    return allocs_alltime, rates_alltime

# For now, the round robin iterating order is random within each cluster (with fixed orders for each round)
# unlike prioritizing the link with the strongest direct link gain
# Can't think of a way to parallelize among layouts or even clusters within the same layout. For now, linearly
# go through each cluster within each layout with cycle iterators.
def Hierarchical_Clustering_prop_fair(general_para, gains_diagonal, gains_nondiagonal, cluster_assignments):
    print("Sequential scheduling for Hierarchical Clustering...")
    number_of_layouts, N = np.shape(gains_diagonal)
    assert np.shape(cluster_assignments) == (number_of_layouts, N)
    n_clusters = (np.max(cluster_assignments, axis=1)+1).astype(int) # number of layouts
    allocs_alltime = []
    rates_alltime = []
    # create iterator
    iterators_all_layouts = []
    for layout_id in range(number_of_layouts):
        iterators_one_layout = []
        for cluster_id in range(n_clusters[layout_id]):
            iterators_one_layout.append(cycle(np.where(cluster_assignments[layout_id]==cluster_id)[0]))
        iterators_all_layouts.append(iterators_one_layout)
    print("Iterators construction completed!")
    # Start sequential time slots scheduling
    for time_slot in range(general_para.log_utility_time_slots):
        if ((time_slot + 1) * 100 / general_para.log_utility_time_slots % 50 == 0):
            print("At {}/{} time slots...".format(time_slot + 1, general_para.log_utility_time_slots))
        allocs = np.zeros([number_of_layouts, N])
        for layout_id in range(number_of_layouts):
            for cluster_id in range(n_clusters[layout_id]):
                iterator_to_schedule = iterators_all_layouts[layout_id][cluster_id]
                link_to_schedule = next(iterator_to_schedule)
                allocs[layout_id][link_to_schedule] = 1
        rates = utils.compute_rates(general_para, allocs, gains_diagonal, gains_nondiagonal)
        allocs_alltime.append(allocs)
        rates_alltime.append(rates)
    allocs_alltime = np.transpose(np.array(allocs_alltime), (1, 0, 2))
    rates_alltime = np.transpose(np.array(rates_alltime), (1, 0, 2))
    assert np.shape(allocs_alltime) == np.shape(rates_alltime) == (number_of_layouts, general_para.log_utility_time_slots, N)
    print("[Hierarchical Clustering multiple timeslots] Computation finished")
    return allocs_alltime, rates_alltime

# For now, the round robin iterating order is random within each cluster (with fixed orders for each round)
# unlike prioritizing the link with the strongest direct link gain
# Can't think of a way to parallelize among layouts or even clusters within the same layout. For now, linearly
# go through each cluster within each layout with cycle iterators.
def K_Means_prop_fair(general_para, gains_diagonal, gains_nondiagonal, cluster_assignments):
    print("Sequential scheduling for K Means...")
    number_of_layouts, N = np.shape(gains_diagonal)
    assert np.shape(cluster_assignments) == (number_of_layouts, N)
    n_clusters = (np.max(cluster_assignments, axis=1) + 1).astype(int)  # number of layouts
    allocs_alltime = []
    rates_alltime = []
    # create iterator
    iterators_all_layouts = []
    for layout_id in range(number_of_layouts):
        iterators_one_layout = []
        for cluster_id in range(n_clusters[layout_id]):
            iterators_one_layout.append(cycle(np.where(cluster_assignments[layout_id] == cluster_id)[0]))
        iterators_all_layouts.append(iterators_one_layout)
    print("Iterators construction completed!")
    # Start sequential time slots scheduling
    for time_slot in range(general_para.log_utility_time_slots):
        if ((time_slot + 1) * 100 / general_para.log_utility_time_slots % 50 == 0):
            print("At {}/{} time slots...".format(time_slot + 1, general_para.log_utility_time_slots))
        allocs = np.zeros([number_of_layouts, N])
        for layout_id in range(number_of_layouts):
            for cluster_id in range(n_clusters[layout_id]):
                iterator_to_schedule = iterators_all_layouts[layout_id][cluster_id]
                link_to_schedule = next(iterator_to_schedule)
                allocs[layout_id][link_to_schedule] = 1
        rates = utils.compute_rates(general_para, allocs, gains_diagonal, gains_nondiagonal)
        allocs_alltime.append(allocs)
        rates_alltime.append(rates)
    allocs_alltime = np.transpose(np.array(allocs_alltime), (1, 0, 2))
    rates_alltime = np.transpose(np.array(rates_alltime), (1, 0, 2))
    assert np.shape(allocs_alltime) == np.shape(rates_alltime) == (number_of_layouts, general_para.log_utility_time_slots, N)
    print("[K Means multiple timeslots] Computation finished")
    return allocs_alltime, rates_alltime

def Greedy_Scheduling_prop_fair(general_para, gains_diagonal, gains_nondiagonal):
    print("Sequential scheduling for Greedy...")
    number_of_layouts, N = np.shape(gains_diagonal)
    allocs_alltime = []
    rates_alltime = []
    prop_weights = np.ones([number_of_layouts, N])
    for i in range(general_para.log_utility_time_slots):
        if ((i + 1) * 100 / general_para.log_utility_time_slots % 50 == 0):
            print("At {}/{} time slots...".format(i + 1, general_para.log_utility_time_slots))
        allocs = benchmarks.greedy_scheduling(general_para, gains_diagonal, gains_nondiagonal, prop_weights)
        rates = utils.compute_rates(general_para, allocs, gains_diagonal, gains_nondiagonal)
        allocs_alltime.append(allocs)
        rates_alltime.append(rates)
        prop_weights = proportional_update_weights(general_para, rates, prop_weights)
    allocs_alltime = np.transpose(np.array(allocs_alltime), (1, 0, 2))
    rates_alltime = np.transpose(np.array(rates_alltime), (1, 0, 2))
    assert np.shape(allocs_alltime) == np.shape(rates_alltime) == (number_of_layouts, general_para.log_utility_time_slots, N)
    np.save(general_para.test_dir + general_para.file_names["Greedy_Multi_Timeslots_Allocs"], allocs_alltime)
    np.save(general_para.test_dir + general_para.file_names["Greedy_Multi_Timeslots_Rates"], rates_alltime)
    print("[Greedy multiple timeslots] Computation finished and results saved")
    return allocs_alltime, rates_alltime

def all_active_prop_fair(general_para, gains_diagonal, gains_nondiagonal):
    number_of_layouts, N = np.shape(gains_diagonal)
    allocs = np.ones([number_of_layouts, N]).astype(float)
    rates = utils.compute_rates(general_para, allocs, gains_diagonal, gains_nondiagonal)
    allocs_alltime = np.tile(np.expand_dims(allocs, axis=0), (general_para.log_utility_time_slots, 1, 1))
    rates_alltime = np.tile(np.expand_dims(rates, axis=0), (general_para.log_utility_time_slots, 1, 1))
    allocs_alltime = np.transpose(np.array(allocs_alltime), (1, 0, 2))
    rates_alltime = np.transpose(np.array(rates_alltime), (1, 0, 2))
    assert np.shape(allocs_alltime) == np.shape(rates_alltime) == (number_of_layouts, general_para.log_utility_time_slots, N)
    return allocs_alltime, rates_alltime

def random_scheduling_prop_fair(general_para, gains_diagonal, gains_nondiagonal):
    number_of_layouts, N = np.shape(gains_diagonal)
    allocs_alltime = np.random.randint(2,size=(general_para.log_utility_time_slots, number_of_layouts, N)).astype(float)
    rates_alltime = []
    for i in range(general_para.log_utility_time_slots):
        rates_oneslot = utils.compute_rates(general_para, allocs_alltime[i], gains_diagonal, gains_nondiagonal)
        rates_alltime.append(rates_oneslot)
    allocs_alltime = np.transpose(np.array(allocs_alltime), (1, 0, 2))
    rates_alltime = np.transpose(np.array(rates_alltime), (1, 0, 2))
    assert np.shape(allocs_alltime) == np.shape(rates_alltime) == (
    number_of_layouts, general_para.log_utility_time_slots, N)
    return allocs_alltime, rates_alltime

# Vanilla round robin: iterating over links, one at a time, blindly based on link indices
def vanilla_round_robin_prop_fair(general_para, gains_diagonal, gains_nondiagonal):
    number_of_layouts, N = np.shape(gains_diagonal)
    allocs_alltime = []
    rates_alltime = []
    iterator = cycle(range(N))
    for i in range(general_para.log_utility_time_slots):
        allocs_oneslot = np.zeros([number_of_layouts, N])
        allocs_oneslot[:, next(iterator)] = 1
        rates_oneslot = utils.compute_rates(general_para, allocs_oneslot, gains_diagonal, gains_nondiagonal)
        allocs_alltime.append(allocs_oneslot)
        rates_alltime.append(rates_oneslot)
    allocs_alltime = np.transpose(np.array(allocs_alltime), (1, 0, 2))
    rates_alltime = np.transpose(np.array(rates_alltime), (1, 0, 2))
    assert np.shape(allocs_alltime) == np.shape(rates_alltime) == (number_of_layouts, general_para.log_utility_time_slots, N)
    return allocs_alltime, rates_alltime

    # # Max weight activation heuristic
    # if("max_weight_sch" in all_benchmarks):
    #     print("==================================Max Weight Only Scheduling heuristic===========================")
    #     method_key = "Max Weight Only"
    #     allocs = []; rates = []
    #     weights = np.ones([test_layouts, N])
    #     start_time = time.time()
    #     for i in range(test_slots_per_layout):  # could make the entire process a function and only feeding in different allocation rule, but for now just plainly write it
    #         allocs_oneslot = np.zeros([test_layouts, N])
    #         allocs_oneslot[np.arange(test_layouts), np.argmax(weights,axis=1)] = 1
    #         allocs.append(allocs_oneslot)
    #         # compute rates parallely over many layouts
    #         rates_oneslot = utils.compute_rates(general_para, allocs_oneslot, test_gains_diagonal, test_gains_nondiagonal) # layouts X N
    #         rates.append(rates_oneslot)
    #         weights = utils.proportional_update_weights(general_para, weights, rates_oneslot)
    #     allocs = np.transpose(np.array(allocs), (1,0,2)); assert np.shape(allocs) == (test_layouts, test_slots_per_layout, N)
    #     rates = np.transpose(np.array(rates), (1,0,2)); assert np.shape(rates) == (test_layouts, test_slots_per_layout, N)
    #     all_allocs[method_key] = allocs
    #     all_rates[method_key] = rates
    #     print("{} with time spent: {}".format(method_key, time.time() - start_time))
    #
    # if("direct_allocs_backprop" in all_benchmarks):
    #     print("==================================Direct Allocations Backprop Scheduling heuristic===========================")
    #     method_key = "Direct Allocs BackProp"
    #     allocs, rates = Direct_Allocs_BackProp.direct_backprop(general_para, test_gains_diagonal, test_gains_nondiagonal)
    #     all_allocs[method_key] = allocs
    #     all_rates[method_key] = rates
    #
    # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<EVALUATION AND COMPARISON>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print("[Activation Portion] ")
    # for method_key in all_allocs.keys():
    #     print("[{}]: {}%; ".format(method_key, round(np.mean(all_allocs[method_key]) * 100, 2)), end="")
    # print("\n")
    #
    # # Sum log mean rates evaluation
    # print("----------------------------------------------------------------")
    # print("[Sum Log Mean Rates (Mean over {} layouts)]:".format(test_layouts))
    # all_sum_log_mean_rates = dict()
    # all_link_mean_rates = dict()
    # global_max_mean_rate = 0 # for plotting upperbound
    # for method_key in all_allocs.keys():
    #     link_mean_rates = np.mean(all_rates[method_key], axis=1); assert np.shape(link_mean_rates) == (test_layouts, N), "Wrong shape: {}".format(np.shape(link_mean_rates))
    #     all_link_mean_rates[method_key] = link_mean_rates.flatten() # (test_layouts X N, )
    #     global_max_mean_rate = max(global_max_mean_rate, np.max(link_mean_rates))
    #     all_sum_log_mean_rates[method_key] = np.mean(np.sum(np.log(link_mean_rates/1e6 + 1e-5),axis=1))
    # for method_key in all_allocs.keys():
    #     print("[{}]: {}; ".format(method_key, all_sum_log_mean_rates[method_key]), end="")
    # print("\n")
    #
    # print("[Bottom 5-Percentile Mean Rate (Aggregate over all layouts)]:")
    # for method_key in all_allocs.keys():
    #     meanR_5pert = np.percentile((all_link_mean_rates[method_key]).flatten(), 5)
    #     print("[{}]: {}; ".format(method_key, meanR_5pert), end="")
    # print("\n")
    #
    # # Produce the CDF plot for mean rates achieved by single links
    # if(formal_CDF_legend_option):
    #     line_styles = dict()
    #     line_styles["Deep Learning"] = 'r-'
    #     line_styles["FP"] = 'g:'
    #     line_styles["Weighted Greedy"] = 'm-.'
    #     line_styles["Max Weight Only"] = 'b--'
    #     line_styles["All Active"] = 'c-.'
    #     line_styles["Random"] = 'y--'
    # fig = plt.figure(); ax = fig.gca()
    # plt.xlabel("Mean Rate for each link (Mbps)")
    # plt.ylabel("Cumulative Distribution Function")
    # plt.grid(linestyle="dotted")
    # ax.set_xlim(left=0, right=0.45*global_max_mean_rate/1e6)
    # ax.set_ylim(ymin=0)
    # if(formal_CDF_legend_option):
    #     allocs_keys_ordered = ["Deep Learning", "FP", "Weighted Greedy", "Max Weight Only", "All Active", "Random"]
    #     for method_key in allocs_keys_ordered:
    #         if(method_key not in all_link_mean_rates.keys()):
    #             print("[{}] Allocation not computed! Skipping...".format(method_key))
    #             continue
    #         plt.plot(np.sort(all_link_mean_rates[method_key])/1e6, np.arange(1, test_layouts*N + 1) / (test_layouts*N), line_styles[method_key], label=method_key)
    # else:
    #     for method_key in all_link_mean_rates.keys():
    #         plt.plot(np.sort(all_link_mean_rates[method_key])/1e6, np.arange(1, test_layouts*N + 1) / (test_layouts*N), label=method_key)
    # plt.legend()
    # plt.show()
    # print("----------------------------------------------------------------")
    #
    # # Plot scheduling and subsets sent as scheduling candidates
    # print("Plotting subset selection for each scheduling")
    # v_layout = np.random.randint(low=0, high=test_layouts)
    # v_layout = 5
    # # Form a plot for every method having binary weights subset links scheduling
    # for method_key in all_subsets.keys():
    #     print("[subsets_select_plotting] Plotting {} allocations...".format(method_key))
    #     v_subsets = all_subsets[method_key][v_layout]
    #     v_allocs = all_allocs[method_key][v_layout]
    #     v_locations = test_locations[v_layout]
    #     fig = plt.figure(); plt.title("{} for layout #{}".format(method_key, v_layout))
    #     ax = fig.gca(); ax.set_xticklabels([]); ax.set_yticklabels([])
    #     for i in range(24):  # visualize first several time steps for each layout
    #         ax = fig.add_subplot(4, 6, i + 1); ax.set_xticklabels([]); ax.set_yticklabels([])
    #         tx_locs = v_locations[:, 0:2]; rx_locs = v_locations[:, 2:4]
    #         plt.scatter(tx_locs[:, 0], tx_locs[:, 1], c='r', s=2); plt.scatter(rx_locs[:, 0], rx_locs[:, 1], c='b', s=2)
    #         for j in range(N):  # plot both subsets and activated links (assume scheduling outputs)
    #             if v_subsets[i][j] == 1:
    #                 plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], 'b', linewidth=2.2, alpha=0.25)
    #             if v_allocs[i][j] == 1:
    #                 plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], 'r', linewidth=0.8)
    #     plt.subplots_adjust(wspace=0, hspace=0)
    #     plt.show()
    #
    # # Sequential Plotting
    # # plot for one randomly selected layout over all methods
    # print("Plotting sequential plotting of schedulings...")
    # for method_key in all_allocs.keys():
    #     if (method_key in ["All Active", "Random"]):
    #         continue  # Don't plot for these trivial allocations
    #     print("[sequential_timeslots_plotting] Plotting {} allocations...".format(method_key))
    #     v_allocs = all_allocs[method_key][v_layout]
    #     v_locations = test_locations[v_layout]
    #     fig = plt.figure(); plt.title("{} for layout #{}".format(method_key, v_layout))
    #     ax = fig.gca(); ax.set_xticklabels([]); ax.set_yticklabels([])
    #     for i in range(24): # visualize first several time steps for each layout
    #         ax = fig.add_subplot(4, 6, i+1); ax.set_xticklabels([]); ax.set_yticklabels([])
    #         v_allocs_oneslot = v_allocs[i]
    #         tx_locs = v_locations[:, 0:2];  rx_locs = v_locations[:, 2:4]
    #         plt.scatter(tx_locs[:, 0], tx_locs[:, 1], c='r', s=3); plt.scatter(rx_locs[:, 0], rx_locs[:, 1], c='b', s=3)
    #         for j in range(N):  # plot all activated links
    #             line_color = 1-v_allocs_oneslot[j]
    #             if line_color==0:
    #                 line_color = 0.0 # deal with 0 formatting error problem
    #             plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], '{}'.format(line_color))# have to do 1 minus since the smaller the number the darker it gets
    #     plt.subplots_adjust(wspace=0, hspace=0)
    #     plt.show()
    #
    # print("Script Completed Successfully!")
