import numpy as np


# Analyze some of the data already saved to files
root_dir = '../saved_models/'
# DIGIT
# prefix = 'mnist_protos_'  # CSN for mnist digit aligned
# prefix = 'mnist_digit_cost_'  # CSN for mnist digit aligned
# prefix = 'hierarchy_mnist_protos_'  # HPN for mnist digit aligned
# prefix = 'hierarchy_mnist_protos_onehotted_'  # HPN for mnist digit aligned
# prefix = 'hierarchy_mnist_protos_cost_'  # HPN reporting average cost
# prefix = 'random_mnist_protos_'  # HPN for mnist digit aligned
prefix = 'random_mnist_cost_'  # CSN random with cost
# FASHION
# prefix = 'mnist_fashion_'  # CSN for mnist fashion aligned
# prefix = 'mnist_fashion_cost_'  # CSN for mnist fashion with average cost
# prefix = 'hierarchy_fashion_'  # HPN for mnist fashion aligned
# prefix = 'hierarchy_fashion_cost_'  # HPN for mnist fashion aligned
# prefix = 'hierarchy_fashion_new_'  # HPN for mnist fashion aligned
# prefix = 'random_fashion_protos_'  # HPN for mnist digit aligned
# prefix = 'random_fashion_cost_'  # HPN for mnist digit with cost
# CIFAR100
# prefix = 'cifar100_protos_'  # CSN for cifar100
# prefix = 'cifar100_csn_stop10_'  # CSN for cifar100 optimized for y acc
# prefix = 'cifar100_csn_align_'  # CSN for cifar100 optimized for alignment
# prefix = 'cifar100_csn_align_cost_'  # CSN for cifar100 with average cost calculated
# prefix = 'cifar100_csn_init_cost_'  # CSN for cifar100 with average cost calculated, no training
# prefix = 'cifar100_csn_align_longer5_'  # CSN for cifar100 optimized for alignment
# prefix = 'hierarchy_cifar100_stop10_'  # HPN for cifar100 optimized for y acc
# prefix = 'hierarchy_cifar100_cost_'  # HPN for cifar100 optimized for y acc
# CIFAR100 with a deeper hierarchy.
# prefix = 'cifar100_csn_align_deep_'  # CSN for cifar100 optimized for y acc with deeper
# prefix = 'cifar100_csn_noalign_deep_'  # CSN for cifar100 optimized for y acc with deeper
# prefix = 'cifar100_csn_init_deep_'  # CSN for cifar100 with no training
# BOLTS
# prefix = 'bolts_'  # CSN for bolts with one run from each subject held out
# prefix = 'bolts_subj0_'  # CSN for bolts with all runs from subject 0 held out.


acc1s = []
acc2s = []
final_aligns = []
diffs = []
sames = []
tree_matches = []
edit_dists = []
average_costs = []
for model_id in range(10):
    with open(root_dir + prefix + str(model_id) + '.csv', 'r') as data_file:
        for line_idx, line in enumerate(data_file):
            split_line = line.split(" ")
            if line_idx == 0:
                accuracies = [float(entry) for entry in split_line]
                acc1 = accuracies[0]
                acc2 = accuracies[1]
                acc1s.append(acc1)
                acc2s.append(acc2)
                continue
            if line_idx == 1:
                final_aligns.append(float(split_line[-1]))
                continue
            if line_idx == 2:
                diffs.append(float(split_line[0]))
                continue
            if line_idx == 3:
                sames.append(float(split_line[0]))
                continue
            if line_idx == 4:
                tree_matches.append(float(split_line[0]))
                continue
            if line_idx == 5:
                edit_dists.append(float(split_line[0]))
                continue
            if line_idx == 6:
                print("Average cost case?")
                average_costs.append(float(split_line[0]))
                continue
            assert False

names = ['accs1', 'accs2', 'aligns', 'diffs', 'sames', 'exact', 'edit', 'average costs']
metrics = [acc1s, acc2s, final_aligns, diffs, sames, tree_matches, edit_dists, average_costs]
for i, name in enumerate(names):
    print()
    print("Average", name, np.round(np.mean(metrics[i]), 3))
    print("Std", name, np.round(np.std(metrics[i]), 3))
