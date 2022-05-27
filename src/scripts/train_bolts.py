import networkx as nx
import numpy as np
import tensorflow as tf

from src.data_parsing.bolts.data_parser import DataParser
from src.models.proto_model import ProtoModel
from src.utils.eval import trees_match, graph_edit_dist
from src.utils.gpu import set_gpu_config
from src.utils.to_files import write_to_file

set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

latent_dim = 16

class_labels = [i for i in range(8)]
bolt_to_lr_mapping = {0: 0,
                      1: 0,
                      2: 0,
                      3: 0,
                      4: 1,
                      5: 1,
                      6: 1,
                      7: 1}


# Define the bolt tree
def get_lr_tree():
    lr = ['left', 'right']
    tree = nx.DiGraph()
    for low, high in bolt_to_lr_mapping.items():
        tree.add_edge(lr[high], low)
    for high in lr:
        tree.add_edge('origin', high)
    nodes = list(tree.nodes)
    for node in nodes:
        tree.add_node(node, name=str(node))
    return tree


ground_truth_tree = get_lr_tree()
include_subj_id = False

# Run a bunch of trials.
# y_accuracy = []
# alignments = []
metric_names = ['y_acc', 'align', 's_diff', 'slopes', 'impact', 'disparity', 'average cost']
metrics = {}
for metric_name in metric_names:
    metrics[metric_name] = []
for model_id in range(10):
    np.random.seed(model_id)
    tf.random.set_seed(model_id)

    DURATION = int(50 * 1.0)  # 50 frames per second
    STEP_SIZE = 10
    parser = DataParser()
    data, targets = parser.get_traj_and_targ()

    excluded_subjects = [0, 1]
    excluded_runs = []
    # for excluded_subject in excluded_subjects:
    #     excluded_runs.extend(data.get_subj_to_global_run_id().get(excluded_subject))
    # TEST_RUNS = excluded_runs  # Subject 0 did runs 0-43, inclusive
    TEST_RUNS = []
    for subj_id in [0, 1, 2, 4, 5, 6, 7]:
        for duplicate in range(5):
            subj_runs = data.get_subj_to_global_run_id().get(subj_id)
            # print("Subj runs", subj_runs)
            TEST_RUNS.append(subj_runs[int(np.random.random() * len(subj_runs))])
    print("Using test runs", TEST_RUNS)
    # TEST_RUNS = [1, 50, 100, 150, 200, 250, 300, 350]  # Subject 0 did runs 0-43, inclusive
    training_data, test_data = data.get_hold_some_out_data(duration=DURATION, step_size=STEP_SIZE, test_idxs=TEST_RUNS)
    bolt_train, traj_train, subj_train = training_data
    bolt_test, traj_test, subj_test = test_data
    bolt_train_one_hot = tf.keras.utils.to_categorical(bolt_train)
    bolt_test_one_hot = tf.keras.utils.to_categorical(bolt_test)
    subj_train_one_hot = tf.keras.utils.to_categorical(subj_train, num_classes=8)
    subj_test_one_hot = tf.keras.utils.to_categorical(subj_test, num_classes=8)

    lr_train_one_hot = np.zeros((bolt_train.shape[0], 2))
    for i, y in enumerate(bolt_train):
        lr_train_one_hot[i][bolt_to_lr_mapping.get(y)] = 1.0
    lr_test = np.zeros((bolt_test.shape[0]))
    lr_test_one_hot = np.zeros((bolt_test.shape[0], 2))
    for i, y in enumerate(bolt_test):
        lr_test[i] = bolt_to_lr_mapping.get(y)
        lr_test_one_hot[i][bolt_to_lr_mapping.get(y)] = 1.0

    # It goes bolts, lr, subject
    tasks_to_use = [True, True, True]
    num_tasks = sum(tasks_to_use)
    all_output_sizes = [8, 2, 8]
    all_train_one_hot_outputs = [bolt_train_one_hot, lr_train_one_hot, subj_train_one_hot]
    all_test_outputs = [bolt_test, lr_test, subj_test]
    all_test_one_hot_outputs = [bolt_test_one_hot, lr_test_one_hot, subj_test_one_hot]

    # classification_weights = [1 for _ in range(num_tasks)]
    classification_weights = [1, 0, 1]
    # proto_dist_weights = [0.1, 0.1, 0.1]  # How realistic are the prototypes
    proto_dist_weights = [0.1, 0, 0.1]  # How realistic are the prototypes
    feature_dist_weights = [1 for _ in range(num_tasks)]  # How close to prototypes are embeddings (cluster size)
    feature_dist_weights[1] = 0
    feature_dist_weights[2] = 1
    disentangle_weight = [[0 for _ in range(num_tasks)] for _ in range(num_tasks)]
    # disentangle_weight[0] = [0, -10, 100]  # For mix
    # disentangle_weight[0] = [0, -10, 0]  # Only hier
    disentangle_weight[0] = [0, 0, 100]  # Only fair
    kl_losses = [2, 0, 2]  # For fair
    # kl_losses = [0 for _ in range(num_tasks)]
    duplication_factors = [1 for _ in range(num_tasks)]

    one_hot_output_train = []
    output_test = []
    one_hot_output_test = []
    output_sizes = []
    for task_idx, use_task in enumerate(tasks_to_use):
        if not use_task:
            continue
        one_hot_output_train.append(all_train_one_hot_outputs[task_idx])
        output_test.append(all_test_outputs[task_idx])
        one_hot_output_test.append(all_test_one_hot_outputs[task_idx])
        output_sizes.append(all_output_sizes[task_idx])
    # one_hot_output_train = [bolt_train_one_hot] if not use_hierarchical else [bolt_train_one_hot, lr_train_one_hot]
    # output_test = [bolt_test] if not use_hierarchical else [bolt_test, lr_test]
    # one_hot_output_test = [bolt_test_one_hot] if not use_hierarchical else [bolt_test_one_hot, lr_test_one_hot]

    # Create, train, and eval the model
    input_size = 150 if not include_subj_id else 151
    proto_model = ProtoModel(output_sizes, duplication_factors=duplication_factors, input_size=input_size,
                             decode_weight=0.1, classification_weights=classification_weights,
                             proto_dist_weights=proto_dist_weights, feature_dist_weights=feature_dist_weights,
                             disentangle_weights=disentangle_weight, kl_losses=kl_losses, latent_dim=latent_dim,
                             align_fn=tf.reduce_mean)
    train_inputs = traj_train if not include_subj_id else np.concatenate([traj_train, np.reshape(subj_train, (-1, 1))], axis=1)
    test_data = traj_test if not include_subj_id else np.concatenate([traj_test, np.reshape(subj_test, (-1, 1))], axis=1)
    test_inputs = [test_data, test_data]
    test_inputs.extend(one_hot_output_test)
    proto_model.train(train_inputs, train_inputs, one_hot_output_train, batch_size=256, epochs=50)  # TODO: 10 epochs
    # y_accs, s_diff, alignment, prob_updates, disparate_impact, demographic_disparity, average_cost = proto_model.evaluate(test_data, test_data, output_test, one_hot_output_test, protected_idx=2, do_fairness_eval=True)
    y_accs, s_diff, alignment, slope, disparate_impact, demographic_disparity, average_cost = proto_model.evaluate(test_data, test_data, output_test, one_hot_output_test, protected_idx=2, do_fairness_eval=True, gold_tree=(ground_truth_tree, class_labels))
    # metric_names = ['y_acc', 'align', 's_diff', 'impact', 'disparity']
    metrics['y_acc'].append(y_accs)
    metrics['s_diff'].append(s_diff)
    metrics['align'].append(alignment)
    metrics['slopes'].append(np.mean(slope))
    metrics['impact'].append(disparate_impact)
    metrics['disparity'].append(demographic_disparity)
    metrics['average cost'].append(average_cost)
    # if tasks_to_use[0] and tasks_to_use[1]:
    #     tree = proto_model.get_mst(plot=False, labels=[[i for i in range(8)], ['left', 'right']])
    #     matched_trees = trees_match(tree, ground_truth_tree)
    #     print("Trees match", matched_trees)
    #     edit_dist = graph_edit_dist(ground_truth_tree, tree)
    #     print("Edit distance", edit_dist)

    # proto_model.viz_projected_latent_space(x_test, subj_test, class_labels, proto_indices=0)
    # proto_model.viz_projected_latent_space(x_test, bolt_test, class_labels, proto_indices=0)
    # proto_model.viz_projected_latent_space(x_test, lr_test, class_labels, proto_indices=1)
    # proto_model.viz_latent_space(x_test, bolt_test, class_labels)
    # proto_model.viz_latent_space(x_test, subj_test, class_labels)
    # mean_diffs, mean_sames = proto_model.eval_proto_diffs_bolts(concept_idx=1)  # Use parity as ground truth.
    # write_to_file([y_accs, alignment, mean_diffs, mean_sames, [1] if matched_trees else [0], [edit_dist]], filename='../../saved_models/bolts_subj0_fair_' + str(model_id) + '.csv')
    tf.keras.backend.clear_session()

    for metric_name, values in metrics.items():
        print("For metric", metric_name)
        print("Mean val:", np.mean(values, axis=0))
        print("Std val:", np.std(values, axis=0))

# for metric_name, values in metrics.items():
#     print("For metric", metric_name)
#     print("All vals:", values)
