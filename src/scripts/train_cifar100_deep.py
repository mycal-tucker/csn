import numpy as np
import tensorflow as tf

from src.data_parsing.cifar_data import get_deep_data, get_deep_tree
from src.models.proto_model import ProtoModel
from src.utils.eval import trees_match, graph_edit_dist
from src.utils.gpu import set_gpu_config
from src.utils.to_files import write_to_file

set_gpu_config()

print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()
# Set seed before getting the data.
np.random.seed(0)
tf.random.set_seed(0)

latent_dim = 100
x_train, y_train0, y_train0_one_hot, y_train1, y_train1_one_hot, y_train2, y_train2_one_hot, \
y_train3, y_train3_one_hot, y_train4, y_train4_one_hot, \
x_test, y_test0, y_test0_one_hot, y_test1, y_test1_one_hot, y_test2, y_test2_one_hot, \
y_test3, y_test3_one_hot, y_test4, y_test4_one_hot, all_labels = get_deep_data()

_, _, ground_truth_tree = get_deep_tree()

output_sizes = [100, 20, 8, 4, 2]
one_hot_output = [y_train0_one_hot, y_train1_one_hot, y_train2_one_hot, y_train3_one_hot, y_train4_one_hot]
test_output = [y_test0, y_test1, y_test2, y_test3, y_test4]
#
classification_weights = [1 for _ in output_sizes]
classification_weights[0] = 5
proto_dist_weights = [0 for _ in output_sizes]
proto_dist_weights[0] = 0.1
feature_dist_weights = [0.1 for _ in output_sizes]
disentangle_weights = [[0 for _ in output_sizes] for _ in output_sizes]
disentangle_weights[0] = [0, -1, -1, -1, -1]
disentangle_weights[1] = [0, 0, -1, -1, -1]
disentangle_weights[2] = [0, 0, 0, -1, -1]
disentangle_weights[3] = [0, 0, 0, 0, -1]
kl_losses = [0 for _ in output_sizes]
duplication_factors = [1 for _ in output_sizes]


# Run a bunch of trials.
for model_id in range(0, 10):
    np.random.seed(model_id)
    tf.random.set_seed(model_id)
    # Create, train, and eval the model
    proto_model = ProtoModel(output_sizes, decode_weight=0.1, duplication_factors=duplication_factors, input_size=32 * 32 * 3,
                             classification_weights=classification_weights, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses, latent_dim=latent_dim, in_plane_clusters=True, network_type='resnet',
                             align_fn=tf.reduce_mean)
    proto_model.train(x_train, x_train, one_hot_output, batch_size=32, epochs=0)
    tree = proto_model.get_deep_mst(plot=False, labels=all_labels)
    tree_matches = trees_match(tree, ground_truth_tree)
    print("Trees match", tree_matches)
    edit_dist = graph_edit_dist(ground_truth_tree, tree)
    # edit_dist = 0
    print("Edit distance", edit_dist)
    y_accs, s_diff, alignments, prob_updates, disparate_impact, demographic, average_cost = proto_model.evaluate(x_test, x_test, test_output, one_hot_output, gold_tree=(ground_truth_tree, all_labels[0]))
#     # proto_model.viz_projected_latent_space(x_test, y_test_fine, fine_labels, proto_indices=0)
#     # proto_model.viz_latent_space(x_test, y_test_fine, fine_labels)
    mean_diffs, mean_sames = proto_model.eval_proto_diffs_parity(concept_idx=1)
    write_to_file([y_accs, alignments, mean_diffs, mean_sames, [1] if tree_matches else [0], [edit_dist], [average_cost]], filename='../saved_models/cifar100_csn_init_deep_' + str(model_id) + '.csv')
    tf.keras.backend.clear_session()
