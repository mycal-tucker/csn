import numpy as np
import tensorflow as tf

from src.data_parsing.cifar_data import get_cifar100_data, get_cifar100_tree
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
x_train, y_train_fine, y_train_coarse, y_train_fine_one_hot, y_train_coarse_one_hot, \
x_test, y_test_fine, y_test_coarse, y_test_fine_one_hot, y_test_coarse_one_hot, fine_labels, coarse_labels = get_cifar100_data()
ground_truth_tree = get_cifar100_tree()

output_sizes = [100, 20]
one_hot_output = [y_train_fine_one_hot, y_train_coarse_one_hot]
output = [y_test_fine, y_test_coarse]

classification_weights = [5, 1]
proto_dist_weights = [0.01, 0.0]
feature_dist_weights = [0.1, 0.1]
disentangle_weights = [[0, -10], [-10, 0]]
kl_losses = [0.01, 0.01]
duplication_factors = [1, 1]


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
    # Just standard images on epoch level.
    proto_model.train(x_train, x_train, one_hot_output, batch_size=32, epochs=0)
    tree = proto_model.get_mst(plot=False, labels=[fine_labels, coarse_labels])
    tree_matches = trees_match(tree, ground_truth_tree)
    print("Trees match", tree_matches)
    edit_dist = graph_edit_dist(ground_truth_tree, tree)
    # edit_dist = 0
    print("Edit distance", edit_dist)
    y_accs, s_diff, alignments, prob_updates, disparate_impact, demographic, average_cost = proto_model.evaluate(x_test, x_test, output, one_hot_output, gold_tree=(ground_truth_tree, fine_labels))
    # proto_model.viz_projected_latent_space(x_test, y_test_fine, fine_labels, proto_indices=0)
    # proto_model.viz_latent_space(x_test, y_test_fine, fine_labels)
    mean_diffs, mean_sames = proto_model.eval_proto_diffs_parity(concept_idx=1)
    write_to_file([y_accs, alignments, mean_diffs, mean_sames, [1] if tree_matches else [0], [edit_dist], [average_cost]], filename='../saved_models/cifar100_csn_init_cost_' + str(model_id) + '.csv')
    tf.keras.backend.clear_session()
