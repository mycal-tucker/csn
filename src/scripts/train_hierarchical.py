# Mycal's recreation of the hierarchical prototype network training idea.

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from src.data_parsing.mnist_data import get_digit_data, make_noisy, get_parity_tree
from src.models.proto_model import ProtoModel
from src.utils.to_files import write_to_file
from src.utils.gpu import set_gpu_config
from src.utils.eval import trees_match, graph_edit_dist


set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

# Set seed before getting the data.
latent_dim = 32
noise_level = 0.0
x_train, digit_train, digit_train_one_hot, x_test, digit_test, digit_test_one_hot, class_labels = get_digit_data()
output_sizes = [10, 2]
classification_weights = [1, 1]  # Mess with these weights as desired.
proto_dist_weights = [1, 0]  # How realistic are the prototypes
feature_dist_weights = [1, 1]  # How close to prototypes are embeddings (cluster size)
disentangle_weights = 0  # They obviously don't use orthogonality.
kl_losses = [0.0, 0.0]  # Similarly, no KL regularization
duplication_factors = [1, 1]

coarse_to_fine = {0: [], 1: []}
for i in range(10):
    coarse_to_fine[i % 2].append(i)

# Ground truth tree
ground_truth_tree = get_parity_tree()
for model_id in range(6, 10):
    np.random.seed(model_id)
    tf.random.set_seed(model_id)

    x_train_noisy = make_noisy(x_train, noise_level=noise_level)
    x_test_noisy = make_noisy(x_test, noise_level=noise_level)
    parity_train_one_hot = np.zeros((digit_train.shape[0], 2))
    for i, y in enumerate(digit_train):
        parity_train_one_hot[i][y % 2] = 1
    parity_test = np.zeros((digit_test.shape[0]))
    parity_test_one_hot = np.zeros((digit_test.shape[0], 2))
    for i, y in enumerate(digit_test):
        parity_test[i] = y % 2
        parity_test_one_hot[i][y % 2] = 1

    inputs_train = [x_train, x_train]
    one_hot_output_train = [digit_train_one_hot, parity_train_one_hot]
    outputs_test = [digit_test, parity_test]
    one_hot_outputs_test = [keras.utils.to_categorical(entry) for entry in outputs_test]

    # Make the masks.
    masks_train = []
    masks_digit = np.zeros((digit_train.shape[0], 10))
    for i, dig in enumerate(digit_train):
        parity = dig % 2
        for candidate_dig in range(10):
            if candidate_dig % 2 == parity:
                masks_digit[i, candidate_dig] = 1
    masks_train.append(masks_digit)
    masks_train.append(np.ones((x_train.shape[0], 2)))

    masks_test = []
    masks_dig_test = np.zeros((digit_test.shape[0], 10))
    for i, dig in enumerate(digit_test):
        parity = dig % 2
        for candidate_dig in range(10):
            if candidate_dig % 2 == parity:
                masks_dig_test[i, candidate_dig] = 1
    masks_test.append(masks_dig_test)
    masks_test.append(np.ones((x_test.shape[0], 2)))

    proto_model = ProtoModel(output_sizes, duplication_factors=duplication_factors, input_size=784,
                             classification_weights=classification_weights, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses, latent_dim=latent_dim, in_plane_clusters=False, align_fn=tf.reduce_mean, network_type='dense_mnist')
    proto_model.train_hierarchical(inputs_train, one_hot_output_train, masks_train, batch_size=128, epochs=20)
    y_accs, alignments, average_cost = proto_model.evaluate_hierarchical(x_test, x_test, outputs_test, one_hot_outputs_test,
                                                           masks=masks_test, coarse_to_fine=coarse_to_fine, gold_tree=(ground_truth_tree, class_labels))
    # y_accs, alignments = proto_model.evaluate_hierarchical(x_test, x_test, outputs_test, one_hot_outputs_test, masks=None, gold_tree=(ground_truth_tree, class_labels))
    mst = proto_model.get_mst(add_origin=True, plot=False, labels=[['even', 'odd'], class_labels])
    mean_diffs, mean_sames = proto_model.eval_proto_diffs_parity(concept_idx=1)
    tree_matches = trees_match(mst, ground_truth_tree)
    print("Tree matches", tree_matches)
    edit_dist = graph_edit_dist(ground_truth_tree, mst)
    # edit_dist = 0
    print("Edit distance", edit_dist)
    write_to_file([y_accs, alignments, mean_diffs, mean_sames, [1] if tree_matches else [0], [edit_dist], [average_cost]],
                  filename='../saved_models/hierarchy_mnist_protos_cost_' + str(model_id) + '.csv')
    tf.keras.backend.clear_session()

