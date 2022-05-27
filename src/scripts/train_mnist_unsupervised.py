import numpy as np
import tensorflow as tf

from src.data_parsing.mnist_data import get_digit_data, make_noisy, get_guided_tree
from src.models.proto_model import ProtoModel
from src.utils.eval import trees_match
from src.utils.gpu import set_gpu_config

set_gpu_config()

print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

latent_dim = 32
noise_level = 1.0
# Get the MNIST data. Do you want digit or fashion data?
x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_digit_data()

x_train_noisy = make_noisy(x_train, noise_level=noise_level)
x_test_noisy = make_noisy(x_test, noise_level=noise_level)

# Ground truth tree
ground_truth_tree = get_guided_tree()

output_sizes = [10, 3]  # Just reading off...
num_classes = len(output_sizes)
one_hot_output = [y_train_one_hot]
output = [y_test]
labels = [class_labels]
for size_idx, size in enumerate(output_sizes[1:]):
    dummy_one_hot = np.zeros((y_train_one_hot.shape[0], size))
    dummy_one_hot[:, 0] = 1
    one_hot_output.append(dummy_one_hot)
    output.append(np.zeros((y_test.shape[0])))
    labels.append(['layer' + str(size_idx) + '_dummy' + str(i) for i in range(size)])
classification_weights = [10, 0]
proto_dist_weights = [1, 0]
feature_dist_weights = [1 for _ in range(num_classes)]  # How close to prototypes are embeddings (cluster size)
disentangle_weights = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
disentangle_weights[0] = [0, -3]
kl_losses = [0 for _ in range(num_classes)]
duplication_factors = [1 for _ in range(num_classes)]

# Run a bunch of trials.
for model_id in range(0, 1):
    # Create, train, and eval the model
    proto_model = ProtoModel(output_sizes, decode_weight=10, duplication_factors=duplication_factors, input_size=784,
                             classification_weights=classification_weights, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses, latent_dim=latent_dim, in_plane_clusters=True, align_fn=tf.reduce_mean)
    # proto_model.load_model('../saved_models/', 0)
    proto_model.train(x_train_noisy, x_train, one_hot_output, batch_size=64, epochs=10)
    y_accs, s_diff, alignments, prob_updates, disparate_impact, demographic = proto_model.evaluate(x_test_noisy, x_test, output, one_hot_output)
    mst = proto_model.get_mst(add_origin=True, plot=True, labels=labels)
    tree_matches = trees_match(mst, ground_truth_tree)
    # proto_model.viz_projected_latent_space(x_test, y_test, class_labels, proto_indices=0)
    # proto_model.viz_projected_latent_space(x_test, parity_test, class_labels, proto_indices=1)
    # proto_model.viz_latent_space(x_test, y_test, class_labels)
    mean_diffs, mean_sames = proto_model.eval_proto_diffs_parity(concept_idx=1)  # Use parity as ground truth.
    tf.keras.backend.clear_session()
