import numpy as np
import tensorflow as tf

from src.data_parsing.mnist_data import get_fashion_data, make_noisy, get_fashion_tree
from src.models.proto_model import ProtoModel
from src.utils.gpu import set_gpu_config
from src.utils.eval import trees_match, graph_edit_dist
from src.utils.plotting import plot_mst
from src.utils.to_files import write_to_file

set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

latent_dim = 32
noise_level = 0.0
use_class_and_group = True
class_only = False
group_only = False
x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_fashion_data()
ground_truth_tree = get_fashion_tree()

x_train_noisy = make_noisy(x_train, noise_level=noise_level)
x_test_noisy = make_noisy(x_test, noise_level=noise_level)
# Label 	Description
# 0 	    T-shirt/top
# 1 	    Trouser
# 2 	    Pullover
# 3 	    Dress
# 4 	    Coat
# 5 	    Sandal
# 6 	    Shirt
# 7 	    Sneaker
# 8 	    Bag
# 9 	    Ankle boot
# Groups:
# 0         Shoes (5, 7, 9)
# 1         Top (0, 2, 4, 6)
# 2         Fancy (1, 3, 8)
class_to_group_mapping = {0: 1,
                          1: 2,
                          2: 1,
                          3: 2,
                          4: 1,
                          5: 0,
                          6: 1,
                          7: 0,
                          8: 2,
                          9: 0}
num_groups = len(set(class_to_group_mapping.values()))
group_train_one_hot = np.zeros((y_train.shape[0], num_groups))
for i, y in enumerate(y_train):
    group_train_one_hot[i][class_to_group_mapping.get(y)] = 1
group_test = np.zeros((y_test.shape[0]))
group_test_one_hot = np.zeros((y_test.shape[0], num_groups))
for i, y in enumerate(y_test):
    group_test[i] = class_to_group_mapping.get(y)
    group_test_one_hot[i][class_to_group_mapping.get(y)] = 1

if use_class_and_group:
    output_sizes = [10, num_groups]
    one_hot_output = [y_train_one_hot, group_train_one_hot]
    output = [y_test, group_test]
elif class_only:
    output_sizes = [10]
    one_hot_output = [y_train_one_hot]
    output = [y_test]
elif group_only:
    output_sizes = [num_groups]
    one_hot_output = [group_train_one_hot]
    output = [group_test]

classification_weights = [10] if not use_class_and_group else [10, 10]  # Mess with these weights as desired.
proto_dist_weights = [1] if not use_class_and_group else [1, 1]  # How realistic are the prototypes
feature_dist_weights = [1] if not use_class_and_group else [1, 1]  # How close to prototypes are embeddings (cluster size)
disentangle_weights = [[0 for _ in range(2)] for _ in range(2)]
disentangle_weights[0] = [0, -10]
kl_losses = [1] if not use_class_and_group else [10, 10]
duplication_factors = [1] if not use_class_and_group else [1, 1]

# Run a bunch of trials.
for model_id in range(10):
    # Create, train, and eval the model
    proto_model = ProtoModel(output_sizes, duplication_factors=duplication_factors, input_size=784,
                             classification_weights=classification_weights, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses, latent_dim=latent_dim, align_fn=tf.reduce_mean)
    # proto_model.load_model('../saved_models/', 0)
    proto_model.train(x_train_noisy, x_train, one_hot_output, batch_size=64, epochs=10)
    y_accs, s_diff, alignments, prob_updates, disparate_impact, demographic, average_cost = proto_model.evaluate(x_test_noisy, x_test, output, one_hot_output, gold_tree=(ground_truth_tree, class_labels))
    # proto_model.viz_projected_latent_space(x_test, y_test, class_labels, proto_indices=0)
    # proto_model.viz_projected_latent_space(x_test, parity_test, class_labels, proto_indices=1)
    # proto_model.viz_latent_space(x_test, y_test, class_labels)
    tree = proto_model.get_mst(plot=False, labels=[class_labels, ['shoes', 'top', 'fancy']])
    tree_matches = trees_match(tree, ground_truth_tree)
    print("Trees match", tree_matches)
    # Super fast if the trees are actually close together.
    edit_dist = graph_edit_dist(ground_truth_tree, tree)
    # edit_dist = 0
    print("Edit distance", edit_dist)
    mean_diffs, mean_sames = proto_model.eval_proto_diffs_fashion(concept_idx=1)  # Use parity as ground truth.
    # write_to_file([y_accs, alignments, mean_diffs, mean_sames, prob_updates[0], prob_updates[1]], filename='../saved_models/mnist_fashion1_' + str(model_id) + '.csv')
    # write_to_file([y_accs, alignments, mean_diffs, mean_sames, [1] if tree_matches else [0], [edit_dist], [average_cost]],
    #               filename='../saved_models/mnist_fashion_cost_' + str(model_id) + '.csv')
    # write_to_file([y_accs, alignments, mean_diffs, mean_sames, [1] if tree_matches else [0], [edit_dist], [average_cost]],
    #               filename='../saved_models/random_fashion_cost_' + str(model_id) + '.csv')
    tf.keras.backend.clear_session()
