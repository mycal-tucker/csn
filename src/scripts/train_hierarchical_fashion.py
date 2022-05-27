# Mycal's recreation of the hierarchical prototype network training idea.

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from src.data_parsing.mnist_data import get_fashion_data, make_noisy, get_fashion_tree
from src.models.proto_model import ProtoModel
from src.utils.to_files import write_to_file
from src.utils.gpu import set_gpu_config
from src.utils.plotting import plot_mst
from src.utils.eval import trees_match, graph_edit_dist


set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

latent_dim = 32
noise_level = 0.0
x_train, label_train, label_train_one_hot, x_test, label_test, label_test_one_hot, class_labels = get_fashion_data()
ground_truth_tree = get_fashion_tree()
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
group_to_class = {}
for class_id, group_id in class_to_group_mapping.items():
    if group_to_class.get(group_id) is None:
        group_to_class[group_id] = []
    group_to_class[group_id].append(class_id)

inputs_train = [x_train, x_train]
outputs_train = [label_train]
group_label_train = np.zeros_like(label_train)
fine_masks_train = np.zeros_like(label_train_one_hot)
for i, fine_label in enumerate(label_train):
    group_label = class_to_group_mapping.get(fine_label)
    group_label_train[i] = group_label
    for fine_candidate in group_to_class.get(group_label):
        fine_masks_train[i, fine_candidate] = 1
outputs_train.append(group_label_train)
masks_train = [fine_masks_train, np.ones((x_train.shape[0], 3))]
one_hot_output_train = [keras.utils.to_categorical(entry) for entry in outputs_train]
# Same but for test data
outputs_test = [label_test]
group_label_test = np.zeros_like(label_test)
fine_masks_test = np.zeros_like(label_test_one_hot)
for i, fine_label in enumerate(label_test):
    group_label = class_to_group_mapping.get(fine_label)
    group_label_test[i] = group_label
    for fine_candidate in group_to_class.get(group_label):
        fine_masks_test[i, fine_candidate] = 1
outputs_test.append(group_label_test)
masks_test = [fine_masks_test, np.ones((x_test.shape[0], 3))]
one_hot_output_test = [keras.utils.to_categorical(entry) for entry in outputs_test]


output_sizes = [10, 3]
classification_weights = [10, 10]  # Mess with these weights as desired.
proto_dist_weights = [1, 0]  # How realistic are the prototypes
feature_dist_weights = [1, 1]  # How close to prototypes are embeddings (cluster size)
disentangle_weights = 0
kl_losses = [0.0, 0.0]  # Similarly, no KL regularization
duplication_factors = [1, 1]

for model_id in range(10):
    proto_model = ProtoModel(output_sizes, duplication_factors=duplication_factors, input_size=784,
                             classification_weights=classification_weights, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses, latent_dim=latent_dim, in_plane_clusters=False,
                             use_shadow_basis=False, align_fn=tf.reduce_mean)
    proto_model.train_hierarchical(inputs_train, one_hot_output_train, masks_train, batch_size=64, epochs=60)
    y_accs, alignments, average_cost = proto_model.evaluate_hierarchical(x_test, x_test, outputs_test, one_hot_output_test, masks=masks_test, coarse_to_fine=group_to_class, gold_tree=(ground_truth_tree, class_labels))
    # proto_model.viz_projected_latent_space(x_test, label_test, class_labels, proto_indices=0)
    # proto_model.viz_projected_latent_space(x_test, group_label_test, class_labels, proto_indices=1)
    # proto_model.viz_projected_latent_space(x_test, shoes_test_y, class_labels, proto_indices=1)
    # proto_model.viz_latent_space(x_test, group_test, class_labels)
    mst = proto_model.get_mst(add_origin=True, plot=False, labels=[['shoes', 'tops', 'fancy'], class_labels])
    tree_matches = trees_match(mst, ground_truth_tree)
    print("Tree matches", tree_matches)
    edit_dist = graph_edit_dist(ground_truth_tree, mst)
    print("Edit distance", edit_dist)
    # Measure the differences between the even and odd prototypes, and the digit prototypes.
    mean_diffs, mean_sames = proto_model.eval_proto_diffs_fashion(concept_idx=1)

    # write_to_file([accuracies, alignments, mean_diffs, mean_sames, prob_updates[1], prob_updates[0]], filename='../saved_models/hierarchy_fashion_protos_' + str(model_id) + '.csv')
    write_to_file([y_accs, alignments, mean_diffs, mean_sames, [1] if tree_matches else [0], [edit_dist], [average_cost]],
                  filename='../saved_models/hierarchy_fashion_cost_' + str(model_id) + '.csv')

    tf.keras.backend.clear_session()

