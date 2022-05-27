import numpy as np
import tensorflow as tf

from src.data_parsing.mnist_data import get_digit_data, make_noisy
from src.models.proto_model import ProtoModel
from src.utils.gpu import set_gpu_config


set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

latent_dim = 32
noise_level = 0.0
x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_digit_data()

x_train_noisy = make_noisy(x_train, noise_level=noise_level)
x_test_noisy = make_noisy(x_test, noise_level=noise_level)
parity_train_one_hot = np.zeros((y_train.shape[0], 2))
for i, y in enumerate(y_train):
    parity_train_one_hot[i][y % 2] = 1
parity_test = np.zeros((y_test.shape[0]))
parity_test_one_hot = np.zeros((y_train.shape[0], 2))
for i, y in enumerate(y_test):
    parity_test[i] = y % 2
    parity_test_one_hot[i][y % 2] = 1

# output_sizes = [10]
# one_hot_output = [y_train_one_hot]
# output = [y_test]

# if use_digit_and_parity:
#     output_sizes = [10, 2]
#     one_hot_output = [y_train_one_hot, parity_train_one_hot]
#     output = [y_test, parity_test]
# elif digit_only:
#
# elif parity_only:
#     output_sizes = [2]
#     one_hot_output = [parity_train_one_hot]
#     output = [parity_test]

# First train on only digit
# classification_weights = [10]
# proto_dist_weights = [1]
# feature_dist_weights = [1]
# disentangle_weight = 0
# kl_losses = [.1]
# duplication_factors = [1]

digit_accuracy = []
parity_accuracy = []
# Run a bunch of trials.
for model_id in range(2):
    # First train on only digit
    classification_weights = [10]
    proto_dist_weights = [1]
    feature_dist_weights = [1]
    disentangle_weights = 0
    kl_losses = [.1]
    duplication_factors = [1]

    output_sizes = [10]
    one_hot_output = [y_train_one_hot]
    output = [y_test]
    # Create, train, and eval the model
    proto_model = ProtoModel(output_sizes, duplication_factors=duplication_factors, input_size=784,
                             classification_weights=classification_weights, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses, latent_dim=latent_dim)
    proto_model.train(x_train_noisy, x_train, one_hot_output, batch_size=64, epochs=10)
    # proto_model.evaluate(x_test_noisy, x_test, output, one_hot_output)
    # proto_model.viz_projected_latent_space(x_test, y_test, class_labels, proto_indices=0)
    # proto_model.viz_latent_space(x_test, y_test, class_labels)

    # Now freeze the model and add a concept net for parity.
    # FREEEEZE
    proto_model.encoder.trainable = False
    proto_model.proto_layers[0].trainable = False
    proto_model.classifier_layers[0].trainable = False
    proto_model.classification_weights.append(10)
    proto_model.proto_dist_weights.append(1)
    proto_model.feature_dist_weights.append(1)
    proto_model.kl_losses.append(0.1)
    proto_model.output_sizes.append(2)

    parity_proto_layer, parity_classifier, parity_label = proto_model.create_new_parts(2, duplication_factor=5)
    proto_model.proto_layers.append(parity_proto_layer)
    proto_model.classifier_layers.append(parity_classifier)
    proto_model.label_layers.append(parity_label)
    proto_model.build_overall_network()

    one_hot_output = [y_train_one_hot, parity_train_one_hot]
    output = [y_test, parity_test]
    proto_model.train(x_train_noisy, x_train, one_hot_output, batch_size=64, epochs=1)
    accuracies, _ = proto_model.evaluate(x_test_noisy, x_test, output, one_hot_output)
    # proto_model.viz_projected_latent_space(x_test, y_test, class_labels, proto_indices=0)
    # proto_model.viz_projected_latent_space(x_test, y_test, class_labels, proto_indices=1)
    digit_accuracy.append(accuracies[0])
    parity_accuracy.append(accuracies[1])

    tf.keras.backend.clear_session()

print("dig accuracy", digit_accuracy)
print("par accuracy", parity_accuracy)

print("dig mean", np.mean(digit_accuracy))
print("dig std", np.std(digit_accuracy))
print("par mean", np.mean(parity_accuracy))
print("par std", np.std(parity_accuracy))
