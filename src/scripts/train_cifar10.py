import tensorflow as tf

from src.data_parsing.cifar_data import get_cifar10_data
from src.models.proto_model import ProtoModel
from src.utils.gpu import set_gpu_config

set_gpu_config()

print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

latent_dim = 40
x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_cifar10_data()

output_sizes = [10]
one_hot_output = [y_train_one_hot]
output = [y_test]

classification_weights = [1]
proto_dist_weights = [0.01]
feature_dist_weights = [0.01]
disentangle_weights = 0
kl_losses = [0.01]
duplication_factors = [2]


# Run a bunch of trials.
for model_id in range(0, 1):
    # Create, train, and eval the model
    proto_model = ProtoModel(output_sizes, decode_weight=0.1, duplication_factors=duplication_factors, input_size=32 * 32 * 3,
                             classification_weights=classification_weights, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses, latent_dim=latent_dim, in_plane_clusters=True, network_type='cifar_conv')
    # Do you want to run elastic distortions or not.
    # Elastic distortions via batches
    # proto_model.train_on_batches([x_train], one_hot_output, batch_size=250, epochs=10, input_height=32)
    # Just standard images on epoch level.
    proto_model.train(x_train, x_train, one_hot_output, epochs=100)
    y_accs, s_diff, alignments, prob_updates, disparate_impact, demographic = proto_model.evaluate(x_test, x_test, output, one_hot_output)
    proto_model.viz_projected_latent_space(x_test, y_test, class_labels, proto_indices=0)
    proto_model.viz_latent_space(x_test, y_test, class_labels)
    tf.keras.backend.clear_session()
