import tensorflow as tf

from src.data_parsing.mnist_data import get_digit_data
from src.models.proto_model import ProtoModel
from src.utils.gpu import set_gpu_config

set_gpu_config()

print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

latent_dim = 40
x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_digit_data()

output_sizes = [10]
one_hot_output = [y_train_one_hot]
output = [y_test]

classification_weights = [20]
proto_dist_weights = [1]
feature_dist_weights = [1]
disentangle_weights = 0
kl_losses = [0.01]
duplication_factors = [2]


# Run a bunch of trials.
for model_id in range(0, 1):
    # Create, train, and eval the model
    proto_model = ProtoModel(output_sizes, duplication_factors=duplication_factors, input_size=784,
                             classification_weights=classification_weights, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses, latent_dim=latent_dim, in_plane_clusters=True, network_type='mnist_conv')
    proto_model.train_on_batches([x_train], one_hot_output, batch_size=250, epochs=5)
    y_accs, _, alignments, prob_updates = proto_model.evaluate(x_test, x_test, output, one_hot_output)
    proto_model.viz_projected_latent_space(x_test, y_test, class_labels, proto_indices=0)
    proto_model.viz_latent_space(x_test, y_test, class_labels)
    tf.keras.backend.clear_session()
