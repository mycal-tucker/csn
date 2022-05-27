import numpy as np
from tensorflow import keras
import tensorflow as tf

from src.data_parsing.adult_data import get_adult_data
from src.models.proto_model import ProtoModel
from src.utils.gpu import set_gpu_config


set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()
wass_setup = False  # FIXME: this is the thing to toggle
y_accuracy = []
s_accuracy = []
disparate_impacts = []
demographics = []
for model_id in range(20):
    np.random.seed(model_id)
    tf.random.set_seed(model_id)
    train_data, train_labels, train_protected, test_data, test_labels, test_protected = get_adult_data('../../data/adult.csv', '../../data/adult_test.csv', wass_setup=wass_setup)
    input_size = train_data.shape[1]
    protected_shape = train_protected.shape
    protected_size = 1 if len(protected_shape) == 1 else train_protected.shape[1]

    # Create one-hot encodings of data
    train_labels_one_hot = train_labels
    train_protected_one_hot = train_protected
    train_protected = train_protected_one_hot if protected_size == 1 else np.argmax(train_protected_one_hot, axis=1)
    test_labels_one_hot = test_labels
    test_labels = np.argmax(test_labels_one_hot, axis=1)
    test_protected_one_hot = test_protected
    test_protected = test_protected_one_hot if protected_size == 1 else np.argmax(test_protected_one_hot, axis=1)

    protected_size = 2 if not wass_setup else 4
    output_sizes = [2, protected_size]
    train_outputs_one_hot = [train_labels_one_hot, train_protected_one_hot]
    test_outputs = [test_labels, test_protected]
    test_outputs_one_hot = [test_labels_one_hot, test_protected_one_hot]

    classification_weight = [1, .1]
    proto_dist_weights = [1, 1]
    feature_dist_weights = [1, 1]
    # disentangle_weights = [[0, 1000], [0, 0]]
    # kl_losses = [.5, .5]
    disentangle_weights = [[0, 1000], [0, 0]]
    kl_losses = [.05, .05]
    # Seems to work best with these just at 0.
    # disentangle_weights = [[0, 0], [0, 0]]
    # kl_losses = [0, 0]
    num_epochs = 20  # Was 20
    if wass_setup:
        classification_weight = [3.0, 1.0]
        proto_dist_weights = [.001, .001]
        feature_dist_weights = [.001, .001]
        disentangle_weights = [[0, 10], [0, 0]]
        kl_losses = [10.0, 50.0]
        num_epochs = 40
    proto_model = ProtoModel(output_sizes, input_size=input_size, decode_weight=0, classification_weights=classification_weight,
                             proto_dist_weights=proto_dist_weights, feature_dist_weights=feature_dist_weights,
                             disentangle_weights=disentangle_weights, kl_losses=kl_losses, latent_dim=2)
    proto_model.train(train_data, train_data, train_outputs_one_hot, batch_size=128, epochs=num_epochs)
    y_accs, s_diff, _, _, disparate_impact, demographic, ac = proto_model.evaluate(test_data, test_data, test_outputs, test_outputs_one_hot, plot=False, do_fairness_eval=True)
    y_acc = y_accs[0]
    y_accuracy.append(y_acc)
    s_accuracy.append(s_diff)
    disparate_impacts.append(disparate_impact)
    demographics.append(demographic)
    print("Y mean", np.mean(y_accuracy))
    print("Y std", np.std(y_accuracy))
    print("S mean", np.mean(s_accuracy))
    print("S std", np.std(s_accuracy))
    print("Disparate impact mean", np.mean(disparate_impacts))
    print("Disparate impact std", np.std(disparate_impacts))
    print("Demographic mean", np.mean(demographics))
    print("Demographic std", np.std(demographics))
    proto_model.viz_latent_space(test_data, test_labels, ['Low Income', 'High Income'])
    # proto_model.viz_latent_space(test_data, test_protected, [i for i in range(protected_size)], proto_indices=1)
    # proto_model.viz_projected_latent_space(test_data, test_labels, [i for i in range(2)], proto_indices=0)
    keras.backend.clear_session()

print("Y accuracy", y_accuracy)
print("S diff", s_accuracy)
print("Disparate impacts", disparate_impacts)
print("Demographics", demographics)

print("Y mean", np.mean(y_accuracy))
print("Y std", np.std(y_accuracy))
print("S mean", np.mean(s_accuracy))
print("S std", np.std(s_accuracy))
print("Disparate impact mean", np.mean(disparate_impacts))
print("Disparate impact std", np.std(disparate_impacts))
print("Demographic mean", np.mean(demographics))
print("Demographic std", np.std(demographics))

