import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from src.data_parsing.german_data import get_german_data
from src.models.proto_model import ProtoModel
from src.utils.gpu import set_gpu_config

set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

wass_setup = False
if wass_setup:
    num_epochs = 60
    classification_weight = [20, 1]
    proto_dist_weights = [.01, .01]
    feature_dist_weights = [.01, .01]
    disentangle_weights = [[0, 10000], [0, 0]]
    # kl_losses = [0.5, 0.5]
    kl_losses = [0.0, 10.0]
    batch_size = 256
else:  # Good results over 30 trials.
    num_epochs = 60
    classification_weight = [10, 10.0]
    proto_dist_weights = [1, 1]
    feature_dist_weights = [1, 1.0]
    # disentangle_weights = [[0, 100], [0, 0]]
    disentangle_weights = [[0, 0], [0, 0]]
    kl_losses = [0., 0.]
    batch_size = 16

y_accuracy = []
s_accuracy = []
slopes = []
disentangle = True
disparate_impacts = []
demographics = []
for model_id in range(30):
    np.random.seed(model_id)
    tf.random.set_seed(model_id)
    train_data, train_labels, train_protected, test_data, test_labels, test_protected = get_german_data('../../data/german_credit_data.csv', wass_setup=wass_setup)
    input_size = train_data.shape[1]

    # Create one-hot encodings of data
    train_labels_one_hot = keras.utils.to_categorical(train_labels, num_classes=2)
    train_protected_one_hot = keras.utils.to_categorical(train_protected)
    test_labels_one_hot = keras.utils.to_categorical(test_labels, num_classes=2)
    test_protected_one_hot = keras.utils.to_categorical(test_protected)

    num_protected_classes = train_protected_one_hot.shape[1]
    if disentangle:
        output_sizes = [2, num_protected_classes]
        train_outputs_one_hot = [train_labels_one_hot, train_protected_one_hot]
        test_outputs = [test_labels, test_protected]
        test_outputs_one_hot = [test_labels_one_hot, test_protected_one_hot]
    else:
        output_sizes = [2]  # Binary choice
        train_outputs_one_hot = [train_labels_one_hot]
        test_outputs = [test_labels]
        test_outputs_one_hot = [test_labels_one_hot]
    mean_train_labels = np.mean(train_labels)
    print("Mean test rate", mean_train_labels)
    mean_test_rate = np.mean(test_labels)
    print("Mean test rate", mean_test_rate)

    proto_model = ProtoModel(output_sizes, input_size=input_size, decode_weight=0,
                             classification_weights=classification_weight, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses)

    proto_model.train(train_data, train_data, train_outputs_one_hot, batch_size=batch_size, epochs=num_epochs)
    y_accs, s_diff, _, slope, disparate_impact, demographic, average_cost = proto_model.evaluate(test_data, test_data, test_outputs, test_outputs_one_hot, plot=False, do_fairness_eval=True)
    y_acc = y_accs[0]
    y_accuracy.append(y_acc)
    s_accuracy.append(s_diff)
    disparate_impacts.append(disparate_impact)
    demographics.append(demographic)
    slopes.append(slope)
    proto_model.viz_latent_space(test_data, test_labels, ['Good Credit', 'Bad Credit'])
    # proto_model.viz_latent_space(test_data, test_protected, [i for i in range(2)], proto_indices=1)
    keras.backend.clear_session()

    print("Y accuracy", y_accuracy)
    print("S diff", s_accuracy)
    print("Disparate impacts", disparate_impacts)
    print("Demographics", demographics)
    print("Slopes", slopes)

    print("Y mean", np.mean(y_accuracy))
    print("Y median", np.median(y_accuracy))
    print("Y std", np.std(y_accuracy))
    print("S mean", np.mean(s_accuracy))
    print("S std", np.std(s_accuracy))
    print("Slope mean", np.mean(slopes), "std", np.std(slopes))
    print("Disparate impact mean", np.mean(disparate_impacts))
    print("Disparate impact std", np.std(disparate_impacts))
    print("Demographic mean", np.mean(demographics))
    print("Demographic median", np.median(demographics))
    print("Demographic std", np.std(demographics))
