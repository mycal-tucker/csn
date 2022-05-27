import numpy as np
import tensorflow as tf
from src.models.proto_model import ProtoModel
import tensorflow.keras as keras
from src.utils.gpu import set_gpu_config

set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

factor = 1


def get_rainy_data(num_examples=10000):
    true_temp = np.zeros(num_examples)
    observed_temp = np.zeros_like(true_temp)
    true_sun = np.zeros_like(true_temp)
    observed_sun = np.zeros_like(true_sun)
    for i in range(len(true_temp)):
        t = 0.1 if np.random.random() < 0.5 else 0.9
        true_temp[i] = t
        sun = factor * t + np.random.uniform(-0.1, 0.1)
        true_sun[i] = sun
        observed_temp[i] = t + np.random.normal(0, 0.05)
        observed_sun[i] = sun + np.random.normal(0, 0.05)
    label1 = np.transpose((true_temp > 0.5) * 1)
    label2 = np.transpose((true_sun > 0.5) * 1)
    observations = np.vstack([observed_temp, observed_sun])

    labels1_one_hot = keras.utils.to_categorical(label1, num_classes=2)
    labels2_one_hot = keras.utils.to_categorical(label2, num_classes=2)
    return np.transpose(observations), label1, label2, labels1_one_hot, labels2_one_hot


classification_weight = [1, 1]
proto_dist_weights = [1, 1]
feature_dist_weights = [1, 1]
disentangle_weights = [[0 for _ in range(2)] for _ in range(2)]
disentangle_weights[0][1] = 2
kl_losses = [0, 0]

batch_size = 128
num_epochs = 30

y_accuracy = []
s_accuracy = []
slopes = []
for model_id in range(10):
    np.random.seed(model_id)
    tf.random.set_seed(model_id)
    train_data, train_labels1, train_labels2, train_labels1_one_hot, train_labels2_one_hot = get_rainy_data()
    test_data, test_labels1, test_labels2, test_labels1_one_hot, test_labels2_one_hot = get_rainy_data()

    train_outputs_one_hot = [train_labels1_one_hot, train_labels2_one_hot]
    test_outputs = [test_labels1, test_labels2]
    test_outputs_one_hot = [test_labels1_one_hot, test_labels2_one_hot]

    proto_model = ProtoModel(output_sizes=[2, 2], input_size=2, decode_weight=1,
                             classification_weights=classification_weight, proto_dist_weights=proto_dist_weights,
                             feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                             kl_losses=kl_losses, latent_dim=2)

    proto_model.train(train_data, train_data, train_outputs_one_hot, batch_size=batch_size, epochs=num_epochs)
    y_accs, s_diff, _, slope, disparate_impact, demographic, average_cost = proto_model.evaluate(test_data, test_data, test_outputs, test_outputs_one_hot, plot=False, do_fairness_eval=True)
    y_accuracy.append(y_accs)
    s_accuracy.append(s_diff)
    if slope[0] > 1:
        slope = 1 / slope[0]
    else:
        slope = slope[0]
    slopes.append(slope)
    print("slope", slope)
    print("slopes", slopes)
    print("Mean Y accuracy", np.mean(y_accuracy), "std", np.std(y_accuracy))
    # print("Mean S diff", np.mean(s_accuracy))
    print("Mean Slopes", np.mean(slopes), "std", np.std(slopes))
    # proto_model.viz_latent_space(test_data, test_labels1, ['Sunny', 'Rainy'])
    keras.backend.clear_session()

