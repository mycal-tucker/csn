import tensorflow as tf
from tensorflow import keras


class DistProbLayer(keras.layers.Layer):
    def __init__(self, num_classes, duplication_factor):  # Note that we assume the same number of classes and prototypes.
        super(DistProbLayer, self).__init__()
        self.num_classes = num_classes
        self.duplication_factor = duplication_factor

    def call(self, inputs):
        distances, mask = inputs
        unnormalized_prob = DistProbLayer.dist_to_prob(distances)
        # If you wanted to try a nonlinear head, you could throw it in here. It just doesn't work well.
        # changed = keras.layers.Dense(self.num_classes, activation='relu')(unnormalized_prob)
        # probabilities = tf.nn.softmax(changed)

        softmaxed = tf.nn.softmax(unnormalized_prob)
        reshaped = tf.reshape(softmaxed, (-1, self.duplication_factor, self.num_classes))
        probabilities = tf.reduce_sum(reshaped, axis=1)
        masked = tf.math.multiply(mask, probabilities)
        new_denom = tf.reduce_sum(masked)
        new_probs = masked / new_denom
        return new_probs

    @staticmethod
    def dist_to_prob(dist):
        # Just return the negative distance. Taking the softmax of all the negative distances exponentiates everything
        # and then normalizes, which is exactly what I'd want to do for Gaussians centered around each prototype.
        return -dist
