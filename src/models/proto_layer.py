import tensorflow as tf
from tensorflow import keras
import numpy as np


class ProtoLayer(tf.keras.layers.Layer):
    def __init__(self, num_prototypes, dim, fixed_protos=None, in_plane=True, unit_cube_init=False):
        super(ProtoLayer, self).__init__()
        self.num_prototypes = num_prototypes
        self.latent_dim = dim
        self.fixed_protos = fixed_protos
        self.in_plane = in_plane
        # Just a parameter for the special case of certain initializations
        self.unit_cube_init = unit_cube_init
        self.build(None)

    def build(self, input_shape):
        if self.fixed_protos is not None:
            print("Non trainable weights")
            self.prototypes = tf.Variable(self.fixed_protos, trainable=False, dtype=tf.float32, name='proto_kern')
        else:
            print("Trainable weights")
            initialization = 'uniform' if not self.unit_cube_init else keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
            self.prototypes = self.add_weight(name='proto_kern',
                                              shape=[self.num_prototypes, self.latent_dim],
                                              initializer=initialization,
                                              trainable=True)
        self.vector_diffs = self.get_proto_basis()
        self.basis = tf.stack(self.vector_diffs)

    @staticmethod
    def get_norms(x):
        return tf.reduce_sum(tf.pow(x, 2), axis=1)

    def call(self, feature_vectors):
        # The normal distance terms:
        # Compute the distance between x and the protos
        dists_to_protos = ProtoLayer.get_distances(feature_vectors, self.prototypes)
        dists_to_latents = ProtoLayer.get_distances(self.prototypes, feature_vectors)

        # Calculate the projected feature vectors in the subspace defined by the prototypes
        projected_features = self.get_projection_in_subspace(feature_vectors)
        dists_in_plane_protos = ProtoLayer.get_distances(projected_features, self.prototypes)
        dists_in_plane_latents = ProtoLayer.get_distances(self.prototypes, projected_features)

        # Compute the difference vector from each feature encoding to each prototype.
        diffs_to_protos = self.get_dist_to_protos(feature_vectors)
        # Or, here, compute the diffs to protos using only the components outside the plane.
        in_plane_diffs_to_protos = self.get_dist_to_protos(projected_features)
        out_of_plane_diffs = -1 * (diffs_to_protos - in_plane_diffs_to_protos)

        # The standard PCN version returns [dists_to_protos, dists_to_latents, diffs_to_protos]
        if not self.in_plane:
            return [dists_to_protos, dists_to_latents, diffs_to_protos]
        return [dists_in_plane_protos, dists_in_plane_latents, out_of_plane_diffs]

    @staticmethod
    def get_distances(tensor1, tensor2):
        t1_squared = tf.reshape(ProtoLayer.get_norms(tensor1), shape=(-1, 1))
        t2_squared = tf.reshape(ProtoLayer.get_norms(tensor2), shape=(1, -1))
        dists_between = t1_squared + t2_squared - 2 * keras.backend.dot(tensor1, tf.transpose(tensor2))
        return dists_between

    # Returns an orthogonal basis defined by the prototypes.
    def get_proto_basis(self):
        if self.latent_dim < self.num_prototypes - 1:
            print("Assuming that prototypes span the whole space, which isn't necessarily true.")
            np_array = np.zeros((self.latent_dim, self.num_prototypes - 1))
            for i in range(self.latent_dim):
                np_array[i, i] = 1
            return tf.constant(np_array, dtype=tf.float32)
        unstacked_protos = tf.unstack(self.prototypes)
        proto0 = self.prototypes[0]
        difference_vectors = []
        for i, proto in enumerate(unstacked_protos):
            if i == 0:
                continue
            difference_vectors.append(proto - proto0)
        difference_tensor = tf.stack(difference_vectors)
        print("Diff", difference_tensor)
        Q, R = tf.linalg.qr(tf.transpose(difference_tensor))
        # Q has latent_dim rows and num_protos - 1 columns
        # Another version would be to use the prototypes as the basis, but that creates one extra dimension.
        # Q, R = tf.linalg.qr(tf.transpose(self.prototypes))
        return Q

    def get_projection_in_subspace(self, features):
        # Calculate an offset to move into a space relative to the first prototype.
        # This offset effect is later undone after projecting down.
        offset = tf.tile(tf.reshape(self.prototypes[0], (1, self.latent_dim)), [tf.shape(features)[0], 1])
        relative_features = features - offset
        feature_dotted = tf.matmul(relative_features, self.basis)
        projected_features = tf.matmul(feature_dotted, tf.transpose(self.basis))
        global_projected_features = projected_features + offset
        return global_projected_features

    def get_dist_to_protos(self, feature_vectors):
        features_shaped = tf.reshape(feature_vectors, shape=(-1, self.latent_dim, 1))
        repeated_features = tf.tile(features_shaped, multiples=[1, 1, self.num_prototypes])
        transposed_prototypes = tf.transpose(self.prototypes)
        protos_shaped = tf.reshape(transposed_prototypes, shape=(1, self.latent_dim, self.num_prototypes))
        repeated_protos = tf.tile(protos_shaped, multiples=[tf.shape(feature_vectors)[0], 1, 1])
        diffs_to_protos = repeated_protos - repeated_features
        return diffs_to_protos
