import networkx as nx
import numpy as np
import tensorflow as tf
from classification_models.tfkeras import Classifiers
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import seaborn as sn
import cv2
import pickle as pkl
import random
from scipy import stats

from src.data_parsing.mnist_data import batch_elastic_transform
from src.models.dist_prob_layer import DistProbLayer
from src.models.proto_layer import ProtoLayer
from src.utils.plotting import plot_latent_prompt, plot_rows_of_images, plot_mst


class ProtoModel:
    def __init__(self, output_sizes, duplication_factors=None, input_size=784, decode_weight=1,
                 classification_weights=None, proto_dist_weights=None, feature_dist_weights=None,
                 disentangle_weights=None, kl_losses=None, latent_dim=32, proto_grids=None, in_plane_clusters=True,
                 network_type='dense', use_shadow_basis=False, align_fn=tf.reduce_max):
        self.decode_weight = decode_weight
        self.latent_dim = latent_dim
        self.output_sizes = output_sizes
        self.in_plane_clusters = in_plane_clusters
        # The function used for measuring alignment. tf.reduce_max good for penalizing, tf.reduce_min good for
        # encouraging.
        self.align_fn = align_fn
        if duplication_factors is None:
            self.duplication_factors = [1 for _ in range(len(output_sizes))]
        else:
            self.duplication_factors = duplication_factors
        self.classification_weights = []
        self.proto_dist_weights = []
        self.feature_dist_weights = []
        self.kl_losses = []
        # Should be a list of lists, where all lists have size n.
        if disentangle_weights is None or disentangle_weights == 0:
            disentangle_weights = []
            for _ in range(len(output_sizes)):
                disentangle_weights.append([0 for _ in range(len(output_sizes))])
        self.disentangle_weights = disentangle_weights
        for i, output_size in enumerate(output_sizes):
            if classification_weights is not None:
                self.classification_weights.append(classification_weights[i])
            else:
                self.classification_weights.append(1)
            if proto_dist_weights is not None:
                self.proto_dist_weights.append(proto_dist_weights[i])
            else:
                self.proto_dist_weights.append(1)
            if feature_dist_weights is not None:
                self.feature_dist_weights.append(feature_dist_weights[i])
            else:
                self.feature_dist_weights.append(1)
            if kl_losses is not None:
                self.kl_losses.append(kl_losses[i])
            else:
                self.kl_losses.append(0)
        self.input_size = input_size
        self.use_shadow_basis = use_shadow_basis
        # Build a dict for each configuration.
        # Goes encoder fn, decoder fn, optimizer, callbacks
        # Set a changeable learning rate for the ResNet cifar100 setup.
        network_configs = {
            'dense_mnist': (self.build_network_parts, self.build_standard_dec, keras.optimizers.Adam(), []),
            'dense': (self.build_network_parts, self.build_standard_dec, keras.optimizers.Adam(),
                      [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.01,
                                                         patience=3,
                                                         min_lr=1e-6),
                       keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                                     patience=10)]),
            'mnist_conv': (self.build_mnist_conv_net_parts, self.build_mnist_conv_dec, keras.optimizers.Adam(), None),
            'cifar_conv': (self.build_cifar_conv_net_parts, self.build_cifar_conv_dec, keras.optimizers.Adam(), None),
            'resnet': (self.build_resnet_parts, self.build_cifar_conv_dec,
                       keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False),
                       [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                          factor=0.01,
                                                          patience=3,
                                                          min_lr=1e-6),
                        keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)]
                       )}

        self.enc_fn, self.dec_fn, self.optimizer, self.callbacks = network_configs.get(network_type)
        self.dec_fn()
        self.proto_layers, self.classifier_layers, self.label_layers, self.dist_enc, self.encoder = self.enc_fn(
            output_sizes)
        self.build_overall_network()

    def build_network_parts(self, output_sizes):
        proto_layers = []
        classifier_layers = []
        label_layers = []
        for i, output_size in enumerate(output_sizes):
            proto_layers.append(ProtoLayer(output_size * self.duplication_factors[i], self.latent_dim,
                                           in_plane=self.in_plane_clusters))
            if output_size > 1:
                classifier_layers.append(DistProbLayer(output_size, self.duplication_factors[i]))
            else:
                classifier_layers.append(layers.Dense(output_size, activation='sigmoid'))
            label_layers.append(keras.Input(shape=(output_size,)))

        img_input = keras.Input(shape=(self.input_size,), name='img_input')
        dense = layers.Dense(128, activation="relu")
        x = dense(img_input)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        feature_enc_mean = layers.Dense(self.latent_dim, activation="linear")(x)
        feature_enc_log_var = layers.Dense(self.latent_dim, activation="linear")(x)
        feature_enc = layers.Lambda(ProtoModel.sampling, output_shape=(self.latent_dim,), name='z')(
            [feature_enc_mean, feature_enc_log_var])
        dist_enc = keras.Model(inputs=img_input, outputs=[feature_enc_mean, feature_enc_log_var], name='enc_params')
        encoder = keras.Model(inputs=img_input, outputs=feature_enc, name='encoder')
        return proto_layers, classifier_layers, label_layers, dist_enc, encoder

    def build_mnist_conv_net_parts(self, output_sizes):
        proto_layers = []
        classifier_layers = []
        label_layers = []
        for i, output_size in enumerate(output_sizes):
            proto_layers.append(ProtoLayer(output_size * self.duplication_factors[i], self.latent_dim,
                                           in_plane=self.in_plane_clusters, unit_cube_init=True))
            if output_size > 1:
                classifier_layers.append(DistProbLayer(output_size, self.duplication_factors[i]))
            else:
                classifier_layers.append(layers.Dense(output_size, activation='sigmoid'))
            label_layers.append(keras.Input(shape=(output_size,)))

        img_input = keras.Input(self.input_size, name='img_input')
        reshaped = layers.Reshape((28, 28, 1))(img_input)
        x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(reshaped)
        x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.Conv2D(10, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(x)
        x = layers.Flatten()(x)
        feature_enc_mean = layers.Dense(self.latent_dim, activation="linear")(x)
        feature_enc_log_var = layers.Dense(self.latent_dim, activation="linear")(x)
        feature_enc = layers.Lambda(ProtoModel.sampling, output_shape=(self.latent_dim,), name='z')(
            [feature_enc_mean, feature_enc_log_var])
        dist_enc = keras.Model(inputs=img_input, outputs=[feature_enc_mean, feature_enc_log_var], name='enc_params')
        encoder = keras.Model(inputs=img_input, outputs=feature_enc, name='encoder')
        return proto_layers, classifier_layers, label_layers, dist_enc, encoder

    def build_cifar_conv_net_parts(self, output_sizes):
        proto_layers = []
        classifier_layers = []
        label_layers = []
        for i, output_size in enumerate(output_sizes):
            proto_layers.append(ProtoLayer(output_size * self.duplication_factors[i], self.latent_dim,
                                           in_plane=self.in_plane_clusters))
            if output_size > 1:
                classifier_layers.append(DistProbLayer(output_size, self.duplication_factors[i]))
            else:
                classifier_layers.append(layers.Dense(output_size, activation='sigmoid'))
            label_layers.append(keras.Input(shape=(output_size,)))

        img_input = keras.Input(shape=(self.input_size,), name='img_input')
        reshaped = layers.Reshape((32, 32, 3), name='cifar_input')(img_input)
        x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(reshaped)
        x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(x)
        x = layers.MaxPool2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(x)
        x = layers.MaxPool2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='sigmoid')(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        feature_enc_mean = layers.Dense(self.latent_dim, activation="linear")(x)
        feature_enc_log_var = layers.Dense(self.latent_dim, activation="linear")(x)
        feature_enc = layers.Lambda(ProtoModel.sampling, output_shape=(self.latent_dim,), name='z')(
            [feature_enc_mean, feature_enc_log_var])
        dist_enc = keras.Model(inputs=img_input, outputs=[feature_enc_mean, feature_enc_log_var], name='enc_params')
        encoder = keras.Model(inputs=img_input, outputs=feature_enc, name='encoder')
        return proto_layers, classifier_layers, label_layers, dist_enc, encoder

    def build_resnet_parts(self, output_sizes):
        proto_layers = []
        classifier_layers = []
        label_layers = []
        for i, output_size in enumerate(output_sizes):
            proto_layers.append(ProtoLayer(output_size * self.duplication_factors[i], self.latent_dim,
                                           in_plane=self.in_plane_clusters))
            classifier_layers.append(DistProbLayer(output_size, self.duplication_factors[i]))
            label_layers.append(keras.Input(shape=(output_size,)))

        img_input = keras.Input(shape=(self.input_size,), name='img_input')
        reshaped = layers.Reshape((32, 32, 3), name='cifar_input')(img_input)
        x = reshaped
        x = layers.UpSampling2D()(x)
        x = layers.UpSampling2D()(x)
        x = layers.UpSampling2D()(x)
        # There are a bunch of possible base networks to choose from, such as ResNet50, ResNet18, or VGG19
        # my_resnet = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        # my_resnet = keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
        # keras by default doesn't have ResNet18, but an imported module does.
        # See here: https://awesomeopensource.com/project/qubvel/classification_models
        ResNetClass, preprocess_input = Classifiers.get('resnet18')
        my_resnet = ResNetClass(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        # Pass the upsampled input through the net, then through a couple of dense layers.
        res_output = my_resnet(x)
        res_output = layers.GlobalAveragePooling2D()(res_output)
        x = layers.Dense(4096, activation='relu')(res_output)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(4096, activation='relu')(x)
        feature_enc_mean = layers.Dense(self.latent_dim, activation="linear")(x)
        feature_enc_log_var = layers.Dense(self.latent_dim, activation="linear")(x)
        feature_enc = layers.Lambda(ProtoModel.sampling, output_shape=(self.latent_dim,), name='z')(
            [feature_enc_mean, feature_enc_log_var])
        dist_enc = keras.Model(inputs=img_input, outputs=[feature_enc_mean, feature_enc_log_var], name='enc_params')
        encoder = keras.Model(inputs=img_input, outputs=feature_enc, name='encoder')
        return proto_layers, classifier_layers, label_layers, dist_enc, encoder

    def build_standard_dec(self):
        dec_input = keras.Input(shape=(self.latent_dim,), name='dec_input')
        dec_l1 = layers.Dense(128, activation='relu', name='dec1')
        dec_l2 = layers.Dense(128, activation='relu', name='dec2')
        dec_l3 = layers.Dense(self.input_size, activation='sigmoid', name='recons')
        dec_output = dec_l1(dec_input)
        dec_output = dec_l2(dec_output)
        dec_output = dec_l3(dec_output)
        self.decoder = keras.Model(inputs=dec_input, outputs=dec_output, name='dec_model')

    def get_decoding(self, feature_enc):
        return self.decoder(feature_enc)

    def build_mnist_conv_dec(self, feature_enc, dec_input):
        d1 = layers.Reshape((2, 2, int(self.latent_dim / 4),))
        d2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', output_padding=(1, 1),
                                    activation='relu')
        d3 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', output_padding=(0, 0),
                                    activation='relu')
        d4 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')
        d5 = layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')
        d6 = layers.Reshape((self.input_size,), name='finalreshape')
        dec_output = d1(dec_input)
        dec_output = d1(dec_output)
        dec_output = d2(dec_output)
        dec_output = d3(dec_output)
        dec_output = d4(dec_output)
        dec_output = d5(dec_output)
        dec_output = d6(dec_output)
        self.decoder = keras.Model(inputs=dec_input, outputs=dec_output, name='dec_model')

    def build_cifar_conv_dec(self, feature_enc, dec_input):
        d1 = layers.Reshape((2, 2, int(self.latent_dim / 4),))
        d2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', output_padding=(1, 1),
                                    activation='relu')
        d3 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', output_padding=(1, 1),
                                    activation='relu')
        d4 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), output_padding=(1, 1), padding='same',
                                    activation='relu')
        d5 = layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), output_padding=(1, 1), padding='same',
                                    activation='sigmoid')
        d6 = layers.Reshape((self.input_size,), name='finalreshape')
        d = d1(feature_enc)
        d = d2(d)
        d = d3(d)
        d = d4(d)
        d = d5(d)
        recons = d6(d)
        dec_output = d1(dec_input)
        dec_output = d1(dec_output)
        dec_output = d2(dec_output)
        dec_output = d3(dec_output)
        dec_output = d4(dec_output)
        dec_output = d5(dec_output)
        dec_output = d6(dec_output)
        self.decoder = keras.Model(inputs=dec_input, outputs=dec_output, name='dec_model')
        self.decoder.summary()
        return recons

    def build_overall_network(self):
        feature_enc = self.encoder.outputs[0]
        feature_enc_mean = self.dist_enc.outputs[0]
        feature_enc_log_var = self.dist_enc.outputs[1]

        # Create a new shadow prototype layer just to get orthogonality between first set of prototypes and
        # the basis for all the last prototypes.
        self.shadow_basis = None
        if self.use_shadow_basis and len(self.proto_layers) > 1:
            all_other_protos = []
            for proto_layer in self.proto_layers:
                all_other_protos.extend(tf.unstack(proto_layer.prototypes))
            # Now that we have all the other prototypes, get the basis
            proto0 = all_other_protos[0]
            difference_vectors = []
            for i, proto in enumerate(all_other_protos):
                if i == 0:
                    continue
                difference_vectors.append(proto - proto0)
            diff_tensor = tf.stack(difference_vectors)
            Q, R = tf.linalg.qr(tf.transpose(diff_tensor))
            self.shadow_basis = Q

        self.label_mask_layers = []
        for output_size in self.output_sizes:
            self.label_mask_layers.append(keras.layers.Input(shape=(output_size,)))

        # Stuff for predicting the digit
        dists_to_protos = []
        mean_dists_to_protos = []
        dists_to_latents = []
        classification_preds = []
        diffs_to_protos = []
        mean_diffs_to_protos = []
        for i, proto_layer in enumerate(self.proto_layers):
            # Is it the sampled encoding or the mean of the encoding distribution that you pass in for prediction?
            # Passing the sampled encoding seems better. Slightly worse classification accuracy, but better recons.
            print("Proto layer", proto_layer)
            dist_to_protos, dist_to_latents, diff_to_protos = proto_layer(feature_enc)
            mean_dist_to_protos, _, mean_diff_to_protos = proto_layer(feature_enc_mean)
            classification_pred = self.classifier_layers[i]([dist_to_protos, self.label_mask_layers[i]])
            dists_to_protos.append(dist_to_protos)
            mean_dists_to_protos.append(mean_dist_to_protos)
            dists_to_latents.append(dist_to_latents)
            diffs_to_protos.append(diff_to_protos)
            mean_diffs_to_protos.append(mean_diff_to_protos)
            classification_preds.append(classification_pred)
        self.predictors = []
        for i, proto_layer in enumerate(self.proto_layers):
            new_input = layers.Input(shape=(self.latent_dim,), name='enc_input')
            dist_to_protos, dist_to_latents, _ = proto_layer(new_input)
            classification_pred = self.classifier_layers[i]([dist_to_protos, self.label_mask_layers[i]])
            self.predictors.append(
                keras.Model([new_input, self.label_mask_layers[i]], classification_pred, name='predictor' + str(i)))
        self.projectors = []
        for i, proto_layer in enumerate(self.proto_layers):
            new_input = layers.Input(shape=(self.latent_dim,), name='projector_input')
            dist_to_protos, dist_to_latents, diff_to_protos = proto_layer(new_input)
            in_plane_point = new_input - tf.reduce_mean(diff_to_protos, axis=2)  # + proto_layer.prototypes[0]
            self.projectors.append(
                keras.Model(new_input, [dist_to_protos, dist_to_latents, diff_to_protos, in_plane_point],
                            name='projector' + str(i)))
        # Stuff for getting the angles between vectors in the sets of prototypes.
        self.vector_diffs = []  # Basis describing the  hyperplane for each set of prototypes.
        for set_idx, proto_set in enumerate(self.proto_layers):
            self.vector_diffs.append(proto_set.vector_diffs)
        if self.shadow_basis is not None:
            print("Adding shadow basis")
            self.vector_diffs.append(self.shadow_basis)
        # Take the dot product of each of vectors in the bases of each set of protos to compare orthogonality of sets.
        self.proto_set_alignment = []
        for i, vector_set1 in enumerate(self.vector_diffs):
            alignment = []
            for j, vector_set2 in enumerate(self.vector_diffs):
                cosines = tf.matmul(tf.transpose(vector_set1), vector_set2)
                cos_squared = tf.pow(cosines, 2)
                # Do you want the average alignment or the maximum alignment?
                alignment.append(self.align_fn(cos_squared))
            self.proto_set_alignment.append(alignment)

        # Stuff for decoding the image. Use the appropriate decoding function, set in constructor.
        targ_img = keras.Input(shape=(self.input_size,), name='targ_img')
        dec_input = keras.Input(shape=(self.latent_dim,), name='dec_input')
        # recons = self.dec_fn(feature_enc, dec_input)
        recons = self.get_decoding(feature_enc)

        # Define the loss components
        pred_loss = 0
        proto_dist_loss = 0
        feature_dist_loss = 0
        total_kl_loss = 0
        for i, label_layer in enumerate(self.label_layers):
            pred_loss_fn = keras.backend.categorical_crossentropy(label_layer, classification_preds[i])
            pred_loss += self.classification_weights[i] * keras.backend.mean(pred_loss_fn)
            min_proto_dist = layers.Lambda(ProtoModel.get_min)(dists_to_protos[i])
            min_feature_dist = layers.Lambda(ProtoModel.get_min)(dists_to_latents[i])
            proto_dist_loss += self.proto_dist_weights[i] * keras.backend.mean(min_proto_dist)
            feature_dist_loss += self.feature_dist_weights[i] * keras.backend.mean(min_feature_dist)

            # KL losses. Pretty tricky. Basically, only want to enforce Gaussian around the closest prototype, so I
            # take the softmax of the l2 distance, enabling me to only care about nearest prototypes, and then I
            # actually go about measuring the KL divergence.
            temperature = 0.1  # Smaller means more confident which one we're talking about.
            softmaxed = keras.backend.softmax(-1 * (1.0 / temperature) * mean_dists_to_protos[i], axis=-1)
            reshaped = tf.reshape(softmaxed, shape=(-1, 1, mean_dists_to_protos[i].shape[1]))
            duplicated = tf.tile(reshaped, multiples=[1, self.latent_dim, 1])
            squared_diffs = keras.backend.square(mean_diffs_to_protos[i])
            product_loss = tf.math.multiply(duplicated, squared_diffs)
            dotted_loss = keras.backend.sum(product_loss, axis=-1)
            mean_loss_term = dotted_loss
            kl_losses = 1 + feature_enc_log_var - mean_loss_term - keras.backend.exp(feature_enc_log_var)
            kl_losses *= -0.5
            total_kl_loss += self.kl_losses[i] * keras.backend.mean(kl_losses)
        decode_loss = keras.backend.mean(keras.backend.square(recons - targ_img))
        alignment_loss = 0
        for i, alignment in enumerate(self.proto_set_alignment):
            for j, align in enumerate(alignment):
                print("i:\t", i, "\tj:\t", j)
                print("align", keras.backend.eval(align))
                print("weight", self.disentangle_weights[i][j])
                alignment_loss += self.disentangle_weights[i][j] * align

        img_input = self.encoder.inputs[0]
        all_inputs = [img_input, targ_img]
        all_inputs.extend(self.label_layers)
        all_inputs.extend(self.label_mask_layers)
        all_outputs = []
        all_outputs.extend(classification_preds)
        all_outputs.append(recons)
        self.model = keras.Model(inputs=all_inputs, outputs=all_outputs, name="proto_model")
        self.model.summary()
        print("Decode loss", decode_loss)
        print("total kl loss", total_kl_loss)
        overall_loss = pred_loss + self.decode_weight * decode_loss + proto_dist_loss + feature_dist_loss + \
                       alignment_loss + total_kl_loss
        self.model.add_loss(overall_loss)
        self.model.compile(loss=None, optimizer=self.optimizer)

    def save(self, path):
        self.model.save_weights(path + 'model.h5')
        self.encoder.save_weights(path + 'encoder.h5')
        self.decoder.save_weights(path + 'decoder.h5')
        for idx, classifier_layer in enumerate(self.classifier_layers):
            weights = classifier_layer.get_weights()
            print("Classifier weights", weights)
            with open(path + 'classifier_' + str(idx) + '.h5', 'wb') as f:
                pkl.dump(weights, f)
        for idx, proto_layer in enumerate(self.proto_layers):
            weights = proto_layer.get_weights()
            print("Proto weights", weights)
            with open(path + 'protolayer_' + str(idx) + '.h5', 'wb') as f:
                pkl.dump(weights, f)
        # Save optimizer weights.
        symbolic_weights = getattr(self.optimizer, 'weights')
        weight_values = keras.backend.batch_get_value(symbolic_weights)
        with open(path + 'optimizer.pkl', 'wb') as f:
            pkl.dump(weight_values, f)

    def load(self, path):
        self.model.load_weights(path + 'model.h5')
        self.encoder.load_weights(path + 'encoder.h5')
        self.decoder.load_weights(path + 'decoder.h5')
        for idx, proto_layer in enumerate(self.proto_layers):
            with open(path + 'protolayer_' + str(idx) + '.h5', 'rb') as f:
                weights = pkl.load(f)
                print("Proto weights", weights)
                proto_layer.set_weights(weights)
        self.build_overall_network()
        self.model._make_train_function()
        with open(path + 'optimizer.pkl', 'rb') as f:
            weight_values = pkl.load(f)
        self.optimizer.set_weights(weight_values)

    def create_new_parts(self, output_size, duplication_factor=1):
        proto_layer = ProtoLayer(output_size * duplication_factor, self.latent_dim)
        classifier = DistProbLayer(output_size, duplication_factor)
        label_layer = keras.Input(shape=(output_size,))
        return proto_layer, classifier, label_layer

    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    @staticmethod
    def sampling(args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    # Helper function for putting in lambda layers
    @staticmethod
    def get_min(x):
        return keras.backend.min(x, axis=-1)

    def train(self, x, targ_x, ys, batch_size=64, epochs=10, validation_data=None):
        all_inputs = [x, targ_x]
        all_inputs.extend(ys)
        # Create dummy mask.
        dummy_masks = [np.ones_like(y) for y in ys]
        all_inputs.extend(dummy_masks)
        history = self.model.fit(all_inputs, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                                 validation_data=validation_data, callbacks=self.callbacks)

    def train_on_batches(self, inputs, outputs, batch_size=64, epochs=10, input_height=28):
        # elastic deformation parameters
        sigma = 4
        alpha = 20
        input_width = input_height

        for epoch in range(epochs):
            # Permute all the data first
            permutation_idxs = [np.random.permutation(len(i)) for i in inputs]
            permuted_inputs = [inputs[i][permutation_idxs[i]] for i in range(len(inputs))]
            permuted_outputs = [outputs[i][permutation_idxs[i]] for i in range(len(outputs))]
            batches_per_epoch = permutation_idxs[0].shape[0] // batch_size
            for batch_idx in range(batches_per_epoch):
                if batch_idx % 100 == 0:
                    print("Batch idx", batch_idx, "of", batches_per_epoch, "in epoch", epoch, "of", epochs)
                full_idx = batch_idx * batch_size
                batch_data_inputs = permuted_inputs[0][full_idx: (batch_idx + 1) * batch_size]
                batch_data_outputs = permuted_outputs[0][full_idx: (batch_idx + 1) * batch_size]
                elastic_batch_x = batch_elastic_transform(batch_data_inputs, sigma=sigma, alpha=alpha,
                                                          height=input_height, width=input_width)
                dummy_mask = np.ones_like(batch_data_outputs)  # Untested.
                training_inputs = [elastic_batch_x, batch_data_inputs, batch_data_outputs, dummy_mask]
                self.model.train_on_batch(training_inputs)

    def train_with_dataflow(self, train_df, val_df=None, batch_size=2):
        running_loss = 0
        alpha = 0.001
        for epoch in range(18, 128):
            print("Epoch", epoch)
            for img_idx, dp in enumerate(train_df):
                if img_idx % 100 == 0:
                    print("Batch number", img_idx)
                    print("Running loss", running_loss)
                images = dp[0]
                labels = dp[1:]
                images = np.reshape(images, (batch_size, -1))
                one_hot = labels
                dummy_mask = [np.ones_like(y) for y in labels]
                training_inputs = [images, images]
                training_inputs.extend(one_hot)
                training_inputs.extend(dummy_mask)
                training_loss = self.model.train_on_batch(training_inputs)
                running_loss = (1 - alpha) * running_loss + alpha * training_loss
                if img_idx > 100000:
                    print("Breaking to debug after", img_idx)
                    break
            print("Exited the whole dataflow")
            print("Evaluate performance after epoch", epoch)
            if val_df:
                self.eval_with_dataflow(val_df, batch_size=batch_size)
            self.save('../saved_models/inat/run2/epoch' + str(epoch) + '/')

    def train_hierarchical(self, inputs, outputs, masks, batch_size=32, epochs=10):
        all_inputs = inputs[:]
        all_inputs.extend(outputs)
        all_inputs.extend(masks)
        history = self.model.fit(all_inputs, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                                 validation_data=None, callbacks=self.callbacks)

    def evaluate_hierarchical(self, x, targ_x, ys, y_one_hots, masks, coarse_to_fine, plot=True, gold_tree=None):
        proto_weights = self.proto_layers[0].get_weights()[0]
        all_inputs = [x, targ_x]
        all_inputs.extend(y_one_hots)
        dummy_masks = [np.ones_like(y) for y in y_one_hots]
        all_inputs.extend(dummy_masks)
        predictions = self.model.predict(all_inputs)
        y_accs, numerical_preds = self.predict_hierarchical(predictions, ys, coarse_to_fine=coarse_to_fine)
        # y_accs, numerical_preds = self.eval_accuracy(predictions, ys)

        # What's the alignment like?
        alignments = self.get_alignment()
        # # prob_updates = self.get_grad_orthogonality(inputs, outputs)
        # prob_updates = None
        # return accuracies, alignments, prob_updates
        if gold_tree is not None:
            average_cost = self.get_confusion_matrix(ys, numerical_preds, true_tree=gold_tree)
            print("Average cost", average_cost)
        return y_accs, alignments, average_cost

    def evaluate(self, x, targ_x, ys, y_one_hots, masks=None, plot=True, protected_idx=1, do_fairness_eval=False,
                 gold_tree=None):
        proto_weights = self.proto_layers[0].get_weights()[0]
        all_inputs = [x, targ_x]
        all_inputs.extend(y_one_hots)
        if masks is None:
            masks = [np.ones_like(y) for y in y_one_hots]
        all_inputs.extend(masks)
        predictions = self.model.predict(all_inputs)
        encodings = self.encoder.predict(x)

        # A bunch of methods for evaluating different aspects; uncomment as desired.
        # New evaluation methods should be implemented as separate methods rather than having the code live in here.
        y_accs, numerical_preds = self.eval_accuracy(predictions, ys) if ys is not None else (None, None)
        # y_baseline = 1 - np.mean(ys[0])
        # print("Baseline", y_baseline)
        self.eval_recons(predictions, targ_x)
        # self.eval_preds_by_protected(predictions, ys)
        # self.eval_corrected_recons(predictions, targ_x, ys, encodings, proto_weights)
        # self.eval_corrected(x, ys)
        # self.plot_protos(proto_weights, plot=True)
        # self.plot_recons(predictions, x, plot)
        alignments = self.get_alignment()
        _, _, slopes = self.get_grad_orthogonality(x, ys, y_one_hots)
        # slopes = 0
        s_diff = None
        disparate_impact = None
        demographic = None
        if do_fairness_eval:
            # This metric only makes sense for discrete values.
            # self.get_discrim_metric(predictions, ys)
            # disparate_impact, demographic, extreme_keys = self.get_disparity_metric(predictions, ys, protected_idx=protected_idx)
            s_diff = 0
            # s_diff = self.predict_protected(encodings, ys, protected_idx=protected_idx, keys=extreme_keys)
        average_cost = -1
        if gold_tree is not None:
            average_cost = self.get_confusion_matrix(ys, numerical_preds, true_tree=gold_tree)
            print("Average cost", average_cost)
        return y_accs, s_diff, alignments, slopes, disparate_impact, demographic, average_cost

    def eval_with_dataflow(self, df, batch_size=32):
        print("Starting evaluation")
        num_total = 0
        num_corrects = None
        for batch_idx, batch in enumerate(df):
            images = batch[0]
            one_hot = batch[1:]
            images = np.reshape(images, (batch_size, -1))
            dummy_mask = [np.ones_like(y) for y in one_hot]
            all_inputs = [images, images]
            all_inputs.extend(one_hot)
            all_inputs.extend(dummy_mask)
            ys = []
            for label in one_hot:
                ys.append(np.argmax(label, axis=1))
            predictions = self.model.predict(all_inputs)
            y_accs, numerical_preds = self.eval_accuracy(predictions, ys)
            if num_corrects is None:
                num_corrects = [y_acc * batch_size for y_acc in y_accs]
            else:
                num_corrects = [num_corrects[i] + y_accs[i] * batch_size for i in range(len(y_accs))]
            num_total += batch_size
            if batch_idx > 100000:
                print("Breaking evaluation after batch number", batch_idx)
                break
        print("Final accuracies", [num_correct / num_total for num_correct in num_corrects])

    def get_confusion_matrix(self, ys, predicted_ys, true_tree):
        tree, class_labels = true_tree
        tree = nx.DiGraph.to_undirected(tree)
        for idx in range(len(self.output_sizes)):
            true_data = ys[idx]
            if len(true_data.shape) > 1:
                true_data = true_data.flatten()
            data = {'actual': true_data,
                    'predicted': predicted_ys[idx]}
            df = pd.DataFrame(data, columns=['actual', 'predicted'])
            confusion_matrix = pd.crosstab(df['actual'], df['predicted'], rownames=['Actual'],
                                           colnames=['Predicted'])

            # Calculate the average cost as the distance to the first common ancestor.
            # shortest_paths = nx.algorithms.all_pairs_shortest_path_length(tree)
            # running_cost = 0
            # num_entries = 0
            # Set the diagonals to 0.

            # Set diagonal of conf_np to 0 so only measure cost of the actual errors.
            for col in confusion_matrix.columns:
                confusion_matrix.at[col, col] = 0
            conf_np = confusion_matrix.to_numpy()

            cost_matrix = np.zeros_like(conf_np)
            for row_idx, row_label in enumerate(class_labels):
                for col_idx, col_label in enumerate(class_labels):
                    # print("Finding path from", row_label, "to", col_label)
                    cost_matrix[row_idx, col_idx] = nx.algorithms.shortest_path_length(tree, row_label, col_label)
            total_cost = np.multiply(cost_matrix, conf_np)
            average_cost = np.sum(total_cost) / np.sum(conf_np)
            # Set the diagonals to 0.
            for col in confusion_matrix.columns:
                confusion_matrix.at[col, col] = 0
            sn.heatmap(confusion_matrix, annot=True)
            # plt.show()
            return average_cost

    def print_proto_bases(self):
        for proto_set in self.proto_layers:
            for diff in tf.unstack(tf.transpose(proto_set.vector_diffs)):
                print("Basis", keras.backend.eval(diff))

    def get_prototypes(self):
        prototypes = []
        for proto_layer in self.proto_layers:
            prototypes.extend(proto_layer.get_weights()[0])
        return prototypes

    def predict_hierarchical(self, predictions, ys, coarse_to_fine):
        fine_to_coarse = {}
        for coarse, fines in coarse_to_fine.items():
            for fine in fines:
                fine_to_coarse[fine] = coarse
        # Assume fine and then coarse.
        # First measure the coarse predictions
        coarse_predictions = []
        num_evaluated = 0
        num_correct = 0
        for i, test_prediction in enumerate(predictions[1]):
            class_prediction = np.argmax(test_prediction)
            coarse_predictions.append(class_prediction)
            num_evaluated += 1
            if class_prediction == ys[1][i]:
                num_correct += 1
        coarse_accuracy = num_correct / num_evaluated
        print("Classification accuracy for coarse predictor:", coarse_accuracy)

        # Now measure the fine predictions
        fine_predictions = []
        num_evaluated = 0
        num_correct = 0
        for i, coarse_prediction in enumerate(predictions[1]):
            one_hotted_coarse = np.zeros_like(coarse_prediction)
            one_hotted_coarse[np.argmax(coarse_prediction)] = 1
            fine_prediction = predictions[0][i]
            # print("coarse", coarse_prediction)
            # print("fine", fine_prediction)

            # For each possible coarse label, compute the conditional probability of each fine label.
            joint_prediction = np.zeros_like(fine_prediction)
            for possible_coarse in coarse_to_fine.keys():
                new_mask = np.zeros_like(fine_prediction)
                for matching_fine in coarse_to_fine.get(possible_coarse):
                    new_mask[matching_fine] = 1
                # Now have the mask, create the condition dist.
                masked_fine = np.multiply(new_mask, fine_prediction)
                normalized_masked = masked_fine / np.sum(masked_fine)
                for fine_idx in coarse_to_fine.get(possible_coarse):
                    joint_prediction[fine_idx] = normalized_masked[fine_idx] * coarse_prediction[
                        fine_to_coarse.get(fine_idx)]

            # Normalize one last time
            joint_prediction = joint_prediction / np.sum(joint_prediction)
            # print("Joint", joint_prediction)
            final_fine_prediction = np.argmax(joint_prediction)
            fine_predictions.append(final_fine_prediction)
            num_evaluated += 1
            if final_fine_prediction == ys[0][i]:
                num_correct += 1
        fine_accuracy = num_correct / num_evaluated
        print("Classification accuracy for fine predictor:", fine_accuracy)
        return [fine_accuracy, coarse_accuracy], [fine_predictions, coarse_predictions]

    def eval_accuracy(self, predictions, ys, masks=None):
        y_accs = []
        numerical_predictions = []
        for predictor_idx, predictor in enumerate(self.predictors):
            num_evaluated = 0
            num_correct = 0
            numerical_prediction = []
            for i, test_prediction in enumerate(predictions[predictor_idx]):
                # if masks is not None:
                #     test_prediction = np.multiply(test_prediction, masks[predictor_idx][i])
                class_prediction = np.argmax(test_prediction)
                numerical_prediction.append(class_prediction)
                num_evaluated += 1
                if class_prediction == ys[predictor_idx][i]:
                    num_correct += 1
            accuracy = num_correct / num_evaluated
            print("Classification accuracy for output", predictor_idx, ":", accuracy)
            y_accs.append(accuracy)
            numerical_predictions.append(numerical_prediction)
        return y_accs, numerical_predictions

    def eval_preds_by_protected(self, predictions, ys):
        for concept_idx, concept_predictor in enumerate(self.predictors):
            concept_predictions = np.argmax(predictions[concept_idx], axis=1)
            for other_concept_idx in range(len(self.predictors)):
                if other_concept_idx == concept_idx:
                    continue
                true_other_concept = ys[other_concept_idx]
                class_to_distribution = {}
                for pred_idx, concept_prediction in enumerate(concept_predictions):
                    distribution = class_to_distribution.get(concept_prediction)
                    if distribution is None:
                        distribution = {}
                        class_to_distribution[concept_prediction] = distribution
                    true_other_val = true_other_concept[pred_idx]
                    if distribution.get(true_other_val) is None:
                        distribution[true_other_val] = 1
                    else:
                        distribution[true_other_val] += 1
                print("For concept idx", concept_idx, "got the following distribution.")
                for key, val in sorted(class_to_distribution.items()):
                    total = sum(val.values())
                    normalized = {}
                    for dist_key, dist_val in val.items():
                        normalized[dist_key] = dist_val / total
                    print("For true concept value", key, "got prediction distribution", sorted(normalized.items()))

    def eval_recons(self, predictions, xs):
        recons = predictions[-1]
        reconstruction_error = np.mean(np.square(recons - xs))
        print("Reconstruction error:", reconstruction_error)

    def eval_corrected_recons(self, predictions, xs, ys, encodings, proto_weights):
        for percent in range(10, 100, 10):
            recons_errors = []
            frac = percent / 100
            for i, test_prediction in enumerate(predictions[0]):
                encoding = encodings[i]
                correct_class = ys[0][i]
                correct_proto = proto_weights[correct_class]
                diff_vector = correct_proto - encoding
                new_enc = np.reshape(encoding + frac * diff_vector, (1, -1))
                new_dec = self.decoder.predict(new_enc)
                recons_errors.append(np.mean(np.square(new_dec - xs[i])))
            print("For frac", frac, "Corrected recons error", np.mean(recons_errors))

    def eval_corrected(self, x, ys):
        proto_weights = self.proto_layers[0].get_weights()[0]
        encodings = self.encoder.predict(x)
        _, _, _, in_plane_points = self.projectors[0](encodings)
        in_plane_points = keras.backend.eval(in_plane_points)
        predictions = self.predictors[0].predict(encodings)
        for target_classification in range(self.output_sizes[0]):
            # Find a mis-classification
            num_examples = 0
            for pred_idx, prediction in enumerate(predictions):
                if ys[0][pred_idx] != target_classification:
                    continue  # Skip if the true value isn't the one we want
                if np.argmax(prediction) == target_classification:
                    continue  # Skip if the predictor made the correct classification
                if num_examples >= 1:  # However many you want, but update the file name.
                    break
                num_examples += 1
                # At this point have an input that should have been classified as target_classification but was not.
                # So get the encoding and move along the gradient.
                encoding = encodings[pred_idx]
                var_enc = tf.constant(np.reshape(encoding, (1, -1)))
                # Calculate the gradient itself.
                grads = self.get_grads(tf.reshape(var_enc, (1, -1)),
                                       tf.reshape(tf.one_hot(target_classification, self.output_sizes[0]), (1, -1)),
                                       self.predictors[0])
                num_steps = 10
                decodings = np.zeros((num_steps + 1, self.input_size))
                decodings[0] = x[pred_idx]
                np_grad = keras.backend.eval(grads)
                for step in range(num_steps):
                    new_enc = np.reshape(encoding - step / num_steps * np_grad, (1, -1))
                    new_dec = self.decoder.predict(new_enc)
                    decodings[step + 1] = new_dec
                img_save_location = '../saved_models/'
                plot_rows_of_images([decodings],
                                    savepath=img_save_location + 'corrected_to_' + str(target_classification),
                                    show=False)

                # Just calculate vector to the right prototype.
                # Instead of taking the normal vector difference, project the enc into the plane, find that difference, and then update.
                diff_to_proto = proto_weights[target_classification] - in_plane_points[pred_idx]
                decodings = np.zeros((num_steps + 1, self.input_size))
                decodings[0] = x[pred_idx]
                for step in range(num_steps):
                    new_enc = np.reshape(encoding + step / num_steps * diff_to_proto, (1, -1))
                    new_dec = self.decoder.predict(new_enc)
                    decodings[step + 1] = new_dec
                img_save_location = '../saved_models/'
                plot_rows_of_images([decodings],
                                    savepath=img_save_location + 'non_grad_corrected_to_' + str(target_classification),
                                    show=False)

                unit_grad = np_grad / np.linalg.norm(np_grad)
                unit_diff = diff_to_proto / np.linalg.norm(diff_to_proto)
                cos = np.dot(unit_grad, unit_diff)
                print("Cosine", cos)
                print("Angle", np.arccos(cos))  # Prints in radians

    def plot_protos(self, proto_weights, plot=True):
        # Plot and save the prototypes.
        num_protos_to_plot = self.output_sizes[0]
        decoded_prototypes = np.zeros((num_protos_to_plot, self.input_size))
        for proto_idx in range(num_protos_to_plot):
            proto_enc = np.reshape(proto_weights[proto_idx], (1, -1))
            # Pass through decoder
            decoded_proto = self.decoder.predict(proto_enc)
            decoded_prototypes[proto_idx] = decoded_proto
        img_save_location = '../saved_models/'
        if plot:
            plot_rows_of_images([decoded_prototypes], savepath=img_save_location + 'prototypes', show=plot)

    def plot_recons(self, predictions, x, plot):
        recons = predictions[-1]
        img_save_location = '../saved_models/'
        # Plot and save reconstructions.
        NUM_EXAMPLES_TO_PLOT = 10
        originals = np.zeros((NUM_EXAMPLES_TO_PLOT, self.input_size))
        reconstructions = np.zeros((NUM_EXAMPLES_TO_PLOT, self.input_size))
        for test_idx in range(NUM_EXAMPLES_TO_PLOT):
            originals[test_idx] = x[test_idx]
            reconstructions[test_idx] = recons[test_idx]
        if plot:
            plot_rows_of_images([originals, reconstructions], img_save_location + 'reconstructions', show=False)

    def get_alignment(self):
        # Evaluate stuff about the alignment of prototype sets
        alignments = []
        for alignment in self.proto_set_alignment:
            for align in alignment:
                evaled_align = keras.backend.eval(align)
                # print("Alignment", evaled_align)
                alignments.append(evaled_align)
        return alignments

    # Calculate the gradient of the classification loss with respect to the encoding and move in that direction
    # in the latent space.
    # Used this site for gradients: https://tensorflow.google.cn/tutorials/generative/adversarial_fgsm?hl=zh-cn
    def get_grad_orthogonality(self, x, ys, y_one_hots):
        prob_updates = []
        other_updates = []
        slopes = []
        encodings = self.encoder.predict(x)
        for predictor_idx, predictor in enumerate(self.predictors):
            if predictor_idx >= 1:
                print("Only do this analysis for first case.")
                continue
            # Add dummy mask if you want to do this.
            dummy_masks = np.ones_like(y_one_hots[predictor_idx])
            curr_predictions = predictor.predict([encodings, dummy_masks])
            for target_classification in range(self.output_sizes[predictor_idx]):
                for magnitude in [1.0]:
                    updated_probs = []
                    other_class_probs = []
                    num_examples_generated = 0
                    for i, test_prediction in enumerate(curr_predictions):
                        if num_examples_generated >= 10:  # However many examples you want. 50 takes maybe a minute per?
                            break
                        encoding = encodings[i]
                        correct_class = ys[predictor_idx][i]
                        if correct_class != target_classification:
                            continue
                        num_examples_generated += 1
                        var_enc = tf.constant(np.reshape(encoding, (1, -1)))
                        var_enc_reshaped = tf.reshape(var_enc, (1, -1))
                        one_hot_correct_class = tf.one_hot(correct_class, self.output_sizes[predictor_idx])
                        reshaped_correct_class = tf.reshape(one_hot_correct_class, (1, -1))
                        grads = self.get_grads(var_enc_reshaped, reshaped_correct_class, predictor)
                        individual_mask = np.reshape(dummy_masks[i], (1, -1))
                        confidence = predictor.predict([np.reshape(encoding, (1, -1)), individual_mask])
                        new_enc = np.reshape(encoding - magnitude * keras.backend.eval(grads), (1, -1))
                        new_confidence = predictor.predict([new_enc, individual_mask])
                        updated_probs.append(new_confidence - confidence)
                        if len(self.predictors) > 1:
                            other_idx = 0 if predictor_idx != 0 else len(self.output_sizes) - 1
                            other_predictor = self.predictors[other_idx]
                            other_mask = np.reshape(np.ones(self.output_sizes[other_idx]), (1, -1))
                            old_other_confidence = other_predictor.predict([np.reshape(encoding, (1, -1)), other_mask])
                            new_other_confidence = other_predictor.predict([new_enc, other_mask])
                            other_class_probs.append(new_other_confidence - old_other_confidence)
                    # print("Along gradient for predictor", predictor_idx, "towards classification", target_classification, "with magnitude", magnitude)
                    mean_updated_probs = np.mean(updated_probs, axis=0)
                    main_update_diff = max(abs(mean_updated_probs[0]))
                    prob_updates.append(mean_updated_probs[0].tolist())
                    if len(self.predictors) > 1:
                        mean_other_update = np.mean(other_class_probs, axis=0)
                        other_updated_diff = max(abs(mean_other_update[0]))
                        slope = other_updated_diff / main_update_diff
                        print("Ratio", slope)
                        # print("Mean confidence change for other predictor", mean_other_update)
                        other_updates.append(mean_other_update[0].tolist())
                        slopes.append(slope)
        return prob_updates, other_updates, slopes

    def get_grads(self, input_enc, classification, predictor):
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            tape.watch(input_enc)
            mask = tf.ones_like(classification)
            prediction = predictor([input_enc, mask])
            loss = loss_object(classification, prediction)
        # Get the gradients of the loss w.r.t to the input.
        gradient = tape.gradient(loss, input_enc)
        return gradient

    def viz_latent_space(self, x, y, class_labels, proto_indices=0):
        # Generate encodings.
        encodings = self.encoder.predict(x)
        # Only take the first k
        num_to_keep = 200
        encodings = encodings[:num_to_keep, :]
        # Get the prototype encodings out to plot as well.
        protos = self.proto_layers[proto_indices].get_weights()[0]
        plot_latent_prompt(encodings, labels=y[:num_to_keep], test_encodings=protos, classes=class_labels)
        all_protos = []
        for proto_layer in self.proto_layers:
            all_protos.extend(proto_layer.get_weights()[0])
        plot_latent_prompt(encodings, labels=y[:num_to_keep], test_encodings=all_protos, classes=class_labels)

    # Plot the latent space, projected down into the subspace defined by the prototypes.
    def viz_projected_latent_space(self, x, y, class_labels, proto_indices):
        # Generate encodings.
        encodings = self.encoder.predict(x)
        # Only take the first k
        num_to_keep = 200
        encodings = encodings[:num_to_keep, :]
        dist_to_protos, dist_to_latents, diff_to_protos, in_plane_point = self.projectors[proto_indices](encodings)
        np_points_in_plane = keras.backend.eval(in_plane_point)
        protos = self.proto_layers[proto_indices].get_weights()[0]
        plot_latent_prompt(np_points_in_plane, labels=y[:num_to_keep], test_encodings=protos, classes=class_labels)

    @staticmethod
    def get_data_matching_value(ys, idx, val):
        matching_indices = []
        for entry_idx, entry in enumerate(ys[idx]):
            if entry == val:
                matching_indices.append(entry_idx)
        return matching_indices

    @staticmethod
    def get_num_with_val(val, predictions):
        num_matching = 0
        for pred in predictions:
            if np.argmax(pred) == val:
                num_matching += 1
        return num_matching

    @staticmethod
    def get_prob_with_val(val, predictions):
        cumulative_prob = 0
        for pred in predictions:
            if np.argmax(pred) == val:
                cumulative_prob += np.max(pred)
        return cumulative_prob

    # Calculates the discrimination metric, defined in Zemel et al 2013, but copied here:
    # https://arxiv.org/pdf/1910.12854.pdf
    def get_discrim_metric(self, predictions, ys):
        prediction_concept_idx = 0
        protected_concept_idx = 1
        for protected_val0 in range(self.output_sizes[protected_concept_idx]):
            # How many data entries are there like this?
            protected0_indices = ProtoModel.get_data_matching_value(ys, protected_concept_idx, protected_val0)
            num_p0_entries = len(protected0_indices)
            predictions0_subset = predictions[prediction_concept_idx][protected0_indices]
            num_p0_matching = ProtoModel.get_num_with_val(val=1, predictions=predictions0_subset)  # FIXME: val
            p0_score = num_p0_matching / num_p0_entries
            prob_p0_matching = ProtoModel.get_prob_with_val(val=1, predictions=predictions0_subset)
            prob_p0_score = prob_p0_matching / num_p0_entries
            for protected_val1 in range(self.output_sizes[protected_concept_idx]):
                if protected_val0 <= protected_val1:
                    continue
                protected1_indices = ProtoModel.get_data_matching_value(ys, protected_concept_idx, protected_val1)
                num_p1_entries = len(protected1_indices)
                predictions1_subset = predictions[prediction_concept_idx][protected1_indices]
                num_p1_matching = ProtoModel.get_num_with_val(val=1, predictions=predictions1_subset)  # FIXME: val
                p1_score = num_p1_matching / num_p1_entries
                prob_p1_matching = ProtoModel.get_prob_with_val(val=1, predictions=predictions1_subset)
                prob_p1_score = prob_p1_matching / num_p1_entries

                print("protected val 0:", protected_val0)
                print("Num entries for protected val 0:", num_p0_entries)
                print("Num correct for protected val 0:", num_p0_matching)
                print("protected val 1:", protected_val1)
                print("Num entries for protected val 1:", num_p1_entries)
                print("Num correct for protected val 1:", num_p1_matching)
                print("P0 score", p0_score)
                print("P1 score", p1_score)
                print("Difference", p0_score - p1_score)
                print("Prob difference", prob_p0_score - prob_p1_score)

    def get_disparity_metric(self, predictions, ys, protected_idx=1):
        protected_totals = {}
        protected_positive = {}
        protected_vals = ys[protected_idx]
        for i, protected_val in enumerate(protected_vals):
            if protected_val not in protected_totals.keys():
                protected_totals[protected_val] = 0
                protected_positive[protected_val] = 0
            protected_totals[protected_val] += 1
            prediction = np.argmax(predictions[0][i])
            # print("Full prediction", predictions[0][i])
            # print("The prediction", prediction)
            # print("The protected val", protected_val)
            if prediction % 2 == 1:
                protected_positive[protected_val] += 1
        print("Protected positives", protected_positive)
        print("Protected totals", protected_totals)
        print("All fractions")
        for key in protected_totals.keys():
            print("For key", key, protected_positive[key] / protected_totals[key])
        sorted_keys = sorted(protected_positive.keys())
        max_key = max(protected_totals, key=lambda x: protected_positive.get(x) / protected_totals.get(x))
        min_key = min(protected_totals, key=lambda x: protected_positive.get(x) / protected_totals.get(x))
        random.shuffle(sorted_keys)
        # sorted_keys = sorted_keys[:2]
        sorted_keys = [max_key, min_key]

        # print("Using protecteds", sorted_keys)
        # key0, key1 = sorted_keys
        key0 = max_key
        key1 = min_key
        fraction_0 = protected_positive.get(key0) / protected_totals.get(key0)
        fraction_1 = protected_positive.get(key1) / protected_totals.get(key1)
        print("Fraction0", fraction_0)
        print("Fraction1", fraction_1)
        disparity_impact = min([fraction_0 / fraction_1, fraction_1 / fraction_0])
        print("Min disparity", disparity_impact)

        # From the Wasserstein paper, compute the "demographic disparity"
        mean_prob = sum(protected_positive.values()) / sum(protected_totals.values())
        dem_disparity = 0
        for key in sorted_keys:
            fraction = protected_positive.get(key) / protected_totals.get(key)
            dem_disparity += abs(fraction - mean_prob)
        # for key, pos_val in protected_positive.items():
        #     fraction = pos_val / protected_totals.get(key)
        #     dem_disparity += abs(fraction - mean_prob)
        print("Dem disparity", dem_disparity)
        return disparity_impact, dem_disparity, (max_key, min_key)

    def predict_protected(self, encodings, ys, protected_idx=1, keys=None):
        _, _, _, in_plane_point = self.projectors[0](encodings)
        # Define the subspace you want to pass in to predict s from. I think the right way is to say just the
        # subspace used for prediction, but alternatively you could use the bit that's perpendicular to the
        # protected subspace.
        from sklearn.linear_model import LogisticRegression
        filtered_ys = ys
        filtered_plane_points = in_plane_point
        if keys is not None:
            filtered_ys = [[], [], []]
            filtered_plane_points = []
            for idx in range(len(ys[0])):
                p_val = ys[protected_idx][idx]
                if p_val in keys:
                    filtered_ys[0].append(ys[0][idx])
                    filtered_ys[1].append(ys[1][idx])
                    filtered_plane_points.append(in_plane_point[idx])
                    if p_val == keys[0]:
                        filtered_ys[2].append(0)
                    elif p_val == keys[1]:
                        filtered_ys[2].append(1)
            filtered_ys = [np.stack(elt) for elt in filtered_ys]
            filtered_plane_points = tf.stack(filtered_plane_points)

        relevant_subspace = keras.backend.eval(filtered_plane_points)
        regression_model = LogisticRegression()
        train_frac = 0.5
        num_points = relevant_subspace.shape[0]
        x_train = relevant_subspace[:int(train_frac * num_points)]
        y_train = filtered_ys[protected_idx][:int(train_frac * num_points)]
        regression_model.fit(x_train, y_train)

        # Use score method to get accuracy of model
        x_test = relevant_subspace[int(train_frac * num_points):]
        y_test = filtered_ys[protected_idx][int(train_frac * num_points):]
        score = regression_model.score(x_test, y_test)
        print("logistic s prediction", score)
        random_chance = stats.mode(y_test)[1] / (num_points - int(train_frac * num_points))
        print("Random baseline", random_chance)

        return score - random_chance

    def eval_proto_diffs_parity(self, concept_idx=0):
        best_align = self.eval_proto_diffs(concept_idx=concept_idx)
        diff_parity = []
        same_parity = []
        for j, best_row in enumerate(best_align):
            for k, entry in enumerate(best_row):
                if j == k:
                    continue
                if j % 2 != k % 2:
                    diff_parity.append(entry)
                    continue
                same_parity.append(entry)
        print("Diff parity", diff_parity)
        print("Same parity", same_parity)
        print("Mean diff", np.mean(diff_parity))
        print("Mean same", np.mean(same_parity))
        print("Median diff", np.median(diff_parity))
        print("Median same", np.median(same_parity))
        mean_diffs = [np.mean(diff_parity)]
        mean_sames = [np.mean(same_parity)]
        return mean_diffs, mean_sames

    def eval_proto_diffs_fashion(self, concept_idx=0):
        best_align = self.eval_proto_diffs(concept_idx=concept_idx)
        diff_groups = []
        same_groups = []
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
        for j, best_row in enumerate(best_align):
            for k, entry in enumerate(best_row):
                if j == k:
                    continue
                if class_to_group_mapping.get(j) == class_to_group_mapping.get(k):
                    same_groups.append(entry)
                    continue
                diff_groups.append(entry)
        print("Diff groups", diff_groups)
        print("Same groups", same_groups)
        print("Mean diff", np.mean(diff_groups))
        print("Mean same", np.mean(same_groups))
        print("Median diff", np.median(diff_groups))
        print("Median same", np.median(same_groups))

        mean_diffs = []
        mean_sames = []
        mean_diffs.append(np.mean(diff_groups))
        mean_sames.append(np.mean(same_groups))
        return mean_diffs, mean_sames

    def eval_proto_diffs_bolts(self, concept_idx=0):
        best_align = self.eval_proto_diffs(concept_idx=concept_idx)
        diff_groups = []
        same_groups = []
        class_to_group_mapping = {0: 0,
                                  1: 0,
                                  2: 0,
                                  3: 0,
                                  4: 1,
                                  5: 1,
                                  6: 1,
                                  7: 1}
        for j, best_row in enumerate(best_align):
            for k, entry in enumerate(best_row):
                if j == k:
                    continue
                if class_to_group_mapping.get(j) == class_to_group_mapping.get(k):
                    same_groups.append(entry)
                    continue
                diff_groups.append(entry)
        print("Diff groups", diff_groups)
        print("Same groups", same_groups)
        print("Mean diff", np.mean(diff_groups))
        print("Mean same", np.mean(same_groups))
        print("Median diff", np.median(diff_groups))
        print("Median same", np.median(same_groups))

        mean_diffs = []
        mean_sames = []
        mean_diffs.append(np.mean(diff_groups))
        mean_sames.append(np.mean(same_groups))
        return mean_diffs, mean_sames

    def eval_proto_diffs(self, concept_idx=0):
        true_protos = keras.backend.eval(self.proto_layers[concept_idx].prototypes)
        true_diffs = ProtoModel.get_vector_differences(true_protos)

        mean_diffs = []
        mean_sames = []
        for i, other_protos in enumerate(self.proto_layers):
            if i == concept_idx:
                continue
            # For each of the true diffs, find how well it aligns with going from one other to another
            alignments = np.zeros(
                (len(true_diffs), len(true_diffs), other_protos.num_prototypes, other_protos.num_prototypes))

            other_diffs = ProtoModel.get_vector_differences(keras.backend.eval(other_protos.prototypes))
            for true_idx1, true_proto1_diffs in enumerate(true_diffs):
                for true_idx2, true_p1_to_p2 in enumerate(true_proto1_diffs):
                    if true_idx1 == true_idx2:
                        continue
                    for j, other_p1_diffs in enumerate(other_diffs):
                        for k, other_p1_to_p2 in enumerate(other_p1_diffs):
                            if j == k:
                                continue
                            cos_alignment = np.square(np.dot(true_p1_to_p2, other_p1_to_p2))
                            alignments[true_idx1, true_idx2, j, k] = cos_alignment
            # print("Alignments", alignments)
            # Now condense down to say how well aligned is
            best_align = np.max(alignments, axis=(0, 1))
            # print("Best alignments", best_align)
            return best_align

    def eval_parity_proto_diff_hierarchy(self):  # FIXME
        best_align = self.eval_proto_diffs_hierarchy()
        diff_parity = []
        same_parity = []
        for j, best_row in enumerate(best_align):
            for k, entry in enumerate(best_row):
                if j == k:
                    continue
                if (j < 5) != (k < 5):
                    diff_parity.append(entry)
                    continue
                same_parity.append(entry)
        print("Diff parity", diff_parity)
        print("Same parity", same_parity)
        print("Mean diff", np.mean(diff_parity))
        print("Mean same", np.mean(same_parity))
        print("Median diff", np.median(diff_parity))
        print("Median same", np.median(same_parity))

        mean_diffs = []
        mean_sames = []
        mean_diffs.append(np.mean(diff_parity))
        mean_sames.append(np.mean(same_parity))
        return mean_diffs, mean_sames

    def eval_fashion_proto_diff_hierarchy(self):
        best_align = self.eval_proto_diffs_hierarchy()
        diff_groups = []
        same_groups = []
        for j, best_row in enumerate(best_align):
            for k, entry in enumerate(best_row):
                if j == k:
                    continue
                if self.protos_from_same_set(j, k):
                    same_groups.append(entry)
                else:
                    diff_groups.append(entry)
        print("Diff groups", diff_groups)
        print("Same groups", same_groups)
        print("Mean diff", np.mean(diff_groups))
        print("Mean same", np.mean(same_groups))
        print("Median diff", np.median(diff_groups))
        print("Median same", np.median(same_groups))

        mean_diffs = []
        mean_sames = []
        mean_diffs.append(np.mean(diff_groups))
        mean_sames.append(np.mean(same_groups))
        return mean_diffs, mean_sames

    def eval_proto_diffs_hierarchy(self):
        true_protos = keras.backend.eval(self.proto_layers[0].prototypes)
        true_diffs = ProtoModel.get_vector_differences(true_protos)

        # Collapse the prototypes from other concepts into one set.
        other_protos = []
        for j, other_concept in enumerate(self.proto_layers):
            if j == 0:
                continue
            other_protos.extend(keras.backend.eval(other_concept.prototypes))
        other_protos = np.asarray(other_protos)

        # For each of the true diffs, find how well it aligns with going from one other to another
        alignments = np.zeros(
            (len(true_diffs), len(true_diffs), len(other_protos), len(other_protos)))

        other_diffs = ProtoModel.get_vector_differences(other_protos)
        for true_idx1, true_proto1_diffs in enumerate(true_diffs):
            for true_idx2, true_p1_to_p2 in enumerate(true_proto1_diffs):
                if true_idx1 == true_idx2:
                    continue
                for j, other_p1_diffs in enumerate(other_diffs):
                    for k, other_p1_to_p2 in enumerate(other_p1_diffs):
                        if j == k:
                            continue
                        cos_alignment = np.square(np.dot(true_p1_to_p2, other_p1_to_p2))
                        alignments[true_idx1, true_idx2, j, k] = cos_alignment
        # print("Alignments", alignments)
        # Now condense down to say how well aligned is
        best_align = np.max(alignments, axis=(0, 1))
        print("Best alignments", best_align)
        return best_align

    @staticmethod
    def get_vector_differences(protos):
        diffs = []
        for i, proto1 in enumerate(protos):
            diffs_for_i = []
            for j, proto2 in enumerate(protos):
                if i == j:
                    diffs_for_i.append(0)
                    continue
                diffs_for_i.append((proto1 - proto2) / np.linalg.norm(proto1 - proto2))
            diffs.append(diffs_for_i)
        return diffs

    def protos_from_same_set(self, idx1, idx2):
        running_id1 = idx1
        running_id2 = idx2
        for size in self.output_sizes:
            running_id1 -= size
            running_id2 -= size
            if running_id1 < 0 and running_id2 < 0:
                return True
            if (running_id1 < 0 and running_id2 >= 0) or (running_id1 >= 0 and running_id2 < 0):
                return False
        return False

    def get_mst(self, add_origin=True, plot=False, labels=None):
        nx_graph = nx.Graph()
        tuple_encs = []
        if len(self.output_sizes) != 2:
            print("WARNING: assume 2 outputs when building MST.")
        encodings = self.get_prototypes()[:self.output_sizes[0] + self.output_sizes[1]]
        all_labels = []
        for group_id, label_group in enumerate(labels):
            duplication_factor = self.duplication_factors[group_id]
            added_labels = [label_group for _ in range(duplication_factor)]
            flattened_added = [item for sublist in added_labels for item in sublist]
            all_labels.extend(flattened_added)
        if add_origin:
            encodings.append(np.zeros_like(encodings[0]))
            all_labels.append("origin")
        for label_id, encoding in enumerate(encodings):
            label = all_labels[label_id]
            tuple_enc = tuple(encoding)
            nx_graph.add_node(label)
            for other_id, other_enc in enumerate(tuple_encs):
                other_label = all_labels[other_id]
                nx_graph.add_edge(label, other_label,
                                  weight=np.linalg.norm(np.asarray(other_enc) - np.asarray(tuple_enc)))
            tuple_encs.append(tuple_enc)
        undirected = nx_graph.to_undirected()
        tree = nx.minimum_spanning_tree(undirected)
        # Create a directed version of the tree via the ordering of prototypes.
        ordered_tree = nx.DiGraph(tree)
        for start_id in range(len(encodings)):
            start_label = all_labels[start_id]
            for end_id in range(len(encodings)):
                end_label = all_labels[end_id]
                # Small to big vs. big to small
                # In this case, only allow big to small
                if ordered_tree.has_edge(end_label, start_label) and end_id < start_id:
                    ordered_tree.remove_edge(end_label, start_label)
                    # print("Removing edge from", end_label, "to", start_label)
        if plot:
            plot_mst(ordered_tree)
        # Set node attributes in tree, which is needed for tree equality comparison.
        for node in ordered_tree.nodes:
            ordered_tree.add_node(node, name=str(node))
        return ordered_tree

    def get_deep_mst(self, add_origin=True, plot=False, labels=None):
        nx_graph = nx.Graph()
        tuple_encs = []
        encodings = self.get_prototypes()
        all_labels = []
        for group_id, label_group in enumerate(labels):
            duplication_factor = self.duplication_factors[group_id]
            assert duplication_factor == 1
            added_labels = [label_group for _ in range(duplication_factor)]
            flattened_added = [item for sublist in added_labels for item in sublist]
            all_labels.extend(flattened_added)
        if add_origin:
            encodings.append(np.zeros_like(encodings[0]))
            all_labels.append("origin")
        for label_id, encoding in enumerate(encodings):
            label = all_labels[label_id]
            tuple_enc = tuple(encoding)
            nx_graph.add_node(label)
            for other_id, other_enc in enumerate(tuple_encs):
                other_label = all_labels[other_id]
                nx_graph.add_edge(label, other_label,
                                  weight=np.linalg.norm(np.asarray(other_enc) - np.asarray(tuple_enc)))
            tuple_encs.append(tuple_enc)
        undirected = nx_graph.to_undirected()
        tree = nx.minimum_spanning_tree(undirected)
        # Create a directed version of the tree via the ordering of prototypes.
        ordered_tree = nx.DiGraph(tree)
        for start_id in range(len(encodings)):
            start_label = all_labels[start_id]
            for end_id in range(len(encodings)):
                end_label = all_labels[end_id]
                # Small to big vs. big to small
                # In this case, only allow big to small
                if ordered_tree.has_edge(end_label, start_label) and end_id < start_id:
                    ordered_tree.remove_edge(end_label, start_label)
                    # print("Removing edge from", end_label, "to", start_label)
        if plot:
            plot_mst(ordered_tree)
        # Set node attributes in tree, which is needed for tree equality comparison.
        for node in ordered_tree.nodes:
            ordered_tree.add_node(node, name=str(node))
        return ordered_tree
