import unittest
from src.models.proto_layer import ProtoLayer
from tensorflow import keras
import numpy as np
from src.models.proto_model import ProtoModel

class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        keras.backend.clear_session()

    def test_projecting1(self):
        num_prototypes = 2
        num_dims = 3
        my_proto_layer = ProtoLayer(num_prototypes, num_dims)
        my_input = keras.Input(shape=(num_dims,))
        my_output = my_proto_layer(my_input)
        my_model = keras.Model(my_input, my_output)
        my_model.compile()

        protos = keras.backend.eval(my_proto_layer.prototypes)
        print("protos", protos)

        np.random.seed(5)
        random_encoding = np.reshape(np.random.random(num_dims), (1, num_dims))
        print("Random encoding", random_encoding)
        dists_in_plane_proto, dists_in_plane_latents, out_of_plane_diffs = my_model.predict(random_encoding)
        print("out of plane diffs", out_of_plane_diffs)

        # Calculate these same things using numpy and check
        offset = protos[0, :]
        basis = protos[1, :] - offset
        offset_enc = random_encoding - offset
        projection = basis * np.dot(offset_enc[0], basis) / np.dot(basis, basis)
        perpendicular = offset_enc - projection
        print("Projection", projection)
        print("Perpendicular", perpendicular)
        # print("Proto 0", offset)
        # print("Along basis", offset + projection)
        # print("To end", offset + projection + perpendicular)
        # print("Final", random_encoding)
        self.assertTrue(np.allclose(np.transpose(out_of_plane_diffs[0, :, 0]), perpendicular[0]))

    def test_projecting2(self):
        num_prototypes = 2
        num_dims = 3

        my_proto_model = ProtoModel([num_prototypes])
        proto_layer = my_proto_model.proto_layers[0]
        protos = keras.backend.eval(proto_layer.prototypes)
        projector = my_proto_model.projectors[0]

        np.random.seed(5)
        random_encoding = np.reshape(np.random.random(num_dims), (1, num_dims))
        dist_to_protos, dist_to_latents, diff_to_protos, in_plane_point = projector.predict(random_encoding)
        # print("Model in plane point", in_plane_point)

        # Calculate these same things using numpy and check
        offset = protos[0, :]
        basis = protos[1, :] - offset
        offset_enc = random_encoding - offset
        projection = basis * np.dot(offset_enc[0], basis) / np.dot(basis, basis)
        global_projection = projection + offset
        # print("Projection", projection)
        # print("Global", global_projection)
        self.assertTrue(np.allclose(in_plane_point[0, :], global_projection))


if __name__ == '__main__':
    unittest.main()
