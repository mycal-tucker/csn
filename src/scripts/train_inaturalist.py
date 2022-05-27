import numpy as np
import tensorflow as tf

from src.data_parsing.inaturalist.inaturalist_data import get_inat_data
from src.models.proto_model import ProtoModel
from src.utils.gpu import set_gpu_config
from src.utils.eval import trees_match, graph_edit_dist
from src.utils.plotting import plot_mst
from src.utils.to_files import write_to_file

set_gpu_config()
print(tf.test.is_gpu_available())
tf.compat.v1.disable_eager_execution()

batch_size = 32  # 64 is good

# latent_dim = 512
latent_dim = 1024
noise_level = 0.0
use_class_and_group = True
class_only = False
group_only = False

train_data = get_inat_data('train', batch_size=batch_size)
val_data = get_inat_data('val', batch_size=batch_size)

train_data.reset_state()
val_data.reset_state()

# level_sizes = [3, 4, 9, 34, 57, 72, 1010]
output_sizes = [3, 4, 9, 34, 57, 72, 1010]
duplication_factors = [1 for _ in output_sizes]
# classification_weights = [10 for _ in output_sizes]
classification_weights = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 10]
# proto_dist_weights = [0.01 for _ in output_sizes]
# feature_dist_weights = [0.01 for _ in output_sizes]
proto_dist_weights = [0.0 for _ in output_sizes]
feature_dist_weights = [0.0 for _ in output_sizes]
# output_sizes = [1010]
# duplication_factors = [1]
# classification_weights = [10]
# proto_dist_weights = [0.01]
# feature_dist_weights = [0.01]
disentangle_weights = 0
kl_losses = [0 for _ in output_sizes]

proto_model = ProtoModel(output_sizes, duplication_factors=duplication_factors, input_size=224 * 224 * 3, decode_weight=0,
                         classification_weights=classification_weights, proto_dist_weights=proto_dist_weights,
                         feature_dist_weights=feature_dist_weights, disentangle_weights=disentangle_weights,
                         kl_losses=kl_losses, latent_dim=latent_dim, align_fn=tf.reduce_mean, network_type='inat')
# proto_model.load('../saved_models/inat/run2/epoch17/')
proto_model.load('../saved_models/inat/run2/epoch19/')
proto_model.train_with_dataflow(train_data, val_df=val_data, batch_size=batch_size)
proto_model.eval_with_dataflow(val_data, batch_size=batch_size)
