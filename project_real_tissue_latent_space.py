import tensorflow as tf
import os 
import argparse
from data_manipulation.data import Data
from models.generative.gans.PathologyGAN_Encoder import PathologyGAN_Encoder
from models.evaluation.features import *
os.umask(0o002)


parser = argparse.ArgumentParser(description='PathologyGAN fake image generator and feature extraction.')
parser.add_argument('--checkpoint', dest='checkpoint', required= True, help='Path to pre-trained weights (.ckt) of PathologyGAN.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=50, help='Batch size.')
parser.add_argument('--z_dim', dest='z_dim', type=int, default=200, help='Latent space size.')
parser.add_argument('--real_hdf5', dest='real_hdf5', required=True, type=str, default=200, help='Path for real image to encode.')
parser.add_argument('--model', dest='model', type=str, default='PathologyGAN', help='Model name.')
parser.add_argument('--img_size', dest='img_size', type=int, default=224, help='Image size for the model.')
parser.add_argument('--dataset', dest='dataset', type=str, default='vgh_nki', help='Dataset to use.')
parser.add_argument('--marker', dest='marker', type=str, default='he', help='Marker of dataset to use.')
parser.add_argument('--dbs_path', dest='dbs_path', type=str, default=None, help='Directory with DBs to use.')
parser.add_argument('--main_path', dest='main_path', type=str, default=None, help='Path for the output run.')
args = parser.parse_args()
args = parser.parse_args()
checkpoint = args.checkpoint
batch_size = args.batch_size
z_dim = args.z_dim
model = args.model
real_hdf5 = args.real_hdf5
img_size = args.img_size
dataset = args.dataset
marker = args.marker
dbs_path = args.dbs_path
main_path = args.main_path

# Paths for runs and datasets.
if dbs_path is None:
    dbs_path = os.path.dirname(os.path.realpath(__file__))
if main_path is None:
    main_path = os.path.dirname(os.path.realpath(__file__))

# Dataset information.
image_width = img_size
image_height = img_size
image_channels = 3

if img_size == 150:
	from models.generative.gans.PathologyGAN_Encoder_150 import PathologyGAN_Encoder
	from models.evaluation.features import *
elif img_size == 28:
	from models.generative.gans.C_PathologyGAN_Encoder_norm import PathologyGAN_Encoder
	from models.evaluation.features_nuance import *
else:
	

# Hyperparameters.
learning_rate_g = 1e-4
learning_rate_d = 1e-4
learning_rate_e = 1e-4
beta_1 = 0.5
beta_2 = 0.9
restore = False
regularizer_scale = 1e-4

# Model
layers_map = {512:7, 256:6, 224:5, 150:5, 128:5, 64:4, 28:2}
layers = layers_map[img_size]
alpha = 0.2
n_critic = 5
gp_coeff = .65
use_bn = False
init = 'orthogonal'
loss_type = 'relativistic gradient penalty'
noise_input_f = True
spectral = True
attention = None

data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=image_channels, batch_size=batch_size, project_path=dbs_path)

with tf.Graph().as_default():
	# Instantiate PathologyGAN Encoder.
    pathgan = PathologyGAN_Encoder(data=data, z_dim=z_dim, layers=layers, use_bn=use_bn, alpha=alpha, beta_1=beta_1, init=init, regularizer_scale=regularizer_scale, attention=attention,
    							   noise_input_f=noise_input_f, spectral=spectral, learning_rate_g=learning_rate_g, learning_rate_d=learning_rate_d, learning_rate_e=learning_rate_e, beta_2=beta_2, 
    							   n_critic=n_critic, gp_coeff=gp_coeff, loss_type=loss_type, model_name=model)

    # Take real tissue samples to encode into latent space.
    real_hdf5_path, num_samples = real_encode_from_checkpoint(model=pathgan, data=data, data_out_path=main_path, checkpoint=checkpoint, real_hdf5=real_hdf5, batches=batch_size, save_img=False)
