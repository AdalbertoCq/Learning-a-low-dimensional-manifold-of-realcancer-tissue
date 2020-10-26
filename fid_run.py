from models.evaluation.features import *
from data_manipulation.data import Data
from models.score.score import Scores
from collections import OrderedDict
import tensorflow as tf
import argparse
import glob
import os


parser = argparse.ArgumentParser(description='PathologyGAN fake image generator.')
parser.add_argument('--num_samples', dest='num_samples', default=1000,      type=int, required=False, help='Number of images to generate.')
parser.add_argument('--batch_size',  dest='batch_size',  default=50,        type=int, required=False, help='Batch size.')
parser.add_argument('--pathgan_run', dest='pathgan_run', default=None,      type=str, required=True, help='Directory with the PathologyGAN run.')
parser.add_argument('--main_path',   dest='main_path',   default=None,      type=str, help='Path for the output run.')
parser.add_argument('--dbs_path',    dest='dbs_path',    default=None,      type=str, help='Directory with DBs to use.')
parser.add_argument('--img_size',    dest='img_size',    default=224,       type=int, help='Image size for the model.')
parser.add_argument('--img_ch',      dest='img_ch',      default=3,         type=int, help='Number of channels for the model.')
parser.add_argument('--dataset',     dest='dataset',     default='vgh_nki', type=str, help='Dataset to use.')
parser.add_argument('--marker',      dest='marker',      default='he',      type=str, help='Marker of dataset to use.')
args = parser.parse_args()
pathgan_run_path = args.pathgan_run
num_samples      = args.num_samples
batch_size       = args.batch_size
dataset          = args.dataset
marker           = args.marker
image_height     = args.img_size
image_width      = args.img_size
img_ch           = args.img_ch
main_path        = args.main_path
dbs_path         = args.dbs_path

# Main paths for data output and databases.
if dbs_path is None:
	dbs_path = os.path.dirname(os.path.realpath(__file__))
if main_path is None:
	main_path = os.path.dirname(os.path.realpath(__file__))


# Get generated images. 
def get_epoch_images(path):
	evaluation_path = os.path.join(path, 'evaluation')
	epoch_path = os.path.join(evaluation_path, 'epoch_*')
	hdf5_path = os.path.join(epoch_path, 'hdf5_epoch_*_generated_images.h5')	
	print(hdf5_path)
	hdf5 = glob.glob(hdf5_path)
	return hdf5

def dump_results(path, results):
	fid_path = os.path.join(path, 'fid_results.txt')
	with open(fid_path, 'w') as content:
		for epoch in results:
			content.write('%s\n' % epoch)
			for data_type in results[epoch]:
				content.write('\tFID %s: %s  \n' % (data_type, results[epoch][data_type]))
			content.write('\n')


# Load dataset to compare against.
data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width, n_channels=img_ch, batch_size=batch_size, project_path=dbs_path, labels=None)

# Grab train, validation, and test sets.
real_hdf5s = real_samples(data=data, data_output_path=main_path, num_samples=10000, save_img=False)
generated_hdf5 = get_epoch_images(path=pathgan_run_path)
hdf5s = list()
hdf5s.extend(sorted(real_hdf5s, key=os.path.getmtime))
hdf5s.extend(generated_hdf5)

# Get Inception features for given sets.
with tf.Graph().as_default():
	features = inception_tf_feature_activations(hdf5s=hdf5s, input_shape=data.training.shape[1:], batch_size=batch_size)

# Reassign hdf5 files 
num_real = len(real_hdf5s)
num_gene = len(generated_hdf5)
real_features =  features[:num_real]
gene_features =  features[num_real:]


results = OrderedDict()
# Check distribution difference between real subsets.
i_real = 0
for real_hdf5_i in real_features:	
	j_real = 0
	data_type_i = real_hdf5_i.split('_')[-3]
	results[data_type_i] = OrderedDict()
	for real_hdf5_j in real_features:	
		if real_hdf5_i == real_hdf5_j: 
			continue
		data_type_j = real_hdf5_j.split('_')[-3]
		with tf.Graph().as_default():
			scores = Scores(real_hdf5_i, real_hdf5_j, data_type_i, data_type_j, k=1, display=False)
			scores.run_scores()
			results[data_type_i][data_type_j] = scores.fid
		j_real += 1
	i_real += 1

# Check distribution difference between real subsets and generated.
i_real = 0
for gen_hdf5 in gene_features:
	epoch = gen_hdf5.split('/')[-2]
	results[epoch] = OrderedDict()
	for real_hdf5 in real_features:	
		data_type = real_hdf5.split('_')[-3]
		with tf.Graph().as_default():
			scores = Scores(real_hdf5, gen_hdf5, data_type, epoch, k=1, display=False)
			scores.run_scores()
			results[epoch][data_type] = scores.fid
	i_real += 1

dump_results(path=pathgan_run_path, results=results)

