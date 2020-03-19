from sklearn import mixture
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


def draw_ellipse(position, covariance, cm, label, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        nsig = 2
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs, facecolor=cm(label)))
        
def plot_gmm(gmm, X, label=True, elip=True):
    cm = plt.cm.get_cmap('viridis_r', 7)
    fig, big_axes = plt.subplots(figsize=(10,10), nrows=1, ncols=1, sharex=True, sharey=True)
    labels = gmm.fit(X).predict(X)
    if label:
        big_axes.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap=cm, zorder=2)
    else:
        big_axes.scatter(X[:, 0], X[:, 1], s=5, zorder=2)

    if elip:
        w_factor = 0.2 / gmm.weights_.max()
        for label, data in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
            pos, covar, w = data
            draw_ellipse(pos, covar, alpha=w*w_factor, cm=cm, label=label)
    
    return labels

def combine_images(imgs, spacing=0.07):
    a, height, width, channels = imgs.shape
    offset_h = int(spacing*height)
    offset_w = int(spacing*width)
    
    grid = np.ones((2*height+offset_h, 2*width+offset_w, channels))
    grid[0:height, 0:width, :] = imgs[0, :, :, :]/255.
    grid[0:height, width+offset_w:, :] = imgs[1, :, :, :]/255.
    grid[height+offset_h:, :width] = imgs[2, :, :, :]/255.
    grid[height+offset_h:, width+offset_w:, :] = imgs[3, :, :, :]/255.
    return grid

def get_images_gmm(labels):
    labels_u = np.unique(labels)
    l_ind = OrderedDict()
    for l in labels_u:
        l_ind[l] = np.argwhere(labels==l)
    return l_ind

def get_images_label(l_ind, label, imgs):
    indeces = l_ind[label][:4]
    return imgs[indeces, :, :, :]

def find_gaussian(gmm, xy):
    xy = np.array(xy)
    distance = np.Infinity
    min_mean = None
    min_label = None
    for label, mean in enumerate(gmm.means_):
        new = np.sqrt(np.sum(np.square(mean-xy)))
        if new < distance: 
            distance = new
            min_mean = mean
            min_label = label
    return min_mean, min_label

def find_linear_interpolations(start, end, gmm, n_points):
    distance_start = np.Infinity
    distance_end = np.Infinity
    min_start_mean = None
    min_end_mean = None
    
    for label, mean in enumerate(gmm.means_):
        start_new = np.sqrt(np.sum(np.square(mean-start)))
        end_new = np.sqrt(np.sum(np.square(mean-end)))
        if start_new < distance_start: 
            distance_start = start_new
            min_start_mean = mean
            start_label = label
        if end_new < distance_end: 
            distance_end = end_new
            min_end_mean = mean
            end_label = label
            
    return np.linspace(min_start_mean, min_end_mean, n_points)

def find_closest_real(interpolation_points, gmm_embedding, dataset_img):
    inter_indeces_real = list()
    inter_imgs_real = list()
    for point in interpolation_points:        
        distance = np.Infinity
        for real_index in range(gmm_embedding.shape[0]):
            real_point = gmm_embedding[real_index]
            new = np.sqrt(np.sum(np.square(point-real_point)))
            if new < distance:
                distance = new
                closes_point = real_index
        inter_indeces_real.append(closes_point)
        inter_imgs_real.append(dataset_img[closes_point])
    return inter_indeces_real, inter_imgs_real
