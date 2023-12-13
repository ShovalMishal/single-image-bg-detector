import os
import argparse
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
from prometheus_client import Enum
from scipy.spatial import cKDTree
from math import ceil
from DOTA_devkit import DOTA
from scipy.spatial.distance import cdist
from skimage.util import view_as_windows

class AlgorithmType(Enum):
    PIXELS = 1
    VIT_CLS_HEADS = 2



def parse_args():
    parser = argparse.ArgumentParser(description="apply single image BG detection for the images in source dir and"
                                                 "output results to the target dir.")
    parser.add_argument("--source", help="Path to the source directory", default='examples')
    parser.add_argument("--target", help="Path to the target directory", default='results')

    args = parser.parse_args()
    return args


def find_bottom_k_indices(x, k):
    idx = np.argpartition(x, k)[:k]  # Indices not sorted
    return idx[np.argsort(x[idx])]  # Indices sorted by value from smallest to largest


def pad_image_to_divisible(image, P):
    H, W, _ = image.shape

    # Calculate the padding required for the left and bottom sides
    pad_left = (P - (W % P)) % P
    pad_bottom = (P - (H % P)) % P

    # Pad the image with zeros
    padded_image = np.pad(image, ((0, pad_bottom), (pad_left, 0), (0, 0)), mode='constant')

    return padded_image


def calculate_patch_centers(patch_size, image_shape):
    centers = []
    for i in range(0, image_shape[0], patch_size):
        for j in range(0, image_shape[1], patch_size):
            center = (i + patch_size // 2, j + patch_size // 2)
            centers.append(center)
    return centers

def get_neighbors_indices(num_neighbors, rows, cols):
    """
  Finds the indices of the neighbors of each element in a matrix and translate the indices into flattened matrix
   representation.

  Args:
    num_neighbors: An integer specifying the number of neighbors per side.
    rows: An integer specifying the number of rows in the matrix.
    cols: An integer specifying the number of columns in the matrix.

  Returns:
    A 3D numpy array where each element contains a list of its neighbor indices as a flattened 1D array.
  """
    offsets = np.arange(-num_neighbors, num_neighbors + 1)
    i_offsets, j_offsets = np.meshgrid(offsets, offsets, indexing='ij')
    i_matrix, j_matrix = np.meshgrid(range(rows), range(cols), indexing='ij')

    # Add offsets to each element in both matrices
    i_neighbors_matrix = i_matrix.ravel()[:, None] + i_offsets.ravel()
    j_neighbors_matrix = j_matrix.ravel()[:, None] + j_offsets.ravel()

    # Create masks for valid i and j indices while satisfying both conditions
    valid_i_indices = (i_neighbors_matrix >= 0) & (i_neighbors_matrix < rows)
    valid_j_indices = (j_neighbors_matrix >= 0) & (j_neighbors_matrix < cols)

    # Combine masks using logical AND and extract valid indices
    valid_indices = valid_i_indices & valid_j_indices

    filtered_i_neigbores = [row[row_mask.astype(bool)] for row, row_mask in zip(i_neighbors_matrix, valid_indices)]
    filtered_j_neigbores = [row[row_mask.astype(bool)] for row, row_mask in zip(j_neighbors_matrix, valid_indices)]
    neighbor_indices = convert_matrix_indices_to_flattened_matrix(filtered_i_neigbores, filtered_j_neigbores, rows)
    return neighbor_indices


def convert_matrix_indices_to_flattened_matrix(i_indices, j_indices, rows):
    indices = [[i * (rows) + j for i, j in zip(curr_i_list, curr_j_list)] for curr_i_list, curr_j_list in
               zip(i_indices, j_indices)]
    return indices

def bg_dector_alg_pixels_ver(padded_image, patch_size, step_size, anns, padded_image_size):
    # calculating image's patches, kdtree, and define neighberhood for each aptch
    patches = view_as_windows(padded_image, patch_size, step=step_size)
    patches_flattened = patches.reshape(patches.shape[0] * patches.shape[1], -1).astype(np.float32) / 255.0
    kdtree = cKDTree(patches_flattened)
    neighborhoods = get_neighbors_indices(num_neighbors=1, rows=patches.shape[0], cols=patches.shape[1])
    # neighborhoods = define_neighborhoods(list(range(patches_flattened.shape[0])), patches.shape[1])

    # plotting ssd matrix between all patches
    ssd_matrix = cdist(patches_flattened, patches_flattened)
    plt.clf()
    plt.imshow(ssd_matrix)
    plt.title(f'ssd matrix for {imgid}')
    plt.colorbar()
    plt.savefig(os.path.join(target_dir, imgid, 'ssd_matrix.png'))
    np.save(os.path.join(target_dir, imgid, 'ssd_matrix.npy'), ssd_matrix)

    # calculating heat map according to different k values
    k_to_heat_map = {k: np.zeros((patches.shape[0], patches.shape[1])).flatten() for k in [2, 5, 10, 20, 50, 100, ]}
    for i, current_patch in enumerate(patches_flattened):
        current_neighbors = neighborhoods[i]
        closest_patches_distances, closest_patches_indices = kdtree.query(current_patch, patches_flattened.shape[0])
        filtered_distances = closest_patches_distances[~np.isin(closest_patches_indices, current_neighbors)]
        for k in [2, 5, 10, 20, 50, 100, ]:
            k_to_heat_map[k][i] = np.mean(filtered_distances[:k])
    for k in [2, 5, 10, 20, 50, 100, ]:
        heatmap = k_to_heat_map[k].reshape((patches.shape[0], patches.shape[1])) / k_to_heat_map[k].reshape(
            (patches.shape[0], patches.shape[1])).max()  # reshape and normalize
        # heatmap = skimage.measure.block_reduce(heatmap,
        #                                        (patch_size[0] // step_size, patch_size[1] // step_size),
        #                                        np.mean)
        plt.clf()
        plt.title(f'distance to {k}-th neighbour, image:{imgid}')
        plt.imshow(heatmap)
        plt.colorbar()
        plt.savefig(os.path.join(target_dir, imgid, f'distance_to_{k}-th_closest_neighbour.png'))

        plt.clf()
        plt.title('ground truth')
        dota_obj.showAnns(anns, imgid, 2)
        plt.savefig(os.path.join(target_dir, imgid, f'original_image_{imgid}_with_anns.png'))

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.title(f'distance to {k}-th neighbour, image:{imgid}')
        extent = 0, padded_image_size[0], 0, padded_image_size[0]
        plt.imshow(np.flipud(padded_image))
        plt.imshow(heatmap, alpha=.5, interpolation='bilinear',
                   extent=extent)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title('ground truth')
        dota_obj.showAnns(anns, imgid, 2)
        fig = plt.gcf()
        fig.set_size_inches((8, 8))
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, imgid, f'{k}-th_neighbour_distance_overlaid_on_image.png'))


def bg_detector_alg(dota_obj, imgid, target_dir, feature_type: AlgorithmType=AlgorithmType.PIXELS):
    # loading image and its anns, defining its patch size and stride
    os.makedirs(os.path.join(target_dir, imgid), exist_ok=True)
    img = dota_obj.loadImgs(imgid)[0]
    anns = dota_obj.loadAnns(imgId=imgid)
    gsd = anns[0]['gsd']
    patch_size_in_meter = (10, 10)
    step_size_in_meters = 10
    patch_size = (ceil(patch_size_in_meter[0] / gsd), ceil(patch_size_in_meter[1] / gsd), 3)
    step_size = ceil(step_size_in_meters / gsd)
    padded_image = pad_image_to_divisible(img, patch_size[0])
    padded_image_size = padded_image.shape
    print(f"Image id is: {imgid},\n image size is: {padded_image_size},\n "
          f"patch size is: {patch_size_in_meter} meters and {patch_size} pixels.")

    if feature_type==AlgorithmType.PIXELS:
        bg_dector_alg_pixels_ver(padded_image=padded_image, patch_size=patch_size, step_size=step_size, anns=anns,
                                 padded_image_size=padded_image_size)

if __name__ == '__main__':
    args = parse_args()
    source_dir = args.source
    target_dir = args.target

    dota_obj = DOTA(source_dir)
    imgids = dota_obj.getImgIds()
    for imgid in imgids:
        bg_detector_alg(dota_obj, imgid, target_dir)
