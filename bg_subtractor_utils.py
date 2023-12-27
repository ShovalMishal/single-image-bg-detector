import os

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def pad_image_to_divisible(image, P):
    H = image.shape[0]
    W = image.shape[1]
    # Calculate the padding required for the left and bottom sides
    pad_left = (P - (W % P)) % P
    pad_bottom = (P - (H % P)) % P

    # Pad the image with zeros
    if len(image.shape) == 3:
        padded_image = np.pad(image, ((0, pad_bottom), (pad_left, 0), (0, 0)), mode='constant')
    else:
        padded_image = np.pad(image, ((0, pad_bottom), (pad_left, 0)), mode='constant')

    return padded_image


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


def calculate_patches_alg_heat_maps_for_k_values(heat_map_width, heat_map_height, flattened_pathces_matrix,
                                                 k_values=[2, 5, 10, 20, 50, 100, ]):
    neighborhoods = get_neighbors_indices(num_neighbors=1, rows=heat_map_width, cols=heat_map_height)
    kdtree = cKDTree(flattened_pathces_matrix)
    k_to_heat_map = {k: np.zeros((heat_map_width, heat_map_height)).flatten()
                     for k in k_values}
    for i, current_patch in enumerate(flattened_pathces_matrix):
        current_neighbors = neighborhoods[i]
        closest_patches_distances, closest_patches_indices = kdtree.query(current_patch,
                                                                          flattened_pathces_matrix.shape[0])
        filtered_distances = closest_patches_distances[~np.isin(closest_patches_indices, current_neighbors)]
        for k in k_values:
            k_to_heat_map[k][i] = np.mean(filtered_distances[:k])
    return k_to_heat_map


def plot_k_heat_maps(k_to_heat_map, heat_map_width, heat_map_height, target_dir, imgid, anns, dota_obj, orig_image,
                     k_values=[2, 5, 10, 20, 50, 100, ]):
    orig_image_size = orig_image.shape

    for k in k_values:
        filenames = [f'distance to {k}-th neighbour, image:{imgid}',
                     f'{k}-th_neighbour_distance_overlaid_on_image_{imgid}']
        heatmap = k_to_heat_map[k].reshape((heat_map_width, heat_map_height)) / k_to_heat_map[k].reshape(
            (heat_map_width, heat_map_height)).max()
        plot_heat_map(heatmap=heatmap,
                      imgid=imgid, target_dir=target_dir, dota_obj=dota_obj, anns=anns, orig_image_size=orig_image_size,
                      orig_image=orig_image, filenames=filenames)


def plot_heat_map(heatmap, imgid, target_dir, dota_obj, anns, orig_image_size,
                  orig_image, filenames=[]):

    plt.clf()
    plt.title(filenames[0])
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(os.path.join(target_dir, imgid, f'{filenames[0]}.png'))

    plt.clf()
    plt.title('ground truth')
    dota_obj.showAnns(anns, imgid, 2)
    plt.savefig(os.path.join(target_dir, imgid, f'original_image_{imgid}_with_anns.png'))

    plt.clf()
    plt.subplot(1, 2, 1)
    plt.title(f'{filenames[0]}')
    extent = 0, orig_image_size[0], 0, orig_image_size[0]
    plt.imshow(orig_image)
    plt.imshow(heatmap, alpha=.5, interpolation='bilinear',
               extent=extent)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('ground truth')
    dota_obj.showAnns(anns, imgid, 2)
    fig = plt.gcf()
    fig.set_size_inches((8, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, imgid, f'{filenames[1]}.png'))