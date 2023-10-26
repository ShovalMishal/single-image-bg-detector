import os
import argparse
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt

from DOTA_devkit import DOTA
from scipy.spatial.distance import cdist
from skimage.util import view_as_windows


def parse_args():
    parser = argparse.ArgumentParser(description="apply single image BG detection for the images in source dir and"
                                                 "output results to the target dir.")
    parser.add_argument("--source", help="Path to the source directory", default='DOTA_devkit/example')
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


def load_image_and_annotations(dota_obj, imgid, target_dir):
    os.makedirs(os.path.join(target_dir, imgid), exist_ok=True)
    img = dota_obj.loadImgs(imgid)[0]
    anns = dota_obj.loadAnns(imgId=imgid)
    gsd = anns[0]['gsd']
    print(len(anns))
    print(img.shape)
    block_shape_in_meter = (10, 10)
    step_size_in_meters = 10
    from math import ceil
    block_shape = (ceil(block_shape_in_meter[0] / gsd), ceil(block_shape_in_meter[1] / gsd), 3)
    step_size = ceil(step_size_in_meters / gsd)
    padded_image = pad_image_to_divisible(img, block_shape[0])
    view = view_as_windows(padded_image, block_shape, step=step_size)
    flatten_view = view.reshape(view.shape[0] * view.shape[1], -1).astype(np.float32) / 255.0
    print(flatten_view.shape)
    ssd_matrix = cdist(flatten_view, flatten_view)
    plt.clf()
    plt.imshow(ssd_matrix)
    plt.title(f'ssd matrix for {imgid}')
    plt.colorbar()
    plt.savefig(os.path.join(target_dir, imgid, 'ssd_matrix.png'))
    np.save(os.path.join(target_dir, imgid, 'ssd_matrix.npy'), ssd_matrix)

    for k in [2, 5, 10, 20, 50, 100, ]:
        k_ssd_distance = np.zeros((view.shape[0], view.shape[1])).flatten()
        for idx, ssd_to_pixel in enumerate(ssd_matrix):
            k_ssd_distance[idx] = ssd_to_pixel[find_bottom_k_indices(ssd_to_pixel, k)[-1]]
        heatmap = k_ssd_distance.reshape((view.shape[0], view.shape[1])) / k_ssd_distance.reshape(
            (view.shape[0], view.shape[1])).max()
        heatmap = skimage.measure.block_reduce(heatmap,
                                               (block_shape[0] // step_size, block_shape[1] // step_size),
                                               np.mean)
        plt.clf()
        plt.title(f'distance to {k}-th neighbour, image:{imgid}')
        plt.imshow(heatmap)
        plt.colorbar()
        plt.savefig(os.path.join(target_dir, imgid, f'distance_to_{k}-th_closest_neighbour.png'))

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.title(f'distance to {k}-th neighbour, image:{imgid}')
        extent = 0, img.shape[0], 0, img.shape[0]
        plt.imshow(np.flipud(img))
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


if __name__ == '__main__':
    args = parse_args()
    source_dir = args.source
    target_dir = args.target

    dota_obj = DOTA(source_dir)
    imgids = dota_obj.getImgIds()
    for imgid in imgids:
        load_image_and_annotations(dota_obj, imgid, target_dir)
