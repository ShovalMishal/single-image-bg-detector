import os
from math import ceil
import numpy as np
from scipy.spatial.distance import cdist
from skimage.util import view_as_windows
import matplotlib.pyplot as plt

from bg_subtractor_utils import pad_image_to_divisible, get_neighbors_indices, \
    calculate_patches_alg_heat_maps_for_k_values, plot_k_heat_maps


def bg_detector_alg_pixels_ver(imgid, target_dir, dota_obj, patch_size_in_meter):
    # loading image and its anns, defining its patch size and stride
    target_dir = os.path.join(target_dir, "pixels_based_alg")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, imgid), exist_ok=True)
    img = dota_obj.loadImgs(imgid)[0]
    anns = dota_obj.loadAnns(imgId=imgid)
    gsd = dota_obj.loadGsd(imgId=imgid)

    patch_size = (ceil(patch_size_in_meter[0] / gsd), ceil(patch_size_in_meter[1] / gsd), 3)
    padded_image = pad_image_to_divisible(img, patch_size[0])
    padded_image_size = padded_image.shape
    print(f"Image id is: {imgid},\n image size is: {padded_image_size},\n "
          f"patch size is: {patch_size_in_meter} meters and {patch_size} pixels.")
    # calculating image's patches, kdtree, and define neighberhood for each patch
    patches = view_as_windows(padded_image, patch_size, step=patch_size[0])
    patches_flattened = patches.reshape(patches.shape[0] * patches.shape[1], -1).astype(np.float32) / 255.0

    # plotting ssd matrix between all patches
    ssd_matrix = cdist(patches_flattened, patches_flattened)
    plt.clf()
    plt.imshow(ssd_matrix)
    plt.title(f'ssd matrix for {imgid}')
    plt.colorbar()
    plt.savefig(os.path.join(target_dir, imgid, 'ssd_matrix.png'))
    np.save(os.path.join(target_dir, imgid, 'ssd_matrix.npy'), ssd_matrix)

    # calculating heat map according to different k values
    k_to_heat_map = calculate_patches_alg_heat_maps_for_k_values(heat_map_width=patches.shape[0],
                                                                 heat_map_height=patches.shape[1],
                                                                 flattened_pathces_matrix=patches_flattened)
    plot_k_heat_maps(k_to_heat_map=k_to_heat_map, heat_map_width=patches.shape[0], heat_map_height=patches.shape[1],
                     target_dir=target_dir, imgid=imgid, anns=anns, dota_obj=dota_obj,
                     orig_image=np.flipud(padded_image))
