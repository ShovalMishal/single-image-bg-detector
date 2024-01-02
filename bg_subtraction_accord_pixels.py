import os
from math import ceil
import numpy as np
from scipy.spatial.distance import cdist
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from torch import nn
import torch
from bg_subtractor_utils import calculate_patches_alg_heat_maps_for_k_values, plot_k_heat_maps, create_gt_attention


def bg_detector_alg_pixels_ver(imgid, target_dir, dota_obj, patch_size, patch_size_in_meter=None, plot_result=False):
    # loading image and its anns, defining its patch size and stride
    os.makedirs(target_dir, exist_ok=True)

    img = dota_obj.loadImgs(imgid)[0]
    img_size = img.shape
    anns = dota_obj.loadAnns(imgId=imgid)
    gsd = dota_obj.loadGsd(imgId=imgid)
    gt_attention = create_gt_attention(anns=anns, image_size=img_size[:-1])
    if patch_size_in_meter:
        patch_size = (ceil(patch_size_in_meter[0] / gsd), ceil(patch_size_in_meter[1] / gsd), 3)
    else:
        patch_size = (patch_size, patch_size, 3)
    w, h = img.shape[0] - img.shape[0] % patch_size[0], img.shape[1] - img.shape[1] % patch_size[1]
    padded_image = img[:w, :h, :]
    padded_image_size = padded_image.shape
    # print(f"Image id is: {imgid},\n image size is: {padded_image_size},\n "
    #       f"patch size is: {patch_size} pixels.")
    # calculating image's patches, kdtree, and define neighberhood for each patch
    patches = view_as_windows(padded_image, patch_size, step=patch_size[0])
    patches_flattened = patches.reshape(patches.shape[0] * patches.shape[1], -1).astype(np.float32) / 255.0

    # plotting ssd matrix between all patches
    if plot_result:
        os.makedirs(os.path.join(target_dir, imgid), exist_ok=True)
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
    if plot_result:
        plot_k_heat_maps(k_to_heat_map=k_to_heat_map, heat_map_width=patches.shape[0], heat_map_height=patches.shape[1],
                         target_dir=target_dir, imgid=imgid, anns=anns, dota_obj=dota_obj,
                         orig_image=np.flipud(padded_image))
    returned_heat_map = k_to_heat_map[20].reshape((patches.shape[0], patches.shape[1])) / k_to_heat_map[20].reshape(
        (patches.shape[0], patches.shape[1])).max()
    returned_heat_map = \
        nn.functional.interpolate(torch.tensor(returned_heat_map).unsqueeze(dim=0).unsqueeze(dim=0),
                                  scale_factor=patch_size[0],
                                  mode="nearest")[0][0]
    return returned_heat_map.numpy(), gt_attention
