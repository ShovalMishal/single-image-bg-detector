import argparse
import json
import os
from enum import Enum
import numpy as np
from DOTA_devkit.DOTA import DOTA
from bg_subtraction_accord_pixels import bg_detector_alg_pixels_ver
from bg_subtraction_with_dino_vit import BGSubtractionWithDinoVit
from bg_subtractor_utils import ViTMode, ViTModelType, plot_precision_recall_curve, plot_roc_curve
from mmengine.runner import Runner
from mmengine.config import Config
import copy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="apply single image BG detection for the images in source dir and"
                                                 "output results to the target dir.")
    parser.add_argument("--source", help="Path to the source directory", default='examples')
    parser.add_argument("--target", help="Path to the target directory", default='results')
    # parser.add_argument('--patch_size_in_meters', nargs=2, default=(5, 5), type=int,
    #                     help='A tuple of two integers representing patch '
    #                          'size in meters')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the vit model.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    source_dir = args.source
    target_dir = args.target
    target_dir = os.path.join(target_dir, "pixels_based_alg")
    # patch_size_in_meters = args.patch_size_in_meters
    patch_size = args.patch_size
    dota_obj = DOTA(basepath=source_dir)
    imgids = dota_obj.getImgIds()
    mean_ap = 0
    mean_auc = 0
    counter = 0
    results_dict = {}
    performance_dict = {key: [] for key in ['imgid', 'ap', 'auc']}
    for imgid in tqdm(imgids):
        if dota_obj.loadAnns(imgId=imgid):
            heatmap, gt = bg_detector_alg_pixels_ver(imgid=imgid, target_dir=target_dir, dota_obj=dota_obj,
                                                     patch_size=patch_size, plot_result=False)
            curr_ap = plot_precision_recall_curve(gt=gt, heatmap=heatmap, title=f"Precision-Recall curve img {imgid}") # result_path=os.path.join(dino_vit_bg_subtractor.target_dir, imgid))
            curr_auc = plot_roc_curve(gt=gt, heatmap=heatmap, title=f"ROC curve img {imgid}") # result_path=os.path.join(dino_vit_bg_subtractor.target_dir, imgid))
            if np.isnan(curr_ap) or np.isnan(curr_auc):
                print(f"For image {imgid} the ap is {curr_ap} and the auc is {curr_auc}\n")
                continue
            counter += 1
            mean_ap += curr_ap
            mean_auc += curr_auc
            performance_dict['imgid'].append(imgid)
            performance_dict['ap'].append(curr_ap)
            performance_dict['auc'].append(curr_auc)
    mean_ap /= counter
    mean_auc /= counter
    results_dict['mean_ap'] = mean_ap
    results_dict['mean_auc'] = mean_auc
    results_dict['performance_dict'] = performance_dict
    with open(os.path.join(target_dir, "results.json"), 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    print(f"For examples images: averaged auc is {mean_auc} and averaged ap is {mean_ap}!")


if __name__ == '__main__':
    main()
