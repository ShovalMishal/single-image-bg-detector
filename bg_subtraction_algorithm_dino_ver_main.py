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


class AlgorithmType(Enum):
    PIXELS = "pixels"
    VIT = "vit"


def create_dataloader(cfg):
    test_dataloader = cfg.get("dataloader")
    dataloader_cfg = copy.deepcopy(test_dataloader)
    dataloader_cfg["dataset"]["_scope_"] = "mmrotate"
    data_loader = Runner.build_dataloader(dataloader_cfg, seed=123456)
    return data_loader


def parse_args():
    parser = argparse.ArgumentParser(description="apply single image BG detection for the images in source dir and"
                                                 "output results to the target dir.")
    parser.add_argument("--source", help="Path to the source directory", default='examples')
    parser.add_argument("--target", help="Path to the target directory", default='results')
    parser.add_argument('--bg_subtraction_alg_type', default='vit', type=str,
                        choices=['pixels', 'vit'], help='Optional background subtraction algorithms.')
    parser.add_argument('--patch_size_in_meters', nargs=2, default=(5, 5), type=int,
                        help='A tuple of two integers representing patch '
                             'size in meters')
    parser.add_argument('--vit_patch_size', default=8, type=int, help='Patch resolution of the vit model.')
    parser.add_argument('--vit_arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument("--vit_image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument("--vit_threshold", type=float, default=None, help="""We visualize masks
            obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--pretrained_weights', default='/home/shoval/Downloads/vit_mc_checkpoint300.pth', type=str,
                        help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--vit_model_mode', default='last_block_output', type=str,
                        choices=["class_token_self_attention", "last_block_output"],
                        help='Optional model mode.')
    parser.add_argument('--vit_model_type', default='dino_vit', type=str,
                        choices=["dino_vit", "dino_mc_vit"], help='Optional model type.')
    parser.add_argument('--dataset_type', default='train', type=str,
                        choices=["train", "val"], help='Dataset type.')
    parser.add_argument('--dataset_config', default='./dataset_config.py', type=str, help='Dataset config file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    source_dir = args.source
    target_dir = args.target
    vit_patch_size = args.vit_patch_size
    vit_arch = args.vit_arch
    vit_image_size = args.vit_image_size
    vit_threshold = args.vit_threshold
    pretrained_weights = args.pretrained_weights
    checkpoint_key = args.checkpoint_key
    vit_model_mode = args.vit_model_mode
    vit_model_type = args.vit_model_type
    dataset_type = args.dataset_type
    dataset_config = args.dataset_config
    dota_obj = DOTA(basepath=source_dir)
    dino_vit_bg_subtractor = BGSubtractionWithDinoVit(target_dir=target_dir, vit_patch_size=vit_patch_size,
                                                      vit_arch=vit_arch, vit_image_size=vit_image_size,
                                                      dota_obj=dota_obj, threshold=vit_threshold,
                                                      pretrained_weights=pretrained_weights,
                                                      checkpoint_key=checkpoint_key,
                                                      model_mode=vit_model_mode,
                                                      model_type=vit_model_type)
    # create dataloader for validation set
    # cfg = Config.fromfile(dataset_config)
    # data_loader = create_dataloader(cfg)

    imgids = dota_obj.getImgIds()
    mean_ap = 0
    mean_auc = 0
    counter = 0
    results_dict = {}
    performance_dict = {key: [] for key in ['imgid', 'ap', 'auc']}
    for imgid in tqdm(imgids):
        if dota_obj.loadAnns(imgId=imgid):
            heatmap, gt = dino_vit_bg_subtractor.run_on_image_path(imgid=imgid, plot_result=False)
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
    with open(os.path.join(dino_vit_bg_subtractor.target_dir, "results.json"), 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    print(f"For examples images: averaged auc is {mean_auc} and averaged ap is {mean_ap}!")


if __name__ == '__main__':
    main()
