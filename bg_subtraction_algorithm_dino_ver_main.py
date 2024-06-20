import argparse
import json
import os
from enum import Enum
import numpy as np
from DOTA_devkit.DOTA import DOTA
from bg_subtraction_with_dino_vit import BGSubtractionWithDinoVit
from bg_subtractor_utils import plot_precision_recall_curve, plot_roc_curve, \
    extract_patches_accord_heatmap, mask_img, assign_predicted_boxes_to_gt, calculate_performance_measures, \
    assign_predicted_boxes_to_gt_accord_intersection
from mmengine.runner import Runner
from mmengine.config import Config
import copy
from tqdm import tqdm
from mmdet.registry import TASK_UTILS

import torch
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
    parser.add_argument('--vit_arch', default='vit_base', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument("--vit_image_size", default=(512, 512), type=int, nargs="+", help="Resize image.")
    parser.add_argument("--proposals_sizes", default=((17, 17), (21, 11), (11, 21)), type=int, nargs="+", help="Proposals optional sizes.")
    parser.add_argument("--vit_threshold", type=float, default=None, help="""We visualize masks
            obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--pretrained_weights', default='/storage/shoval/vit_representation_experiment/vit_mc_checkpoint300.pth', type=str,
                        help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--vit_model_mode', default='key', type=str,
                        choices=["class_token_self_attention", "last_block_output", "key", "query", "value"],
                        help='Optional model mode.')
    parser.add_argument('--layer_num', default=11, type=int,
                        help='layer number for representation.')
    parser.add_argument('--attention_num', default=0, type=int,
                        help='attention head number - 0 means average attention, 1-12 means the suitable head.')
    parser.add_argument('--vit_model_type', default='dino_vit', type=str,
                        choices=["dino_vit", "dino_mc_vit"], help='Optional model type.')
    parser.add_argument('--dataset_type', default='train', type=str,
                        choices=["train", "val"], help='Dataset type.')
    parser.add_argument('--config', default='./config.py', type=str, help='Dataset config file.')
    args = parser.parse_args()
    return args


def calculate_pixel_based_performance_measures(source_dir, target_dir, vit_patch_size, vit_image_size, vit_arch,
                                               vit_threshold, pretrained_weights, checkpoint_key, vit_model_mode,
                                               vit_model_type, attention_num, layer_num):
    print("starting..")
    dota_obj = DOTA(basepath=source_dir)
    print("dota_obj is created!")
    dino_vit_bg_subtractor = BGSubtractionWithDinoVit(target_dir=target_dir, vit_patch_size=vit_patch_size,
                                                      vit_arch=vit_arch, vit_image_size=vit_image_size,
                                                      dota_obj=dota_obj, threshold=vit_threshold,
                                                      pretrained_weights=pretrained_weights,
                                                      checkpoint_key=checkpoint_key,
                                                      model_mode=vit_model_mode,
                                                      model_type=vit_model_type,
                                                      attention_num=attention_num,
                                                      layer_num=layer_num)
    imgids = dota_obj.getImgIds()
    mean_ap = 0
    mean_auc = 0
    counter = 0
    results_dict = {}
    performance_dict = {key: [] for key in ['imgid', 'ap', 'auc']}
    print("start iterating dataset..")
    for img_ind, imgid in tqdm(enumerate(imgids)):
        if dota_obj.loadAnns(imgId=imgid):
            heatmap, gt = dino_vit_bg_subtractor.run_on_image_path(imgid=imgid, plot_result=False)
            curr_ap = plot_precision_recall_curve(gt=gt, heatmap=heatmap,
                                                  title=f"Precision-Recall curve img {imgid}")  # result_path=os.path.join(dino_vit_bg_subtractor.target_dir, imgid))
            curr_auc = plot_roc_curve(gt=gt, heatmap=heatmap,
                                      title=f"ROC curve img {imgid}")  # result_path=os.path.join(dino_vit_bg_subtractor.target_dir, imgid))
            if np.isnan(curr_ap) or np.isnan(curr_auc):
                print(f"For image {imgid} the ap is {curr_ap} and the auc is {curr_auc}\n")
                continue
            counter += 1
            mean_ap += curr_ap
            mean_auc += curr_auc
            performance_dict['imgid'].append(imgid)
            performance_dict['ap'].append(curr_ap)
            performance_dict['auc'].append(curr_auc)
    print("saving results..")
    mean_ap /= counter
    mean_auc /= counter
    results_dict['mean_ap'] = mean_ap
    results_dict['mean_auc'] = mean_auc
    results_dict['performance_dict'] = performance_dict
    with open(os.path.join(dino_vit_bg_subtractor.target_dir, f"results_head_{attention_num}.json"), 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    print(f"For examples images: averaged auc is {mean_auc} and averaged ap is {mean_ap}!")
    return


def calculate_boxes_based_performance_measures(source_dir, target_dir, vit_patch_size, vit_image_size, vit_arch,
                                               vit_threshold, pretrained_weights, checkpoint_key, vit_model_mode,
                                               vit_model_type, config, proposals_sizes,layer_num, attention_num):
    dota_obj = DOTA(basepath=source_dir)
    dino_vit_bg_subtractor = BGSubtractionWithDinoVit(target_dir=target_dir, vit_patch_size=vit_patch_size,
                                                      vit_arch=vit_arch, vit_image_size=vit_image_size,
                                                      dota_obj=dota_obj, threshold=vit_threshold,
                                                      pretrained_weights=pretrained_weights,
                                                      checkpoint_key=checkpoint_key,
                                                      model_mode=vit_model_mode,
                                                      model_type=vit_model_type, layer_num=layer_num,
                                                      attention_num=attention_num)
    # create dataloader for validation set
    cfg = Config.fromfile(config)
    data_loader = create_dataloader(cfg)
    all_labels = data_loader.dataset.METAINFO['classes']
    bbox_assigner = TASK_UTILS.build(cfg["patches_assigner"])
    counter = 0
    predicted_boxes_scores = torch.tensor([])
    predicted_boxes_labels = torch.tensor([])
    performance_dict = {key: [] for key in ['ap', 'auc']}
    # by default plot 100 examples for review
    plot = True
    for batch in tqdm(data_loader):
        # measure performance for images with objects only
        if batch['data_samples'][0].gt_instances.bboxes.size()[0] > 0:
            counter += 1
            heatmap = dino_vit_bg_subtractor.run_on_image_tensor(img=batch['inputs'][0])
            curr_img_matches = torch.tensor([])
            gts = np.array([])
            for proposal_bbox_size in proposals_sizes:
                title = "square" if proposal_bbox_size[0] == proposal_bbox_size[1] else (
                    "vertical" if proposal_bbox_size[0] > proposal_bbox_size[1] else "horizontal")
                predicted_patches, mask, scores = extract_patches_accord_heatmap(heatmap=heatmap,
                                                                                 img_id=batch['data_samples'][0].img_id,
                                                                                 patch_size=proposal_bbox_size,
                                                                                 plot=plot, title=title)
                gt_labels, gt_inds, dt_labels, dt_match = assign_predicted_boxes_to_gt(bbox_assigner=bbox_assigner,
                                                                                       predicted_boxes=predicted_patches,
                                                                                       data_batch=batch,
                                                                                       all_labels=all_labels,
                                                                                       img_id=batch['data_samples'][
                                                                                           0].img_id,
                                                                                       dota_obj=dota_obj,
                                                                                       heatmap=heatmap,
                                                                                       patch_size=proposal_bbox_size,
                                                                                       plot=plot, title=title)
                predicted_boxes_scores = torch.cat((predicted_boxes_scores, torch.tensor(scores)))
                predicted_boxes_labels = torch.cat((predicted_boxes_labels, dt_labels))
                if counter == 100:
                    plot = False
                curr_img_matches = torch.cat((curr_img_matches, dt_match))
                gts = np.concatenate((gts, gt_inds))
            tp = torch.sum(torch.unique(curr_img_matches) > 0).item()
            gt = len(np.unique(gts))
            # add undiscovered bboxes to statistics
            unmatched_boxes_num = gt - tp
            predicted_boxes_scores = torch.cat((predicted_boxes_scores, torch.tensor([0] * unmatched_boxes_num)))
            predicted_boxes_labels = torch.cat((predicted_boxes_labels, torch.tensor([1] * unmatched_boxes_num)))
            print(f"Discovered {tp} bbox out of {gt} for img {batch['data_samples'][0].img_id}\n")

    ap, auc = calculate_performance_measures(gt=predicted_boxes_labels, scores=predicted_boxes_scores)
    performance_dict['ap'] = ap
    performance_dict['auc'] = auc
    with open(os.path.join(dino_vit_bg_subtractor.target_dir, "results.json"), 'w') as json_file:
        json.dump(performance_dict, json_file, indent=4)
    print(f"For examples images: auc is {auc} and ap is {ap}!")


def find_best_filtering_boxes_threshold(source_dir, target_dir, vit_patch_size, vit_image_size, vit_arch,
                                               vit_threshold, pretrained_weights, checkpoint_key, vit_model_mode,
                                               vit_model_type, config, proposals_sizes):
    dota_obj = DOTA(basepath=source_dir)
    dino_vit_bg_subtractor = BGSubtractionWithDinoVit(target_dir=target_dir, vit_patch_size=vit_patch_size,
                                                      vit_arch=vit_arch, vit_image_size=vit_image_size,
                                                      dota_obj=dota_obj, threshold=vit_threshold,
                                                      pretrained_weights=pretrained_weights,
                                                      checkpoint_key=checkpoint_key,
                                                      model_mode=vit_model_mode,
                                                      model_type=vit_model_type)

    # create dataloader for validation set
    cfg = Config.fromfile(config)
    data_loader = create_dataloader(cfg)
    all_labels = data_loader.dataset.METAINFO['classes']
    bbox_assigner = TASK_UTILS.build(cfg["patches_assigner"])
    counter = 0
    performance = {}
    for threshold in range(75, 96, 5):
        num_of_gt = 0
        num_of_tp = 0
        num_of_fp = 0
        for batch in tqdm(data_loader):
            # measure performance for images with objects only
            if batch['data_samples'][0].gt_instances.bboxes.size()[0] > 0:
                counter += 1
                heatmap = dino_vit_bg_subtractor.run_on_image_tensor(img=batch['inputs'][0])
                predicted_patches, mask, scores = extract_patches_accord_heatmap(heatmap=heatmap,
                                                                                 img_id=batch['data_samples'][0].img_id,
                                                                                 patch_size=proposals_sizes, plot=True,
                                                                                 threshold_percentage=threshold)
                tp, gt, fp = assign_predicted_boxes_to_gt_accord_intersection(bbox_assigner=bbox_assigner,
                                                          predicted_boxes=predicted_patches,
                                                          data_batch=batch, all_labels=all_labels,
                                                          img_id=batch['data_samples'][0].img_id,
                                                          dota_obj=dota_obj, heatmap=heatmap,
                                                          patch_size=proposals_sizes, plot=False)
                num_of_gt += gt
                num_of_tp += tp
                num_of_fp += fp
        curr_performance = {}
        curr_performance['precision'] = num_of_tp / (num_of_tp + num_of_fp)
        curr_performance['recall'] = num_of_tp / num_of_gt
        print(
            f"For threshold {threshold}: precision is {curr_performance['precision']} and recall is {curr_performance['recall']}\n")
        performance[threshold] = curr_performance
    with open(os.path.join(dino_vit_bg_subtractor.target_dir, "results.json"), 'w') as json_file:
        json.dump(performance, json_file, indent=4)


def main():
    args = parse_args()
    source_dir = args.source
    dataset_type = args.dataset_type
    target_dir = os.path.join(args.target, dataset_type)
    vit_patch_size = args.vit_patch_size
    vit_arch = args.vit_arch
    vit_image_size = args.vit_image_size
    vit_threshold = args.vit_threshold
    pretrained_weights = args.pretrained_weights
    checkpoint_key = args.checkpoint_key
    vit_model_mode = args.vit_model_mode
    vit_model_type = args.vit_model_type
    attention_num = args.attention_num
    layer_num = args.layer_num
    # calculate_boxes_based_performance_measures(source_dir=source_dir, target_dir=target_dir,
    #                                            vit_patch_size=vit_patch_size, vit_image_size=vit_image_size,
    #                                            vit_arch=vit_arch, vit_threshold=vit_threshold,
    #                                            pretrained_weights=pretrained_weights,
    #                                            checkpoint_key=checkpoint_key, vit_model_mode=vit_model_mode,
    #                                            vit_model_type=vit_model_type, attention_num=attention_num,
    #                                            layer_num=layer_num)
    calculate_pixel_based_performance_measures(source_dir=source_dir, target_dir=target_dir,
                                               vit_patch_size=vit_patch_size, vit_image_size=vit_image_size,
                                               vit_arch=vit_arch, vit_threshold=vit_threshold,
                                               pretrained_weights=pretrained_weights,
                                               checkpoint_key=checkpoint_key, vit_model_mode=vit_model_mode,
                                               vit_model_type=vit_model_type, attention_num=attention_num,
                                               layer_num=layer_num)


if __name__ == '__main__':
    main()
