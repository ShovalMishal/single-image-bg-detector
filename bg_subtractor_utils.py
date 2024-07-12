import json
import math
import os
from enum import Enum
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, average_precision_score, roc_auc_score
from torchvision import transforms as pth_transforms
from PIL import Image
from mmdet.structures.bbox import HorizontalBoxes
from mmengine.structures import InstanceData
from mmdet.models.utils import unpack_gt_instances
import torch.nn.functional as F
from mmrotate.structures.bbox import RotatedBoxes, rbox2qbox
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ViTMode(Enum):
    CLS_SELF_ATTENTION = "class_token_self_attention"
    LAST_BLOCK_OUTPUT = "last_block_output"
    KEY = "key"
    VALUE = "value"
    QUERY = "query"


class ViTModelType(Enum):
    DINO_VIT = "dino_vit"
    DINO_MC = "dino_mc_vit"


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
        filenames = [f'distance to {k}-th neighbour',
                     f'{k}-th_neighbour_distance_overlaid_on_image']
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
    # plt.colorbar()
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
    # plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('ground truth')
    dota_obj.showAnns(anns, imgid, 2)
    fig = plt.gcf()
    fig.set_size_inches((8, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, imgid, f'{filenames[1]}.png'))


def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array


def create_gt_attention(anns, image_size, scale_factor=1):
    gt_attention = np.zeros(image_size)
    for ann in anns:
        transposed_tuples = list(zip(*ann["poly"]))
        min_values = [max(0, int(min(values) * scale_factor)) for values in transposed_tuples]
        max_values = [min(int(max(values) * scale_factor), image_size[0]) for values in transposed_tuples]
        gt_attention[min_values[1]:max_values[1], min_values[0]:max_values[0]] = np.ones(
            (max_values[1] - min_values[1], max_values[0] - min_values[0]))
    return gt_attention


def plot_precision_recall_curve(gt, heatmap, title="", result_path: str = ""):
    heatmap = normalize_array(heatmap).flatten()
    gt = gt.flatten()
    # plot precision recall curve
    # print(f"Calculating precision recall curve {title}...")
    precision, recall, _ = metrics.precision_recall_curve(gt.tolist(), heatmap.tolist())
    ap = average_precision_score(gt, heatmap)
    # print("AP val is " + str(ap))
    if result_path:
        display = PrecisionRecallDisplay.from_predictions(gt.tolist(),
                                                          heatmap.tolist(),
                                                          name=f"ood vs id",
                                                          color="darkorange"
                                                          )
        _ = display.ax_.set_title(title)
        plt.savefig(result_path + f"/{title}.png")
    return ap


def plot_roc_curve(gt, heatmap, title="", result_path: str = ""):
    heatmap = normalize_array(heatmap).flatten()
    gt = gt.flatten()
    # plot roc curve
    # print(f"Calculating AuC for {title}...")
    fpr, tpr, thresholds = metrics.roc_curve(gt.tolist(), heatmap.tolist())
    auc = metrics.auc(fpr, tpr)
    # print("auc val is " + str(auc))

    if result_path:
        RocCurveDisplay.from_predictions(
            gt.tolist(),
            heatmap.tolist(),
            name=f"ood vs id",
            color="darkorange",
        )

        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(result_path + f"/{title}.png")
    return auc


def conv_heatmap(patch_size, heatmap):
    window = torch.ones(patch_size, device=device)
    heatmap_shape = heatmap.shape
    result = F.conv2d(heatmap.view(1, 1, heatmap_shape[0], heatmap_shape[1]),
                            window.view(1, 1, patch_size[0], patch_size[1]),
                            padding='same')
    result = result[0][0]
    return result

def calculate_box_by_size_and_centers_in_xyxy_format(patch_size, center, max_size):
    curr_patch_top_left_w, curr_patch_top_left_h = max(center[1] - patch_size[1] // 2, 0), max(
        center[0] - patch_size[0] // 2, 0)
    curr_patch_bottom_right_w, curr_patch_bottom_right_h = min(center[1] + patch_size[1] // 2 + 1,
                                                               max_size[1]), min(
        center[0] + patch_size[0] // 2 + 1, max_size[0])
    curr_bbox = (curr_patch_top_left_w, curr_patch_top_left_h, curr_patch_bottom_right_w, curr_patch_bottom_right_h)
    return curr_bbox


def unravel_index(index, shape):
  """Converts a flattened index into a tuple of coordinate indices.

  Args:
    index: A tensor of flattened indices (shape: (*,)).
    shape: The target shape of the original tensor (tuple).

  Returns:
    A tensor of unraveled coordinates (shape: (*, D)).
  """

  out = []
  for dim in reversed(shape):
    out.append(index % dim)
    index = torch.div(index, dim, rounding_mode='floor')
  return tuple(reversed(out))

def extract_patches_accord_heatmap(heatmap: np.ndarray, patch_size: tuple, img_id: str, target_dir=None, threshold_percentage=85,
                                   padding=True, plot=False, title="", image_path="") -> np.ndarray:
    score_heatmap = conv_heatmap(patch_size=patch_size, heatmap=heatmap)
    threshold_value = torch.quantile(heatmap, threshold_percentage/100)
    heatmap_copy = heatmap.clone()
    curr_max_val = torch.max(heatmap_copy)
    argmax_index = torch.argmax(heatmap_copy)
    max_index_matrix = unravel_index(argmax_index, heatmap_copy.shape)
    patches_list = []
    patches_scores_conv = []
    patches_scores=[]
    heatmap_size = heatmap.shape
    mask = np.zeros(heatmap_size)
    while curr_max_val > threshold_value:
        curr_bbox = calculate_box_by_size_and_centers_in_xyxy_format(patch_size, max_index_matrix, heatmap_size)
        patches_list.append(curr_bbox)
        curr_conv_score = score_heatmap[max_index_matrix[0], max_index_matrix[1]]
        patches_scores_conv.append(curr_conv_score.detach().cpu().item())
        curr_score = heatmap[max_index_matrix[0], max_index_matrix[1]]
        patches_scores.append(curr_score.detach().cpu().item())

        curr_patch_size = curr_bbox[3] - curr_bbox[1], curr_bbox[2] - curr_bbox[0]
        if padding:
            curr_buffered_patch_top_left_w, curr_buffered_patch_top_left_h = max(max_index_matrix[1] - patch_size[1],
                                                                                 0), max(
                max_index_matrix[0] - patch_size[0], 0)
            curr_buffered_patch_bottom_right_w, curr_buffered_patch_bottom_right_h = min(
                max_index_matrix[1] + patch_size[1] + 1,
                heatmap_size[1]), min(max_index_matrix[0] + patch_size[0] + 1, heatmap_size[0])
            curr_buffered_patch_size = (curr_buffered_patch_bottom_right_h - curr_buffered_patch_top_left_h,
                                        curr_buffered_patch_bottom_right_w - curr_buffered_patch_top_left_w)
            heatmap_copy[curr_buffered_patch_top_left_h:curr_buffered_patch_bottom_right_h,
            curr_buffered_patch_top_left_w:curr_buffered_patch_bottom_right_w] = torch.zeros(
                (curr_buffered_patch_size[0], curr_buffered_patch_size[1]))
        else:
            heatmap_copy[curr_bbox[1]:curr_bbox[3],
        curr_bbox[0]:curr_bbox[2]] = torch.zeros((curr_patch_size[0], curr_patch_size[1]))

        mask[curr_bbox[1]:curr_bbox[3],
        curr_bbox[0]:curr_bbox[2]] = torch.ones((curr_patch_size[0], curr_patch_size[1]))

        curr_max_val = torch.max(heatmap_copy)
        argmax_index = torch.argmax(heatmap_copy)
        max_index_matrix = unravel_index(argmax_index, heatmap_copy.shape)
    patches_tensor = torch.tensor(patches_list, device=device)
    if plot:
        # show image with heatmap
        plt.clf()

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.imshow(heatmap.detach().cpu().numpy(), alpha=.5)
        plt.savefig(
            os.path.join(target_dir, f"{img_id}_heatmap.png"))
        # plt.colorbar()
        #
        # show image with heatmap and predicted boxes
        fig, ax = plt.subplots()
        plt.imshow(img)
        plt.imshow(heatmap.detach().cpu().numpy(), alpha=.5)
        show_predicted_boxes(patches_tensor, ax)
        plt.savefig(
            os.path.join(target_dir, f"{img_id}_heatmap_with_{title}_predicted_boxes.png"))
    return patches_tensor, mask, patches_scores_conv, patches_scores


def mask_img(imgid, dota_obj, mask):
    image_path = os.path.join(dota_obj.imagepath, imgid + '.png')
    if os.path.isfile(image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    transform_to_view_original = pth_transforms.Compose([
        pth_transforms.Resize(mask.shape),
        pth_transforms.ToTensor()
    ])
    img = transform_to_view_original(img).permute(1, 2, 0)
    masked_img = img * mask[:, :, None]
    # plt.imshow(img)
    plt.show()
    plt.clf()
    plt.imshow(masked_img)
    plt.show()

def assign_predicted_boxes_to_gt(bbox_assigner, regressor_results, gt_instances, img_id, image_path, plot=False, title="",
                                 target_dir="", visualizer=None):
    formatted_predicted_patches = RotatedBoxes(regressor_results[0].pred_instances.bboxes)
    formatted_predicted_patches = InstanceData(priors=formatted_predicted_patches)

    assign_result = bbox_assigner.assign(
        formatted_predicted_patches.to(device), gt_instances.to(device))


    if plot:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # show proposals and gts
        results = regressor_results[0]
        results.pred_instances.scores = torch.tensor([1] * len(results.pred_instances.bboxes))
        results.pred_instances.labels = torch.tensor([0] * len(results.pred_instances.bboxes))
        visualizer.add_datasample(name='result',
                                  image=img,
                                  data_sample=results.detach().cpu(),
                                  draw_gt=True,
                                  out_file=os.path.join(target_dir,
                                                        img_id + "_regressor_results.png"),
                                  wait_time=0,
                                  draw_text=False)
    return assign_result

def extract_and_save_single_bbox(poly, image, class_name, output_dir, name, logger, hashmap_locations=None, anomaly_scores=None):
    poly = poly.detach().cpu()
    if isinstance(hashmap_locations, dict):
        hashmap_locations[name] = {}
        hashmap_locations[name]["poly"] = poly.cpu().numpy().tolist()
        hashmap_locations[name]["anomaly_score"] = anomaly_scores[0]
        hashmap_locations[name]["anomaly_score_conv"] = anomaly_scores[1]
    poly_qbox_rep = rbox2qbox(poly)
    points = np.array(poly_qbox_rep.numpy() , dtype="float32").reshape(4, 2)
    width, height = poly[2], poly[3]
    mapping_points = np.float32([[0, 0],
                        [0, height],
                        [width , height ],
                        [width , 0]])
    M = cv2.getPerspectiveTransform(points, mapping_points)
    patch = cv2.warpPerspective(image, M, (int(width), int(height))
                                , flags=cv2.INTER_LINEAR)
    # im = cv2.warpPerspective(image, M, (int(512), int(512))
    #                          , flags=cv2.INTER_LINEAR)

    try:
        patch_path = os.path.join(output_dir, class_name, name + '.png')

        patch_img = Image.fromarray(patch)
        patch_img.save(patch_path)
    except Exception as error:
        logger.info(f"[FILE WRITE ERROR] The file {name} can't be saved, {error} \n")
        return

def extract_and_save_bboxes(labels_names, predicted_boxes, predicted_boxes_labels, image, output_dir, img_id, logger,
                            hashmap_locations=None, scores_dict=None):
    unique_labels = torch.unique(predicted_boxes_labels)
    for unique_label in unique_labels:
        label = "background" if unique_label == -1 else labels_names[unique_label]
        label_dir = os.path.join(output_dir, label)
        os.makedirs(os.path.join(label_dir), exist_ok=True)
    for box_ind, box in enumerate(predicted_boxes):
        class_name = "background" if predicted_boxes_labels[box_ind].item() == -1 else labels_names[predicted_boxes_labels[box_ind].item()]
        scores=None
        if isinstance(scores_dict, dict):
            curr_anomaly_score = scores_dict["patches_scores"][box_ind]
            curr_anomaly_score_conv = scores_dict["patches_scores_conv"][box_ind]
            scores = (curr_anomaly_score, curr_anomaly_score_conv)
        extract_and_save_single_bbox(poly=box, image=image, class_name=class_name, output_dir=output_dir,
                                     name=f"{img_id}_{box_ind}", logger=logger, hashmap_locations=hashmap_locations,
                                     anomaly_scores=scores)


def assign_predicted_boxes_to_gt_boxes_and_save(bbox_assigner, regressor_results, gt_instances, img_id, image_path,
                                                labels_names, extract_bbox_path, logger, plot=False,
                                                title="", target_dir="", visualizer=None, train=True, val=False,
                                                hashmap_locations=None, scores_dict=None):
    assign_result = assign_predicted_boxes_to_gt(bbox_assigner=bbox_assigner, regressor_results=regressor_results,
                                                gt_instances=gt_instances, img_id=img_id, image_path=image_path,
                                                plot=plot, title=title, target_dir=target_dir, visualizer=visualizer)
    # collect predicted_boxes labels and cut it out the images and save it in an appropriate folders
    if train:
        # sample bgs
        assigned_bboxes_indices_foreground = torch.nonzero(assign_result.gt_inds > 0)
        assigned_bboxes_indices_background = torch.nonzero(assign_result.gt_inds == 0)
        assigned_bboxes_indices_background = assigned_bboxes_indices_background[torch.randperm(
            len(assigned_bboxes_indices_background))[:int(0.1 * len(assigned_bboxes_indices_background))]]
        assigned_bboxes_indices = torch.cat((assigned_bboxes_indices_foreground, assigned_bboxes_indices_background),
                                            dim=0)
        predicted_boxes_labels = assign_result.labels[assigned_bboxes_indices]
        predicted_boxes = regressor_results[0].pred_instances.bboxes[assigned_bboxes_indices.squeeze(dim=1)]
    elif val:
        # sample not fgs
        assigned_bboxes_indices_background = torch.nonzero(assign_result.labels == -1)
        assigned_bboxes_indices_foreground = torch.nonzero(assign_result.labels != -1)
        assigned_bboxes_indices_background = assigned_bboxes_indices_background[torch.randperm(
            len(assigned_bboxes_indices_background))[:int(0.1 * len(assigned_bboxes_indices_background))]]
        assigned_bboxes_indices = torch.cat(
            (assigned_bboxes_indices_foreground, assigned_bboxes_indices_background),
            dim=0)
        predicted_boxes_labels = assign_result.labels[assigned_bboxes_indices]
        predicted_boxes = regressor_results[0].pred_instances.bboxes[assigned_bboxes_indices.squeeze(dim=1)]
    else:
        predicted_boxes_labels = assign_result.labels
        predicted_boxes = regressor_results[0].pred_instances.bboxes

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    extract_and_save_bboxes(labels_names=labels_names,
                            predicted_boxes=predicted_boxes,
                            predicted_boxes_labels=predicted_boxes_labels,
                            image=img, output_dir=extract_bbox_path, img_id=img_id, logger=logger,
                            hashmap_locations=hashmap_locations, scores_dict=scores_dict)

def assign_predicted_boxes_to_gt_boxes_and_save_val_stage(bbox_assigner, predicted_boxes, data_batch, patch_size, img_id,
                                                        labels_names, dota_obj, heatmap, extract_bbox_path, logger,
                                                          plot=False, title="", target_dir="", is_val=False):
    assign_result, _ = assign_predicted_boxes_to_gt(bbox_assigner=bbox_assigner, predicted_boxes=predicted_boxes,
                                                    data_batch=data_batch, patch_size=patch_size, img_id=img_id,
                                                    dota_obj=dota_obj, heatmap=heatmap,
                                                    filter_instances_accord_size=False,
                                                    plot=plot, title=title, target_dir=target_dir)
    image = dota_obj.loadImgs(img_id)
    image = cv2.resize(image[0], heatmap.shape)
    predicted_boxes_labels = assign_result.labels
    # In the validation case we sample the background
    if is_val:
        assigned_bboxes_indices_foreground = torch.nonzero(assign_result.gt_inds > 0)
        assigned_bboxes_indices_background = torch.nonzero(assign_result.gt_inds == 0)
        assigned_bboxes_indices_background = assigned_bboxes_indices_background[torch.randperm(
            len(assigned_bboxes_indices_background))[:int(0.01 * len(assigned_bboxes_indices_background))]]
        assigned_bboxes_indices = torch.cat((assigned_bboxes_indices_foreground, assigned_bboxes_indices_background),
                                            dim=0)
        predicted_boxes = predicted_boxes[assigned_bboxes_indices].squeeze(dim=1)
        predicted_boxes_labels = assign_result.labels[assigned_bboxes_indices]
    extract_and_save_bboxes(labels_names=labels_names,
                            predicted_boxes=predicted_boxes,
                            predicted_boxes_labels=predicted_boxes_labels,
                            image=image, output_dir=extract_bbox_path, img_id=img_id, logger=logger)


def assign_predicted_boxes_to_gt_and_calc_performance(bbox_assigner, predicted_boxes, data_batch, img_id, dota_obj, heatmap,
                                                      patch_size, plot=False, title=""):
    # TODO: refer the case in which patch size has odd sizes
    assign_result, filtered_instances_indices = assign_predicted_boxes_to_gt(bbox_assigner, predicted_boxes, data_batch, patch_size, img_id, dota_obj, heatmap,
                                 plot=plot, title=title)
    dt_labels = assign_result.labels
    dt_match = assign_result.gt_inds
    return filtered_instances_indices, dt_labels, dt_match

def assign_predicted_boxes_to_gt_accord_intersection(bbox_assigner, predicted_boxes, data_batch, img_id, dota_obj, heatmap,
                                                     patch_size, additional_patches_sizes=((11,21), (21,11)),
                                                     plot=False, title="", target_dir=""):
    # TODO: refer the case in which patch size has odd sizes
    patch_diag = np.sqrt(np.square(patch_size[0]) + np.square(patch_size[1]))
    formatted_predicted_patches = HorizontalBoxes(predicted_boxes)
    formatted_predicted_patches = InstanceData(priors=formatted_predicted_patches)

    gt_instances = unpack_gt_instances(data_batch['data_samples'])
    batch_gt_instances, batch_gt_instances_ignore, _ = gt_instances
    diags = torch.sqrt(
        torch.square(batch_gt_instances[0].bboxes.widths) + torch.square(batch_gt_instances[0].bboxes.heights))
    filtered_instances_indices = np.where((diags<(patch_diag*1.2)) & (diags>(patch_diag/1.2)))[0]
    filtered_instances = batch_gt_instances[0][filtered_instances_indices]
    overlaps = bbox_assigner.iou_calculator(filtered_instances.bboxes.to(device), formatted_predicted_patches.priors.to(device))
    tp = torch.sum(torch.max(overlaps, axis=1).values>0).item()
    found_gt_indices = torch.nonzero(torch.max(overlaps, axis=1).values>0)

    # If there is intersection taking square and rectangle boxes
    intersected_predicted_bboxes_indices = torch.max(overlaps, axis=1).indices
    intersected_predicted_bboxes_centers = formatted_predicted_patches[intersected_predicted_bboxes_indices].priors.centers
    additional_hypothesis = []
    for additional_patch_size in additional_patches_sizes:
        for center in intersected_predicted_bboxes_centers:
            center = center.int().tolist()
            curr_bbox=calculate_box_by_size_and_centers_in_xyxy_format(patch_size=additional_patch_size, center=(center[1],center[0]), max_size=heatmap.shape)
            additional_hypothesis.append(curr_bbox)
    additional_hypothesis_tensor = torch.tensor(additional_hypothesis).to(torch.int64)
    predicted_boxes = torch.cat((predicted_boxes, additional_hypothesis_tensor))
    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.axis('off')
        original_img = cv2.imread(os.path.join(dota_obj.imagepath, img_id + '.png'))
        plt.title(f'heatmap')
        extent = 0, heatmap.shape[0], 0, heatmap.shape[0]
        ax1.imshow(np.flipud(original_img), extent=extent)
        ax1.imshow(np.flipud(heatmap), alpha=.5, extent=extent)
        # fig.colorbar(ax1.images[-1], ax=ax1)
        ax2 = fig.add_subplot(1, 2, 2)
        plt.title(f'ground truth')
        ax2.imshow(np.flipud(original_img), extent=extent)
        ax2.axis('off')
        gt_instances = unpack_gt_instances(data_batch['data_samples'])
        batch_gt_instances, batch_gt_instances_ignore, _ = gt_instances
        instances = batch_gt_instances[0]
        show_gts(instances['bboxes'].tensor, ax2)
        plt.savefig(
            os.path.join(target_dir, f"{img_id}_with_heatmap_and_gt.png"))
    return predicted_boxes, filtered_instances_indices, tp



def assign_predicted_boxes_to_gt_accord_intersection_and_calc_performance(bbox_assigner, predicted_boxes, data_batch, all_labels, img_id, dota_obj, heatmap,
                                                                          patch_size, additional_patches_sizes=((11,21), (21,11)), plot=False, title=""):
    predicted_boxes, filtered_instances_indices, tp = assign_predicted_boxes_to_gt_accord_intersection(bbox_assigner, predicted_boxes, data_batch, all_labels, img_id, dota_obj, heatmap,
                                                                          patch_size, additional_patches_sizes, plot, title)

    gt = len(np.unique(filtered_instances_indices))
    fp = len(predicted_boxes) - tp
    print(f"Discovered {tp} bbox out of {gt} for img {img_id}\n")
    return tp, gt, fp

def show_gts(gt_boxes, ax):
    if gt_boxes is not None:
        gt_boxes_xy_format = [[(box[0].item(), box[1].item()), (box[2].item(), box[1].item()),
                               (box[2].item(), box[3].item()), (box[0].item(), box[3].item())] for box in
                              gt_boxes]
        polygons = []
        color = []
        for box_ind, box in enumerate(gt_boxes_xy_format):
            c = 'g'
            color.append(c)
            polygons.append(Polygon(box))
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)

def show_predicted_boxes(predicted_boxes, ax, found_gt_indices=None, gt_boxes=None):
    predicted_boxes_xy_format = [[(box[0].item(), box[1].item()), (box[2].item(), box[1].item()),
                                  (box[2].item(), box[3].item()), (box[0].item(), box[3].item())] for box in
                                 predicted_boxes]
    polygons = []
    color = []
    for box in predicted_boxes_xy_format:
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        color.append(c)
        polygons.append(Polygon(box))
    p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)
    if gt_boxes is not None:
        gt_boxes_xy_format = [[(box[0].item(), box[1].item()), (box[2].item(), box[1].item()),
                               (box[2].item(), box[3].item()), (box[0].item(), box[3].item())] for box in
                              gt_boxes]
        polygons = []
        color = []
        for box_ind, box in enumerate(gt_boxes_xy_format):
            c = 'r' if box_ind not in found_gt_indices else 'g'
            color.append(c)
            polygons.append(Polygon(box))
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)

def calculate_performance_measures(gt, scores):
    gt[gt>=0] = 1
    gt[gt<0] = 0
    ap = average_precision_score(gt, scores)
    auc = roc_auc_score(gt, scores)
    return ap, auc

def plot_comparison_graphs(mode="attentions", vit_architecture="small"):
    results_dir = "/home/shoval/Documents/Repositories/single-image-bg-detector/results/normalized_gsd/train/"
    dino_vit_dir = os.path.join(results_dir, "dino_vit")
    dino_mc_vit_dir = os.path.join(results_dir, "dino_mc_vit")
    dino_vit_base_dir = os.path.join(results_dir, "dino_vit_base")

    if mode == "representations":
        pass
    elif mode == "attentions":
        if vit_architecture == "small":
            dino_vit_atten_dir = os.path.join(dino_vit_dir, "class_token_self_attention")
            dino_mc_vit_atten_dir = os.path.join(dino_mc_vit_dir, "class_token_self_attention")
            dino_vit, max_ap_dino_vit_head_num, max_auc_dino_vit_head_num = load_performance_measures(dino_vit_atten_dir)
            dino_mc, max_ap_dino_mc_head_num, max_auc_dino_mc_head_num = load_performance_measures(dino_mc_vit_atten_dir)
            if dino_vit[max_ap_dino_vit_head_num]['ap'] > dino_mc[max_ap_dino_mc_head_num]['ap']:
                print(f"dino_vit head number {max_ap_dino_vit_head_num} yields the best ap - {dino_vit[max_ap_dino_vit_head_num]['ap']}\n" )
            else:
                print(f"dino_mc head number {max_ap_dino_mc_head_num} yields the best ap - { dino_mc[max_ap_dino_mc_head_num]['ap']}\n")
            if dino_vit[max_auc_dino_vit_head_num]['auc'] > dino_mc[max_auc_dino_mc_head_num]['auc']:
                print(f"dino_vit head number {max_auc_dino_vit_head_num} yields the best auc - {dino_vit[max_auc_dino_vit_head_num]['auc']}\n")
            else:
                print(f"dino_mc head number {max_auc_dino_mc_head_num} yields the best auc - {dino_mc[max_auc_dino_mc_head_num]['auc']}\n")

            plt.clf()
            plt.plot([head_num for head_num, values in dino_vit.items()], [values["ap"] for head_num, values in dino_vit.items()], 'bo', label="dino_vit_ap")
            plt.plot([head_num for head_num, values in dino_mc.items()], [values["ap"] for head_num, values in dino_mc.items()], 'ro', label="dino_mc_ap")
            plt.plot([head_num for head_num, values in dino_vit.items()], [values["auc"] for head_num, values in dino_vit.items()], 'go', label="dino_vit_auc")
            plt.plot([head_num for head_num, values in dino_mc.items()], [values["auc"] for head_num, values in dino_mc.items()], 'yo', label="dino_mc_auc")
            plt.legend()
            plt.grid()
            plt.title("Vit small based models - AP and AuC comparison")
            plt.savefig(os.path.join(results_dir, "vit_small_arch_attention_comparison.png"))



def load_performance_measures(dir):
    results = {}
    max_ap_head_num = 0
    max_auc_head_num = 0
    max_ap = 0
    max_auc = 0
    for filename in os.listdir(dir):
        if filename.endswith(".json"):
            file_path = os.path.join(dir, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
            head_num=int(filename.split(".")[0].split("_")[-1])
            results[head_num] = {}
            results[head_num]["ap"] = json_data["mean_ap"]
            results[head_num]["auc"] = json_data["mean_auc"]
            if results[head_num]["ap"] > max_ap:
                max_ap = results[head_num]["ap"]
                max_ap_head_num=head_num
            if results[head_num]["auc"] > max_auc:
                max_auc = results[head_num]["auc"]
                max_auc_head_num=head_num

    return results, max_ap_head_num, max_auc_head_num

if __name__ == '__main__':
    plot_comparison_graphs()



